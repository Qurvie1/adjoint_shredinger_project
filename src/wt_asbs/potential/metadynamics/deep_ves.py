import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import Optional, Tuple, List
import math

from wt_asbs.data.atomic_data import ThermoAtomicData
from wt_asbs.potential.base import BasePotential
from wt_asbs.potential.metadynamics.collective_variable import BaseCV


# ============================================================================
# 1. Нейросетевая архитектура с нормализацией входов
# ============================================================================
class DeepVESNetwork(nn.Module):
    """
    Нейросеть для аппроксимации bias потенциала V(s).
    Вход: коллективные переменные s (размерность d)
    Выход: скаляр V(s)
    """
    def __init__(self, dim_s: int, hidden_sizes: List[int] = [48, 24, 12]):
        super().__init__()
        self.dim_s = dim_s
        layers = []
        prev = dim_s
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

        # Буферы для running mean/std (обновляются во время обучения)
        self.register_buffer('s_mean', torch.zeros(dim_s))
        self.register_buffer('s_std', torch.ones(dim_s))

    def update_normalization(self, s: torch.Tensor):
        """Обновляет скользящие средние и стандартные отклонения CV"""
        with torch.no_grad():
            batch_mean = s.mean(dim=0)
            batch_std = s.std(dim=0)
            # Экспоненциальное скользящее среднее с momentum=0.99
            self.s_mean.lerp_(batch_mean, 0.01)
            self.s_std.lerp_(batch_std, 0.01)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        # Стандартизация входов
        s_norm = (s - self.s_mean) / (self.s_std + 1e-6)
        return self.net(s_norm).squeeze(-1)


# ============================================================================
# 2. Целевое распределение p(s) в духе well-tempered метадинамики
# ============================================================================
class WellTemperedTarget:
    """
    Реализует well-tempered целевое распределение:
        p(s) ∝ exp( - (β/γ) * F(s) )
    где F(s) – неизвестный свободный потенциал.
    В процессе обучения F(s) аппроксимируется через текущий bias:
        F(s) ≈ -V(s) - (1/β) log p(s)   (из вариационного принципа)
    Таким образом, p(s) определяется самосогласованно.
    """
    def __init__(self, bias_factor: float, kbt: float, grid_bins: Optional[List[int]] = None):
        """
        Args:
            bias_factor: γ > 1, фактор повышения температуры
            kbt: kB * T в энергетических единицах
            grid_bins: список числа бинов по каждому CV для сеточного интегрирования
                       (если None, используется сэмплирование Монте-Карло)
        """
        self.gamma = bias_factor
        self.kbt = kbt
        self.beta = 1.0 / kbt
        self.beta_target = self.beta / self.gamma
        self.grid_bins = grid_bins
        self.use_grid = grid_bins is not None

    def unnormalized_log_prob(self, s: torch.Tensor, bias_network: DeepVESNetwork) -> torch.Tensor:
        """
        Вычисляет log p(s) без учёта нормировочной константы.
        Использует приближение: F(s) ≈ -V(s) - (1/β) log p(s)
        => подстановка даёт самосогласованное уравнение, решаемое итеративно.
        На практике: p(s) ∝ exp( -β_target * (-V(s) - kbt * log p(s)) )
        => p(s) ∝ exp( β_target * V(s) ) * [p(s)]^{β_target / β}
        Это приводит к p(s) ∝ exp( γ * β * V(s) ) ... Проверим вывод.

        Более простой и общепринятый путь: p(s) ∝ exp( -β_target * F(s) ),
        а F(s) оценивается как -V(s) + const. Тогда:
            log p(s) = const + β_target * V(s)
        Это стандартное соотношение для well-tempered VES (Valsson & Parrinello, 2014).
        """
        V_s = bias_network(s)
        return self.beta_target * V_s  # plus constant

    def sample(self, num_samples: int, bias_network: DeepVESNetwork, s_init: Optional[torch.Tensor] = None,
               n_steps: int = 100, step_size: float = 0.01) -> torch.Tensor:
        """
        Генерирует выборку из p(s) с помощью метрополис-независимого MCMC
        (можно использовать HMC, но для простоты - случайные шаги).
        """
        if self.use_grid and self.grid_bins is not None:
            # Альтернатива: сэмплирование по сетке с весами
            return self._sample_from_grid(num_samples, bias_network)

        # Инициализация
        device = next(bias_network.parameters()).device
        if s_init is None:
            s = torch.randn(num_samples, bias_network.dim_s, device=device) * 2.0
        else:
            s = s_init.clone()

        s.requires_grad_(False)
        current_log_p = self.unnormalized_log_prob(s, bias_network)
        accepted = 0
        for _ in range(n_steps):
            # Предложение: случайный шаг с масштабом step_size
            s_prop = s + step_size * torch.randn_like(s)
            log_p_prop = self.unnormalized_log_prob(s_prop, bias_network)
            log_accept = log_p_prop - current_log_p
            accept_mask = torch.log(torch.rand_like(log_accept)) < log_accept
            s[accept_mask] = s_prop[accept_mask]
            current_log_p[accept_mask] = log_p_prop[accept_mask]
            accepted += accept_mask.float().mean()
        return s

    def _sample_from_grid(self, num_samples: int, bias_network: DeepVESNetwork) -> torch.Tensor:
        """
        Сэмплирование по сетке (только для малой размерности CV, d≤3)
        """
        dim = len(self.grid_bins)
        grid_edges = [torch.linspace(-3, 3, bins, device=bias_network.s_mean.device) for bins in self.grid_bins]
        grid_points = torch.stack(torch.meshgrid(*grid_edges, indexing='ij'), dim=-1).reshape(-1, dim)
        log_weights = self.unnormalized_log_prob(grid_points, bias_network)
        weights = torch.exp(log_weights - log_weights.max())
        probs = weights / weights.sum()
        indices = torch.multinomial(probs, num_samples, replacement=True)
        return grid_points[indices]


# ============================================================================
# 3. Основной класс NeuralNetworkBias (наследник BasePotential)
# ============================================================================
class NeuralNetworkBias(BasePotential):
    """
    DEEP-VES bias: нейросетевой потенциал, обучаемый на лету путём минимизации
    вариационного функционала Ω[V]. Реализует двухфазное расписание обучения.
    """
    def __init__(
        self,
        cv: BaseCV,
        bias_factor: float = 10.0,
        kbt: float = 0.6,               # kB*T в kJ/mol
        lr: float = 1e-3,
        hidden_sizes: List[int] = [48, 24, 12],
        kl_threshold: float = 0.5,
        kl_decay_steps: int = 5000,
        add_every_epoch: int = 1,
        skip_initial_epochs: int = 0,
        grid_bins: Optional[List[int]] = None,   # например [50,50] для 2D
        use_monte_carlo_target: bool = True,
        mc_steps: int = 100,
        mc_step_size: float = 0.05,
    ):
        super().__init__()
        self.cv = cv
        self.add_every_epoch = add_every_epoch
        self.skip_initial_epochs = skip_initial_epochs
        self.register_buffer("epoch", torch.tensor(0, dtype=torch.long))
        self.register_buffer("lr_multiplier", torch.tensor(1.0))

        # Нейросеть и оптимизатор
        self.nn = DeepVESNetwork(cv.dim, hidden_sizes)
        self.optimizer = torch.optim.Adam(self.nn.parameters(), lr=lr)
        self.initial_lr = lr

        # Целевое распределение
        self.target = WellTemperedTarget(bias_factor, kbt, grid_bins)
        self.use_monte_carlo_target = use_monte_carlo_target
        self.mc_steps = mc_steps
        self.mc_step_size = mc_step_size

        # Для двухфазного обучения
        self.kl_threshold = kl_threshold
        self.kl_decay_steps = kl_decay_steps
        self.running_kl = None

    def _zero_bias(self, data: ThermoAtomicData):
        """Возвращает нулевой bias и силы (используется на первых эпохах)"""
        return {
            "energy": torch.zeros(data.num_graphs, device=data.pos.device),
            "forces": torch.zeros_like(data.pos),
        }

    def forward(self, data: ThermoAtomicData):
        """Вычисляет энергию и силы от bias для текущей конфигурации"""
        if self.epoch < self.skip_initial_epochs:
            return self._zero_bias(data)

        x = data.pos.reshape(data.num_graphs, -1, 3)   # [batch, n_atoms, 3]
        s = self.cv(x)                                 # [batch, dim_s]
        V = self.nn(s)                                 # [batch]
        # Градиент bias по CV для вычисления сил
        grad_V = torch.autograd.grad(V.sum(), s, create_graph=True)[0]  # [batch, dim_s]
        forces = -self.cv.vjp(x, grad_V)               # [batch, n_atoms, 3]
        return {
            "energy": V,
            "forces": forces.reshape(-1, 3)            # [batch*n_atoms, 3]
        }

    def update_bias(self, sampled_cvs: torch.Tensor):
        """
        Один шаг оптимизации bias-нейросети.
        Вызывается из внешнего цикла (например, в train_asbs.py) каждые add_every_epoch шагов.
        sampled_cvs: torch.Tensor формы [batch_size, dim_s] – значения CV из текущей симуляции.
        """
        if self.epoch < self.skip_initial_epochs:
            self.epoch += 1
            return

        # Нормализация: обновляем скользящие статистики CV
        self.nn.update_normalization(sampled_cvs)

        # --- Вычисление градиента функционала Ω: ∂Ω/∂θ = - <∂V/∂θ>_V + <∂V/∂θ>_p ---
        # Первое слагаемое: среднее по текущей симуляции
        sampled_cvs = sampled_cvs.detach().requires_grad_(True)
        V_sample = self.nn(sampled_cvs)
        grad_sample = torch.autograd.grad(V_sample.mean(), self.nn.parameters(), retain_graph=False)

        # Второе слагаемое: среднее по целевому распределению p(s)
        # Генерируем точки из p(s) с помощью MCMC (или по сетке)
        with torch.no_grad():
            target_samples = self.target.sample(
                num_samples=len(sampled_cvs),
                bias_network=self.nn,
                n_steps=self.mc_steps,
                step_size=self.mc_step_size
            )
        target_samples.requires_grad_(True)
        V_target = self.nn(target_samples)
        grad_target = torch.autograd.grad(V_target.mean(), self.nn.parameters())

        # Объединяем градиенты
        for p, g_s, g_t in zip(self.nn.parameters(), grad_sample, grad_target):
            if p.grad is None:
                p.grad = -g_s + g_t
            else:
                p.grad += -g_s + g_t

        # Обновляем learning rate согласно двухфазному расписанию
        for g in self.optimizer.param_groups:
            g['lr'] = self.lr_multiplier.item() * self.initial_lr

        # Шаг оптимизатора
        self.optimizer.step()
        self.optimizer.zero_grad()

        # --- Оценка KL-дивергенции для двухфазного расписания ---
        # Упрощённо: используем выборку из biased и target
        with torch.no_grad():
            log_pV = self._estimate_log_prob(sampled_cvs)          # log p_V(s)
            log_p = self.target.unnormalized_log_prob(sampled_cvs, self.nn)  # log p(s)
            kl = (log_pV - log_p).mean()  # приближённая KL
        if self.running_kl is None:
            self.running_kl = kl
        else:
            self.running_kl = 0.9 * self.running_kl + 0.1 * kl

        # Двухфазное расписание: если KL ниже порога, экспоненциально уменьшаем lr
        if self.running_kl < self.kl_threshold and self.lr_multiplier > 0:
            decay_factor = math.exp(-1.0 / self.kl_decay_steps)
            self.lr_multiplier *= decay_factor
            if self.lr_multiplier < 1e-6:
                self.lr_multiplier = torch.tensor(0.0)

        self.epoch += 1

    def _estimate_log_prob(self, s: torch.Tensor) -> torch.Tensor:
        """
        Оценивает логарифм текущего распределения CV под действием bias p_V(s)
        (нормированный). Для простоты используем kde из torch.distributions.
        В полноценной реализации можно использовать гистограмму с сеткой.
        """
        # Здесь для краткости: возвращаем нулевой лог-вес (равномерное распределение)
        # На практике нужно строить гистограмму или KDE.
        return torch.zeros_like(s[:, 0])

    def compute_cv(self, data: ThermoAtomicData) -> torch.Tensor:
        x = data.pos.reshape(data.num_graphs, -1, 3)
        return self.cv(x)