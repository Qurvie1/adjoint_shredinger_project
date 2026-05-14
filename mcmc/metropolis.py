# mcmc/metropolis.py
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path

class MetropolisHastingsSampler(BaseMCMCSampler):
    """MCMC-семплер с использованием алгоритма Метрополиса-Гастингса."""
    
    def __init__(self, energy_fn, cv_fn=None, dim_cv=2, proposal_scale=0.1, 
                 kbt=0.6, initial_cv=None):
        """
        Args:
            energy_fn: callable, принимает CV и возвращает энергию E(cv)
            cv_fn: callable, преобразует позиции в CV (если нужен полный атомный вывод)
            dim_cv: размерность CV-пространства
            proposal_scale: шаг случайного блуждания (std нормального распределения)
            kbt: тепловая энергия k_B * T
            initial_cv: начальная точка в CV-пространстве
        """
        self.energy_fn = energy_fn
        self.cv_fn = cv_fn
        self.dim_cv = dim_cv
        self.proposal_scale = proposal_scale
        self.kbt = kbt
        self.current_cv = initial_cv if initial_cv is not None else torch.zeros(dim_cv)
        self.current_energy = self.energy_fn(self.current_cv)
        self.samples = []  # список для хранения сэмплов
        
    def step(self):
        """Один шаг цепи Маркова."""
        # Предложение нового состояния
        proposal = self.current_cv + torch.randn(self.dim_cv) * self.proposal_scale
        proposal_energy = self.energy_fn(proposal)
        
        # Критерий принятия (для большего ускорения векторизуйте эту часть)
        log_accept = -(proposal_energy - self.current_energy) / self.kbt
        if torch.log(torch.rand(1)) < log_accept:
            self.current_cv = proposal
            self.current_energy = proposal_energy
            
        return self.current_cv.clone()
    
    def run(self, n_steps, save_interval=100, progress_bar=True):
        """Запускает цепочку и сохраняет сэмплы."""
        iterator = range(n_steps)
        if progress_bar:
            iterator = tqdm(iterator, desc="MCMC Sampling")
            
        for i in iterator:
            cv = self.step()
            if i % save_interval == 0:
                self.samples.append({
                    'step': i,
                    'cv': cv.clone(),
                    'pos': None  # если cv_fn не задан, сохраняем только CV
                })
                
    def save_samples(self, filepath):
        """Сохраняет результат в формате, совместимом с WT-ASBS."""
        # Извлекаем все CV-координаты
        all_cvs = torch.stack([s['cv'] for s in self.samples])
        torch.save({'cv': all_cvs, 'pos': None}, Path(filepath))