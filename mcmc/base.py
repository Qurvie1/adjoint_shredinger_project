import torch
from abc import ABC, abstractmethod

class BaseMCMCSampler(ABC):
    @abstractmethod
    def step(self):
        """Выполняет один шаг MCMC и возвращает {'pos': координаты, 'cv': значения}"""
        pass

    @abstractmethod
    def run(self, n_steps, save_interval=100):
        """Запускает цепочку и сохраняет результаты в буфер"""
        pass

    @abstractmethod
    def save_samples(self, filepath):
        """Сохраняет результат в формате совместимом с WT-ASBS"""
        pass