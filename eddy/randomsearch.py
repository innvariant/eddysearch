import numpy as np
from .strategy import Strategy


class RandomUniformSearch(Strategy):
    def __init__(self, dimensions : int, lower : float, upper : float):
        self._dimensions = dimensions
        self._lower = lower
        self._upper = upper
        self._objective = None

    def has_finished(self) -> bool:
        return False  # continue with as many evaluations as possible

    def start(self, objective):
        self._objective = objective

    def step(self):
        if self._objective is None:
            raise ValueError('Objective is none. Have you forgot to call start()?')

        self._objective(np.random.uniform([self._lower] * self._dimensions, [self._upper] * self._dimensions))

    def end(self):
        pass
