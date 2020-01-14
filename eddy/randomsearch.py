import numpy as np
from .strategy import Strategy


class RandomUniformSearch(Strategy):
    def __init__(self, dimensions: int, lower: np.ndarray, upper: np.ndarray):
        """

        :param dimensions: Integer with dimensions of objective
        :param lower: Lower value for each dimension. Shape (d,)
        :param upper: Upper value for each dimension. Shape (d,)
        """
        self._dimensions = dimensions
        assert len(lower.shape) == 1
        assert lower.shape[0] == dimensions
        assert len(upper.shape) == 1
        assert upper.shape[0] == dimensions

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

        self._objective(np.random.uniform(self._lower, self._upper))

    def end(self):
        pass
