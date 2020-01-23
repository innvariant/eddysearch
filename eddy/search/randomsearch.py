import numpy as np
from eddy.strategy import SearchStrategy


class RandomUniformSearch(SearchStrategy):
    def has_finished(self) -> bool:
        return False  # continue with as many evaluations as possible

    def step(self):
        if self._objective is None:
            raise ValueError('Objective is none. Have you forgot to call start()?')

        self._objective(np.random.uniform(self._lower, self._upper))

    def end(self):
        pass

    def __str__(self):
        return 'RandomSearch(dim=%s)' % (self._dimensions)
