import numpy as np

from eddysearch.strategy import SearchStrategy


class RandomSearch(SearchStrategy):
    def has_finished(self) -> bool:
        return False  # continue with as many evaluations as possible

    def sample_random(self):
        raise NotImplementedError(
            "Your random search strategy has to specify how to draw elements within the R^n space."
        )

    def step(self):
        if self._objective is None:
            raise ValueError("Objective is none. Have you forgot to call start()?")

        self._objective(self.sample_random())

    def end(self):
        pass

    def __str__(self):
        return "RandomSearch(dim=%s)" % (self._dimensions)


class RandomUniformSearch(RandomSearch):
    def sample_random(self):
        return np.random.uniform(self._lower, self._upper)

    def __str__(self):
        return "RandomUniformSearch(dim=%s)" % (self._dimensions)
