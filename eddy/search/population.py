import numpy as np

from eddy.objective import Objective
from eddy.strategy import SearchStrategy


class PopulationSearch(SearchStrategy):
    def __init__(self, *args, population_size: int=10, num_generations: int=10, **kwargs):
        self._population_size = population_size
        self._num_generations = num_generations

        super().__init__(*args, **kwargs)

    @property
    def population_size(self) -> int:
        return self._population_size

    @property
    def generations(self) -> int:
        return self._num_generations

    def __str__(self):
        return 'PopulationSearch(dim=%s, pop_size=%s)' % (self._dimensions, self._population_size)


class CMAESSearch(PopulationSearch):
    # TODO in progress
    def __init__(self, *args, mu_important: int = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._mu_important = int(self.population_size/2) if mu_important is None else int(mu_important)

    def _sample_random(self):
        return np.random.multivariate_normal(self._mean, self._covariance)

    def _cached_evaluation(self, member):
        if member not in self._evaluated_members:
            self._evaluated_members[member] = self.objective(self._map_member_to_euclidean(member))
        return self._evaluated_members[member]

    def _map_member_to_euclidean(self, member):
        return np.fromstring(member)

    def start(self, objective: Objective):
        super().start(objective)
        self._current_population = 0
        self._mean = np.mean([self._lower, self._upper], axis=0)
        self._covariance = np.eye(self.num_dimensions)
        self._evaluated_members = {}

        self._population = {self._sample_random().tostring() for _ in range(self._population_size)}

    def step(self):
        members = np.copy([np.fromstring(mem) for mem in self._population])
        covariance = np.cov(members.T)
        print(covariance)
        population_eval = {mem: self._cached_evaluation(mem) for mem in self._population}

        # Sorted population by evaluation. Largest/worst member first
        sorted_population = sorted(population_eval, key=lambda x: population_eval[x], reverse=True)

        for idx in range(self._mu_important):
            self._population.remove(sorted_population[idx])

        # Move mean to better solutions
        self._mean = np.mean([np.fromstring(mem) for mem in self._population], axis=0)

        for _ in range(self._mu_important):
            self._population.add(self._sample_random().tostring())

    def end(self):
        pass

    def has_finished(self):
        return False  # never stop

    def __str__(self):
        return 'CMAESSearch(dim=%s, pop_size=%s)' % (self._dimensions, self._population_size)
