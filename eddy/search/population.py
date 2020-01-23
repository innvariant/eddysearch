from eddy.strategy import SearchStrategy


class PopulationSearch(SearchStrategy):
    def __init__(self, population_size: int=10, num_generations: int=10, **kwargs):
        self._population_size = population_size
        self._num_generations = num_generations

        super().__init__(**kwargs)

    @property
    def population_size(self) -> int:
        return self._population_size

    @property
    def generations(self) -> int:
        return self._num_generations

    def __str__(self):
        return 'PopulationSearch(dim=%s, pop_size=%s)' % (self._dimensions, self._population_size)
