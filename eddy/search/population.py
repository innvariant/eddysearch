import numpy as np

from eddy.objective import Objective
from eddy.search.randomsearch import RandomUniformSearch
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

    def _encode_member(self, member):
        return member.tostring()

    def _decode_member(self, encoded_member):
        return np.fromstring(encoded_member)

    def _cached_evaluation(self, member):
        if member not in self._evaluated_members:
            self._evaluated_members[member] = self.objective(self._decode_member(member))
        return self._evaluated_members[member]

    def start(self, objective: Objective):
        super().start(objective)
        self._current_generation = 0
        self._evaluated_members = {}
        self._population = {self.sample_random().tostring() for _ in range(self._population_size)}

    def step(self):
        self._current_generation += 1

        population_eval = {mem: self._cached_evaluation(mem) for mem in self._population}

        # Sorted population by evaluation. Largest/worst member first
        sorted_population = sorted(population_eval, key=lambda x: population_eval[x], reverse=True)

        self._population = self.derive_population(sorted_population, population_eval)

    def derive_population(self, sorted_population, evaluated_population):
        raise NotImplementedError('Your population search has to provide a method to derive a new generation (population) set given a sorted population.')

    def __str__(self):
        return 'PopulationSearch(dim=%s, pop_size=%s)' % (self._dimensions, self._population_size)


class EvolutionarySearch(PopulationSearch):
    def __init__(self, *args, selection_p=0.1, mutation_p=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self._selection_p = selection_p
        self._mutation_p = mutation_p

    def mutation(self, member):
        member += np.random.normal(0, 1, self.num_dimensions)
        return member

    def select_removal(self, sorted_population):
        selection_k = int(np.ceil(self._selection_p*len(sorted_population)))
        return sorted_population[:selection_k]

    def derive_selected_population(self, sorted_population, evaluated_population):
        # Derive population after selection / kill
        kill_members = self.select_removal(sorted_population)
        return [mem for mem in sorted_population if mem not in kill_members], {mem: evaluated_population[mem] for mem in evaluated_population if mem not in kill_members}

    def derive_mutated_population(self, sorted_population):
        # Perform mutation on part of the remaining members
        mutation_k = int(np.ceil(self._mutation_p*len(sorted_population)))
        mutate_members = np.random.choice(sorted_population, mutation_k, replace=False)
        population = [mem for mem in sorted_population if mem not in mutate_members]
        population += [self._encode_member(self.mutation(self._decode_member(mem))) for mem in mutate_members]
        return population

    def derive_population(self, sorted_population, evaluated_population):
        # 1st: kill sel_p% of the population and derive a new population
        sorted_population, evaluated_population = self.derive_selected_population(sorted_population, evaluated_population)

        # 2nd: mutate mut_p% of the remaining population
        population = self.derive_mutated_population(sorted_population)

        # 3rd: generate new members
        new_members_k = self.population_size - len(population)
        population += [self._encode_member(self.sample_random()) for _ in range(new_members_k)]

        return set(population)

    def __str__(self):
        return 'EvolutionarySearch(dim=%s, pop_size=%s, selection_p=%s, mutation_p=%s' % (self._dimensions, self._population_size, self._selection_p, self._mutation_p)


class RandomEvolutionarySearch(EvolutionarySearch, RandomUniformSearch):
    pass


class CMAESSearch(EvolutionarySearch):
    # TODO in progress
    def __init__(self, *args, mu_important: int = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._mu_important = int(self.population_size/2) if mu_important is None else int(mu_important)

    def sample_random(self):
        return np.random.multivariate_normal(self._mean, self._covariance)

    def start(self, objective: Objective):
        super().start(objective)
        self._mean = np.mean([self._lower, self._upper], axis=0)
        self._covariance = np.eye(self.num_dimensions)

    def derive_population(self, sorted_population, evaluated_population):
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
