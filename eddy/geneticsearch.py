import numpy as np
import itertools
from .strategy import Strategy


def encode(p1, p2, pspace):
    p1 = min(max(0, p1), 2**pspace-1)
    p2 = min(max(0, p2), 2**pspace-1)
    return (p1 << pspace) + p2


def decode(x, pspace):
    return x >> pspace, x - ((x >> pspace) << pspace)


def get_shifted_interval(v1, v2):
    dist_v = abs(v1 - v2)
    min_v = min(v1, v2)
    max_v = max(v1, v2)
    if v1 > v2:
        min_v += dist_v
    else:
        max_v -= dist_v
    return min_v, max_v


def crossover(x1, y1, x2, y2, pspace):
    # Assume that f(x1) < f(x2), otherwise swap
    if y1 > y2:
        tmp = x1
        x1 = x2
        x2 = tmp

    a1, b1 = decode(x1, pspace)
    a2, b2 = decode(x2, pspace)

    a_low, a_high = get_shifted_interval(a1, a2)
    b_low, b_high = get_shifted_interval(a1, a2)

    return encode(np.random.randint(a_low, a_high+1), np.random.randint(b_low, b_high+1), pspace=pspace)


def phenotypical_mapping(gene_bits, pspace, map_lower=-5.0, map_upper=5.0):
    map_range = abs(map_lower-map_upper)
    bit_cover_area = 2**pspace
    discrete_steps = map_range/bit_cover_area
    a, b = decode(gene_bits, pspace=pspace)
    x0_lower = map_lower + discrete_steps * a
    x1_lower = map_lower + discrete_steps * b
    return np.array([np.random.uniform(x0_lower, x0_lower+discrete_steps), np.random.uniform(x1_lower, x1_lower+discrete_steps)])


class GeneticGridSearch(Strategy):
    def __init__(self, dimensions : int, lower : float, upper : float, population_size=10, num_generations=10, binary_space=5):
        self._dimensions = dimensions
        self._lower = lower
        self._upper = upper
        self._population_size = population_size
        self._num_generations = num_generations
        self._binary_space = binary_space
        self._num_select_and_crossover = np.ceil(population_size/10)
        self._mutation_max = np.ceil((2 ** (2 * binary_space)) / 10)

        self._objective = None

    def has_finished(self) -> bool:
        return self._current_generation > self._num_generations

    def start(self, objective):
        self._objective = objective
        self._current_generation = 0
        self._population = {np.random.randint(0, 2 ** (2 * self._binary_space)) for _ in range(self._population_size)}
        self._evaluated_genes = {}

    def _get_gene_eval(self, gene):
        if self._objective is None:
            raise ValueError('Objective is none. Have you forgot to call start()?')

        if gene in self._evaluated_genes:
            return self._evaluated_genes[gene]

        phenotype = phenotypical_mapping(gene, self._binary_space, self._lower, self._upper)
        eval = self._objective(phenotype)
        self._evaluated_genes[gene] = eval
        return eval

    def step(self):
        if self._objective is None:
            raise ValueError('Objective is none. Have you forgot to call start()?')

        self._current_generation += 1
        population_eval = {mem: self._get_gene_eval(mem) for mem in self._population}

        # Sorted population by evaluation. Largest/worst member first
        sorted_population = sorted(population_eval, key=lambda x: population_eval[x], reverse=True)

        # Remove largest k**2 members
        k = int(self._num_select_and_crossover)
        for idx in range(min(k ** 2, len(sorted_population))):
            self._population.remove(sorted_population[idx])

        # Crossover smallest members k yielding new k**2 member
        for p1, p2 in itertools.product(sorted_population[-k:], sorted_population[-k:]):
            cross_gene = crossover(p1, population_eval[p1], p2, population_eval[p2], pspace=self._binary_space)
            self._population.add(cross_gene)

        while len(self._population) < self._population_size:
            operation = np.random.choice(['random', 'mutate'])
            if operation is 'random':
                random_gene = np.random.randint(0, 2 ** (2 * self._binary_space))
                self._population.add(random_gene)
            else:
                member = np.random.choice(list(self._population))
                member += np.random.randint(-self._mutation_max, self._mutation_max)
                self._population.add(member)

    def end(self):
        pass


class GeneticRingSearch(Strategy):
    def __init__(self,
                 dimensions : int,
                 lower : float,
                 upper : float,
                 population_size=10,
                 num_generations=10,
                 min_radius=0.1,
                 max_radius=3.0,
                 mutation_max_pos=0.5,
                 mutation_max_radius=0.5):
        self._dimensions = dimensions
        self._lower = lower
        self._upper = upper
        self._population_size = population_size
        self._num_generations = num_generations
        self._min_radius = min_radius
        self._max_radius = max_radius
        self._mutation_max_pos = mutation_max_pos
        self._mutation_max_radius = mutation_max_radius

        self._num_select_and_crossover = np.ceil(population_size/5)

    def has_finished(self) -> bool:
        return self._current_generation > self._num_generations

    def start(self, objective):
        self._objective = objective
        self._current_generation = 0
        # Each gene is a tuple of coordinates and the radius, so it has dimensions+1 elements drawn from a uniform distribution
        self._population = {self._random_gene() for _ in range(self._population_size)}
        self._evaluated_genes = {}

    def _random_gene(self):
        return np.random.uniform(
            [self._lower] * self._dimensions + [self._min_radius],
            [self._upper] * self._dimensions + [self._max_radius]
        ).tostring()

    def _phenotypical_mapping(self, gene):
        gene_array = np.fromstring(gene)

        assert len(gene_array) == self._dimensions + 1

        center_point = np.array([gene_array[:-1]])
        radius = gene_array[-1]

        # Sample one point from the hypersphere with dimension self._dimensions
        # See http://mathworld.wolfram.com/HyperspherePointPicking.html
        normal_deviates = np.random.normal(size=(self._dimensions, 1))
        norm = np.sqrt((normal_deviates ** 2).sum(axis=0))
        points = center_point + ((radius * normal_deviates) / norm)

        return points[0]

    def _crossover(self, gene1, y1, gene2, y2):
        gene1_array = np.fromstring(gene1)
        gene2_array = np.fromstring(gene2)

        assert len(gene1_array) == self._dimensions + 1
        assert len(gene2_array) == self._dimensions + 1

        # Assume that f(gene1) < f(gene2), otherwise swap
        if y1 > y2:
            tmp = gene1_array
            gene1_array = gene2_array
            gene2_array = tmp

        # Get points from gene
        center_point1 = gene1_array[:-1]
        radius1 = gene1_array[-1]
        center_point2 = gene2_array[:-1]
        radius2 = gene2_array[-1]

        # Calculate mid points
        mid_point = (center_point1-center_point2)/2
        mid_radius = np.array([(radius1-radius2)/2])
        return np.concatenate([mid_point, mid_radius]).tostring()

    def _get_gene_eval(self, gene):
        if self._objective is None:
            raise ValueError('Objective is none. Have you forgot to call start()?')

        if gene in self._evaluated_genes:
            return self._evaluated_genes[gene]

        phenotype = self._phenotypical_mapping(gene)
        eval = self._objective(phenotype)
        self._evaluated_genes[gene] = eval
        return eval

    def step(self):
        if self._objective is None:
            raise ValueError('Objective is none. Have you forgot to call start()?')

        self._current_generation += 1
        population_eval = {mem: self._get_gene_eval(mem) for mem in self._population}

        # Sorted population by evaluation. Largest/worst member first
        sorted_population = sorted(population_eval, key=lambda x: population_eval[x], reverse=True)

        # Remove largest k**2 members
        k = int(self._num_select_and_crossover)
        for idx in range(min(k ** 2, len(sorted_population))):
            self._population.remove(sorted_population[idx])

        # Crossover smallest members k yielding new k**2 member
        for p1, p2 in itertools.product(sorted_population[-k:], sorted_population[-k:]):
            cross_gene = self._crossover(p1, population_eval[p1], p2, population_eval[p2])
            self._population.add(cross_gene)

        # Fill up population with new random members
        while len(self._population) < self._population_size:
            operation = np.random.choice(['random', 'mutate'])
            if operation is 'random':
                self._population.add(self._random_gene())
            else:
                chosen_member_idx = np.random.randint(0, len(self._population))
                chosen_member = list(self._population)[chosen_member_idx]

                mutation = np.random.uniform(
                    [-self._mutation_max_pos] * self._dimensions + [-self._mutation_max_radius],
                    [self._mutation_max_pos] * self._dimensions + [self._mutation_max_radius]
                )
                new_member = (np.fromstring(chosen_member)+mutation).tostring()
                self._population.add(new_member)

    def end(self):
        pass
