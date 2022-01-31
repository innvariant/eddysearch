import itertools

import numpy as np

from eddysearch.search.population import PopulationSearch


class GeneticSearch(PopulationSearch):
    # TODO: adapt to new interface with PopulationSearch
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._num_select_and_crossover = np.ceil(self.population_size / 10)

    def encode(self, x):
        """
        Given a g-dimensional input x (the alleles) returns its genetic representation.
        The single parameters (alleles) could be restricted to certain ranges.
        The dimension g is the number of alleles.
        The genetic representation must be hashable.

        :param x:
        :return: The gene code object on which crossover and mutation can be applied.
        """
        raise NotImplementedError()

    def decode(self, gene):
        """
        Transforms the genetic representation into a d-dimensional parameter which can be used to perform a phenotypical mapping.
        The g-dimensional output contains the alleles of the gene.
        The dimension g is the number of alleles.

        :param gene:
        :return:
        """
        raise NotImplementedError()

    def phenotypical_mapping(self, gene):
        """
        Maps a genetic representation (mostly non-linearly) to a d-dimensional input vector for the objective f: R^d -> R^1

        :param gene:
        :return:
        """
        raise NotImplementedError()

    @property
    def available_operations(self) -> set:
        raise NotImplementedError()

    def crossover(self, gene1, gene2):
        raise NotImplementedError()

    def mutate(self, gene):
        raise NotImplementedError()

    def sample_gene(self):
        raise NotImplementedError()

    def _eval_gene(self, gene):
        if self._objective is None:
            raise ValueError('Objective is none. Have you forgot to call start()?')

        if gene in self._evaluated_genes:
            return self._evaluated_genes[gene]

        phenotype = self.phenotypical_mapping(gene)
        eval = self._objective(phenotype)
        self._evaluated_genes[gene] = eval
        return eval

    def has_finished(self) -> bool:
        return self._current_generation > self._num_generations

    def start(self, *args, **kwargs):
        super().start(*args, **kwargs)
        self._current_generation = 0
        self._population = {self.sample_gene() for _ in range(self._population_size)}
        self._evaluated_genes = {}

    def step(self):
        if self._objective is None:
            raise ValueError("Objective is none. Have you forgot to call start()?")

        self._current_generation += 1
        population_eval = {member: self._eval_gene(member) for member in self._population}

        # Sorted population by evaluation. Largest/worst member first
        sorted_population = sorted(population_eval, key=lambda x: population_eval[x], reverse=True)

        # Remove largest k**2 members
        k = int(self._num_select_and_crossover)
        for idx in range(min(k ** 2, len(sorted_population))):
            self._population.remove(sorted_population[idx])

        # Crossover smallest members k yielding new k**2 member
        for p1, p2 in itertools.combinations(sorted_population[-k:], 2):
            cross_gene = self.crossover(p1, p2)
            self._population.add(cross_gene)

        while len(self._population) < self._population_size:
            operation = np.random.choice(["random", "mutate"])
            if operation == "random":
                new_random_member = self.sample_gene()
                if new_random_member is None:
                    raise ValueError("sample_gene() returned None but should return a randomly sampled gene for the population")
                self._population.add(self.sample_gene())
            else:
                member = np.random.choice(list(self._population))
                new_member = self.mutate(member)
                if new_member is None:
                    raise ValueError("mutate(member) returned None but should return a randomly sampled gene for the population")
                self._population.add(new_member)

    def end(self):
        pass

    def __str__(self):
        return 'GeneticSearch(dim=%s, pop_size=%s, num_gens=%s)' % (self._dimensions, self._population_size, self._num_generations)


class GeneticGridSearch(GeneticSearch):
    def __init__(self, binary_space=5, **kwargs):
        super().__init__(**kwargs)

        self._binary_space = binary_space
        self._mutation_max = np.ceil((2 ** (2 * binary_space)) / 10)

        # We encode as many alleles as we have dimensions
        self._num_alleles = self.num_dimensions
        self._allele_codes = np.array([2**(i * binary_space) - 1 - (2**((i - 1) * binary_space) - 1) for i in range(1, self._num_alleles + 1)])

    def _restrict(self, allele):
        return min(max(0, allele), 2**self._binary_space - 1)

    def encode(self, x):
        assert len(x) == self._num_alleles
        return sum(self._restrict(alelle) * (2**(pos * self._binary_space)) for pos, alelle in enumerate(x))

    def decode(self, gene):
        return np.array([(gene & code) >> (pos * self._binary_space) for pos, code in enumerate(self._allele_codes)])

    def phenotypical_mapping(self, gene):
        range_for_each_dimension = np.abs(np.subtract(self._lower, self._upper))
        bit_cover_area = 2**self._binary_space
        discrete_steps_per_dimension = range_for_each_dimension / bit_cover_area
        alleles = self.decode(gene)  # Each allele represents one covered block for the corresponding dimension
        lower_area = self._lower + discrete_steps_per_dimension * alleles
        upper_area = lower_area + discrete_steps_per_dimension
        return np.random.uniform(lower_area, upper_area)

    @property
    def available_operations(self) -> set:
        pass

    def _shifted_interval(self, v1: float, v2: float) -> (float, float):
        dist_v = abs(v1 - v2)
        min_v = min(v1, v2)
        max_v = max(v1, v2)
        if v1 > v2:
            min_v += dist_v
        else:
            max_v -= dist_v
        return min_v, max_v + 1

    def crossover(self, gene1, gene2):
        # Assume that f(x1) < f(x2), otherwise swap
        if self._eval_gene(gene1) > self._eval_gene(gene2):
            tmp = gene1
            gene1 = gene2
            gene2 = tmp

        alleles1 = self.decode(gene1)
        alleles2 = self.decode(gene2)

        return self.encode(np.array([np.random.randint(*self._shifted_interval(x1, x2)) for x1, x2 in zip(alleles1, alleles2)]))

    def mutate(self, gene):
        gene_alleles = self.decode(gene)
        gene_alleles += np.random.randint([-self._mutation_max] * len(gene_alleles), [self._mutation_max] * len(gene_alleles))
        return self.encode(gene_alleles)

    def sample_random(self):
        return self.decode(self.sample_gene())

    def sample_gene(self):
        return np.random.randint(0, 2 ** (2 * self._binary_space))

    def __str__(self):
        return 'GeneticGridSearch(dim=%s, pop_size=%s, num_gens=%s, bin_space=%s)' % (self._dimensions, self._population_size, self._num_generations, self._binary_space)


class GeneticRingSearch(GeneticSearch):
    def __init__(self, min_radius=0.1, max_radius=3.0, mutation_max_pos=0.5, mutation_max_radius=0.5, **kwargs):
        super().__init__(**kwargs)
        self._min_radius = min_radius
        self._max_radius = max_radius
        self._mutation_max_pos = mutation_max_pos
        self._mutation_max_radius = mutation_max_radius

    def has_finished(self) -> bool:
        return self._current_generation > self._num_generations

    def sample_gene(self):
        return self._encode_member(self.sample_random())

    def sample_random(self):
        # Sample between lower and upper for each dimension and one additional radius value between _min_radius and _max_radius
        return np.random.uniform(
            np.append(self._lower, self._min_radius),
            np.append(self._upper, self._max_radius)
        )

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

        # Restrict the sampled point on the hypersphere with boundaries defined for each dimension
        # as the search might not exceed those ranges
        return np.minimum(np.maximum(points[0], self._lower), self._upper)

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
        mid_point = (center_point1 - center_point2) / 2
        mid_radius = np.array([(radius1 - radius2) / 2])
        return np.concatenate([mid_point, mid_radius]).tostring()

    def _get_gene_eval(self, gene):
        if self.objective is None:
            raise ValueError('Objective is none. Have you forgot to call start()?')

        if gene in self._evaluated_genes:
            return self._evaluated_genes[gene]

        phenotype = self._phenotypical_mapping(gene)
        eval = self.objective(phenotype)
        self._evaluated_genes[gene] = eval
        return eval

    def step(self):
        if self.objective is None:
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
            operation = np.random.choice(["random", "mutate"])
            if operation == "random":
                self._population.add(self._encode_member(self.sample_random()))
            else:
                chosen_member_idx = np.random.randint(0, len(self._population))
                chosen_member = list(self._population)[chosen_member_idx]

                mutation = np.random.uniform(
                    [-self._mutation_max_pos] * self._dimensions + [-self._mutation_max_radius],
                    [self._mutation_max_pos] * self._dimensions + [self._mutation_max_radius]
                )
                new_member = (np.fromstring(chosen_member) + mutation).tostring()
                self._population.add(new_member)

    def end(self):
        pass

    def __str__(self):
        return 'GeneticRingSearch(dim=%s, pop_size=%s, num_gens=%s, min_radius=%s, max_radius=%s)' % (self._dimensions, self._population_size, self._num_generations, self._min_radius, self._max_radius)
