import numpy as np

from eddysearch.objective import Objective
from eddysearch.search.randomsearch import RandomUniformSearch
from eddysearch.strategy import SearchStrategy


class PopulationSearch(SearchStrategy):
    def __init__(
        self, *args, population_size: int = 10, num_generations: int = 10, **kwargs
    ):
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
            self._evaluated_members[member] = self.objective(
                self._decode_member(member)
            )
        return self._evaluated_members[member]

    def start(self, objective: Objective):
        super().start(objective)
        self._current_generation = 0
        self._evaluated_members = {}
        self._population = {
            self._encode_member(self.sample_random())
            for _ in range(self._population_size)
        }

    def step(self):
        self._current_generation += 1

        population_eval = {
            mem: self._cached_evaluation(mem) for mem in self._population
        }

        # Sorted population by evaluation. Largest/worst member first
        sorted_population = sorted(
            population_eval, key=lambda x: population_eval[x], reverse=True
        )

        self._population = self.derive_population(sorted_population, population_eval)
        assert (
            self._population is not None and len(self._population) > 0
        ), "derive_population() returned None or empty population"

    def derive_population(self, sorted_population, evaluated_population):
        raise NotImplementedError(
            "Your population search has to provide a method to derive a new generation (population) set given a sorted population."
        )

    def __str__(self):
        return "PopulationSearch(dim=%s, pop_size=%s)" % (
            self._dimensions,
            self._population_size,
        )


class EvolutionarySearch(PopulationSearch):
    def __init__(self, *args, selection_p=0.1, mutation_p=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self._selection_p = selection_p
        self._mutation_p = mutation_p

    def mutation(self, member):
        member += np.random.normal(0, 1, self.num_dimensions)
        return member

    def select_removal(self, sorted_population):
        selection_k = int(np.ceil(self._selection_p * len(sorted_population)))
        return sorted_population[:selection_k]

    def derive_selected_population(self, sorted_population, evaluated_population):
        # Derive population after selection / kill
        kill_members = self.select_removal(sorted_population)
        return [mem for mem in sorted_population if mem not in kill_members], {
            mem: evaluated_population[mem]
            for mem in evaluated_population
            if mem not in kill_members
        }

    def derive_mutated_population(self, sorted_population):
        # Perform mutation on part of the remaining members
        mutation_k = int(np.ceil(self._mutation_p * len(sorted_population)))
        mutate_members = np.random.choice(sorted_population, mutation_k, replace=False)
        population = [mem for mem in sorted_population if mem not in mutate_members]
        population += [
            self._encode_member(self.mutation(self._decode_member(mem)))
            for mem in mutate_members
        ]
        return population

    def derive_population(self, sorted_population, evaluated_population):
        # 1st: kill sel_p% of the population and derive a new population
        sorted_population, evaluated_population = self.derive_selected_population(
            sorted_population, evaluated_population
        )

        # 2nd: mutate mut_p% of the remaining population
        population = self.derive_mutated_population(sorted_population)

        # 3rd: generate new members
        new_members_k = self.population_size - len(population)
        population += [
            self._encode_member(self.sample_random()) for _ in range(new_members_k)
        ]

        return set(population)

    def __str__(self):
        return (
            "EvolutionarySearch(dim=%s, pop_size=%s, selection_p=%s, mutation_p=%s"
            % (
                self._dimensions,
                self._population_size,
                self._selection_p,
                self._mutation_p,
            )
        )


class RandomEvolutionarySearch(EvolutionarySearch, RandomUniformSearch):
    pass


class CMAESSearch(EvolutionarySearch):
    # TODO in progress
    def __init__(
        self, *args, learning_rate: float = 1, mu_important: int = None, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self._learning_rate = learning_rate  # corresponds to c_m in CMA-ES
        self._mu_important = (
            int(self.population_size / 2) if mu_important is None else int(mu_important)
        )

    def sample_random(self):
        return np.random.multivariate_normal(self._mean, self._covariance)

    def start(self, objective: Objective):
        self._mean = np.random.uniform(self._lower, self._upper)
        self._covariance = np.eye(self.num_dimensions) * np.mean(
            np.abs(np.subtract(self._lower, self._upper))
        )
        super().start(objective)

    def derive_population(self, sorted_population, evaluated_population):
        # Discard #mu_important members (first x members by sorted population)
        # print('---'*5)
        # print('Mean of population: ', np.mean([np.fromstring(mem) for mem in sorted_population], axis=0))
        member_points = np.array(
            [
                self._decode_member(mem)
                for mem in sorted_population[self._mu_important :]
            ]
        )
        # print(len(member_points))
        # print('Mean after selection: ', np.mean(member_points, axis=0))

        # Move covariance to better solution
        # Use the mean of previous generation
        covariance = np.copy(self._covariance)
        num_members = len(member_points)
        for i in range(member_points.shape[1]):
            # E_i = np.mean(member_points[:,i])  # estimated expected value for dimension i
            E_i = self._mean[i]
            for j in range(i, member_points.shape[1]):
                covariance[i, j] = np.sum((member_points[:, i] - E_i) ** 2) / (
                    num_members - 1
                )
                covariance[j, i] = covariance[i, j]
        self._covariance = np.cov(member_points.T)

        # Move mean to better solutions
        # print('Stored distribution mean: ', self._mean)
        self._mean = self._mean + self._learning_rate * (
            np.mean(member_points, axis=0) - self._mean
        )

        new_members = np.array(
            [self.sample_random() for _ in range(self._mu_important)]
        )
        member_points = np.concatenate([member_points, new_members])

        return {self._encode_member(mem) for mem in member_points}

    def end(self):
        pass

    def has_finished(self):
        return False  # never stop

    def __str__(self):
        return "CMAESSearch(dim=%s, pop_size=%s)" % (
            self._dimensions,
            self._population_size,
        )


class SpeciesCMAESSearch(PopulationSearch):
    def __init__(
        self, *args, learning_rate: float = 1, mu_important: int = None, **kwargs
    ):
        super().__init__(*args, **kwargs)

        self._learning_rate = learning_rate  # corresponds to c_m in CMA-ES
        self._mu_important = (
            int(self.population_size / 2) if mu_important is None else int(mu_important)
        )

    def sample_random(self, species=0):
        return np.random.multivariate_normal(
            self._centroids[species], self._covariances[species]
        )

    def start(self, objective: Objective):
        self._objective = objective
        self._current_generation = 0
        self._evaluated_members = {}
        self._centroids = [np.random.uniform(self._lower, self._upper)]
        self._covariances = [
            np.eye(self.num_dimensions)
            * np.mean(np.abs(np.subtract(self._lower, self._upper)))
        ]
        self._centroid_histories = [[np.copy(self._centroids[0])]]
        self._covariance_histories = [[np.copy(self._covariances[0])]]
        self._populations = [
            {
                self._encode_member(self.sample_random())
                for _ in range(self._population_size)
            }
        ]
        # force_start = [np.array([ 0.767164  , -0.76002355]), np.array([0.3757698 , 1.15696329]), np.array([ 0.53418772, -0.69829167]), np.array([-0.90615645,  0.35733911]), np.array([-0.95456183, -0.25032612]), np.array([-1.86690963, -0.26622719]), np.array([1.59126521, 1.24442322]), np.array([1.47703175, 0.51871846]), np.array([-0.87818858, -0.79254416]), np.array([-2.00827781, -1.84839004]), np.array([ 1.88570074, -0.80706232]), np.array([0.10968603, 0.416345  ]), np.array([-0.15412068, -0.3120081 ]), np.array([-0.18308967, -1.61745235]), np.array([-0.52871902,  1.0097951 ]), np.array([-0.85989092, -0.82414516]), np.array([-0.26492876, -0.11489943]), np.array([-1.041764  , -0.58269327]), np.array([-1.60976258, -0.62300809]), np.array([-1.01895488,  0.17777552])]
        # self._populations = [{self._encode_member(mem) for mem in force_start}]
        # print('Start')
        # print([self._decode_member(mem) for mem in self._populations[0]])

    def step(self):
        self._current_generation += 1

        print("-" * 10)
        print("Generation", self._current_generation)

        for species, population in enumerate(self._populations):
            print("-- Species", species)
            self.current_group = species
            population_eval = {mem: self._cached_evaluation(mem) for mem in population}

            # Sorted population by evaluation. Largest/worst member first
            sorted_population = sorted(
                population_eval, key=lambda x: population_eval[x], reverse=True
            )

            member_points = np.array(
                [
                    self._decode_member(mem)
                    for mem in sorted_population[self._mu_important :]
                ]
            )

            covariance = np.copy(self._covariances[species])
            num_members = len(member_points)
            for i in range(member_points.shape[1]):
                # E_i = np.mean(member_points[:,i])  # estimated expected value for dimension i
                E_i = self._centroids[species][i]
                for j in range(i, member_points.shape[1]):
                    covariance[i, j] = np.sum((member_points[:, i] - E_i) ** 2) / (
                        num_members - 1
                    )
                    covariance[j, i] = covariance[i, j]
            self._covariances[species] = np.cov(member_points.T)
            self._covariance_histories[species].append(
                np.copy(self._covariances[species])
            )

            self._centroids[species] = self._centroids[
                species
            ] + self._learning_rate * (
                np.mean(member_points, axis=0) - self._centroids[species]
            )
            self._centroid_histories[species].append(np.copy(self._centroids[species]))

            print("Covariance & mean:")
            print(self._covariances[species])
            print("\t\t\t\t", self._centroids[species])
            print("Expansion?")
            # for d in range(self.num_dimensions):
            avg_covariance = np.mean(self._covariance_histories[species][-5:], axis=0)
            cur_dimension_variance = np.mean(np.diag(self._covariances[species]))
            avg_dimension_variance = np.mean(np.diag(avg_covariance))
            print("Cur variance over dimensions:", cur_dimension_variance)
            print("Avg variance over dimensions:", avg_dimension_variance)
            print("Averaged historical covariance:")
            print(np.mean(self._covariance_histories[species][-5:], axis=0))
            split_dimensions = []
            for d in range(self.num_dimensions):
                if len(self._centroid_histories[species]) > 2:
                    print(
                        "Centroid path for dimension %s is %.4f -> %.4f -> %.4f"
                        % (
                            d,
                            self._centroid_histories[species][-3][d],
                            self._centroid_histories[species][-2][d],
                            self._centroid_histories[species][-1][d],
                        )
                    )
                    d0 = (
                        self._centroid_histories[species][-3][d]
                        - self._centroid_histories[species][-2][d]
                    )
                    d1 = (
                        self._centroid_histories[species][-2][d]
                        - self._centroid_histories[species][-1][d]
                    )

                    if np.sign(d0) != np.sign(d1):
                        print("Centroid path for dim %s changed direction" % d)
                        print(
                            "abs(c[-2]-c[-1]) = %.4f -> %.2f of variance V_d[t-1]=%.4f"
                            % (
                                abs(d0),
                                abs(d0) / self._covariance_histories[species][-2][d, d],
                                self._covariance_histories[species][-2][d, d],
                            )
                        )
                        print(
                            "abs(c[-1]-c[0]) = %.4f -> %.2f of variance V_d[t]=%.4f"
                            % (
                                abs(d1),
                                abs(d1) / self._covariances[species][d, d],
                                self._covariance_histories[species][-1][d, d],
                            )
                        )

                        if self._covariances[species][d, d] > np.maximum(
                            1, self._covariance_histories[species][-2][d, d]
                        ):
                            split_dimensions.append(d)

                if (
                    self._covariances[species][d, d]
                    > self._covariance_histories[species][-2][d, d]
                ):
                    print("Dimension %s variance is greater than its last variance" % d)
                if self._covariances[species][d, d] / np.mean(avg_covariance[d, d]) > 1:
                    print(
                        "Dimension %s is greater than its historical average variance"
                        % d
                    )
                if self._covariances[species][d, d] / avg_dimension_variance > 1:
                    print(
                        "Dimension %s is greater than total historical average variance"
                        % d
                    )
                if self._covariances[species][d, d] / cur_dimension_variance > 1:
                    print("Dimension %s is greater than current average variance" % d)

            new_members = np.array(
                [self.sample_random(species) for _ in range(self._mu_important)]
            )
            member_points = np.concatenate([member_points, new_members])

            self._populations[species] = {
                self._encode_member(mem) for mem in member_points
            }

            # Perform possible species merges/deaths
            # Deaths: species is consistently worse than others
            # Merges: species close to each other

            # Perform possible species splits
            if len(split_dimensions) > 0:
                print(
                    "Performing population split along dimensions %s" % split_dimensions
                )
                # centroid_0 = self._centroids[species]
                centroid_1 = np.copy(self._centroid_histories[species][-2])
                centroid_1[split_dimensions] = self._centroids[species][
                    split_dimensions
                ]
                covariance = np.copy(self._covariances[species])
                new_species = len(self._populations)
                self._centroids.append(centroid_1)
                self._covariances.append(covariance)
                self._centroid_histories.append([centroid_1])
                self._covariance_histories.append([covariance])
                self._populations.append(
                    {
                        self._encode_member(self.sample_random(new_species))
                        for _ in range(self._population_size)
                    }
                )

                # delete historical record
                self._centroid_histories[species] = self._centroid_histories[species][
                    -2:
                ]
                self._covariance_histories[species] = self._covariance_histories[
                    species
                ][-2:]

    def end(self):
        pass

    def has_finished(self):
        return False  # never stop

    def __str__(self):
        return "SpeciesCMAESSearch(dim=%s, pop_size=%s)" % (
            self._dimensions,
            self._population_size,
        )
