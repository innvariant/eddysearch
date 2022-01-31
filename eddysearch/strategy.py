import numpy as np

from eddysearch.objective import Objective


class SearchRunner(object):
    def __init__(
        self,
        objective: Objective,
        strategy,
        soft_evaluation_limit=1000,
        hard_evaluation_limit=1100,
        soft_bounded_search=True,
        bounded_search=False,
    ):
        self._objective = objective
        self._strategy = strategy
        self._soft_evaluation_limit = soft_evaluation_limit
        self._hard_evaluation_limit = hard_evaluation_limit
        self._soft_bounded_search = True if soft_bounded_search else False
        self._bounded_search = True if bounded_search else False

    def run(self):
        self._num_evaluations = 0
        self._minimum_eval = np.inf
        self._minimum_arg = None
        self._search_paths = (
            {}
        )  # one can walk multiple search paths, indexed by keys. default is 0
        self._current_search_group = {}

        def eval_objective(x):
            if self._num_evaluations >= self._hard_evaluation_limit:
                raise StopIteration()

            # For bounded searches, check if the vector is within the allowed search bounds
            if self._soft_bounded_search or self._bounded_search:
                if not np.all(x >= self._objective.search_bounds[:, 0]) or not np.all(
                    x <= self._objective.search_bounds[:, 1]
                ):
                    if self._bounded_search:
                        raise ValueError(
                            "Trying to evaluate x=%s which is out of bounds %s"
                            % (x, self._objective.search_bounds)
                        )
                    else:
                        import warnings

                        warnings.warn(
                            "Evaluating x=%s which is out of bounds %s"
                            % (x, self._objective.search_bounds)
                        )

            group = self._strategy.current_group  # hashable group
            if group not in self._current_search_group:
                self._current_search_group[group] = []
            self._current_search_group[group].append(x)

            self._num_evaluations += 1
            eval = self._objective(x)
            if eval < self._minimum_eval:
                self._minimum_eval = eval
                self._minimum_arg = x
            return eval

        self._strategy.start(eval_objective)

        while (
            not self._strategy.has_finished()
            and self._num_evaluations < self._soft_evaluation_limit
        ):
            self._current_search_group = (
                {}
            )  # In each iteration we collect new sets of (d,)-points within various groups

            self._strategy.step()

            for group_key in self._current_search_group:
                if group_key not in self._search_paths:
                    self._search_paths[group_key] = []
                self._search_paths[group_key].append(
                    np.array(self._current_search_group[group_key])
                )

        self._search_path = [
            self._search_paths[group_key] for group_key in self._search_paths
        ]

        self._strategy.end()


class SearchStrategy(object):
    def __init__(self, dimensions: int, lower: np.ndarray, upper: np.ndarray):
        self._dimensions = dimensions
        assert len(lower.shape) == 1
        assert lower.shape[0] == dimensions
        assert len(upper.shape) == 1
        assert upper.shape[0] == dimensions

        self._lower = lower
        self._upper = upper
        self._objective = None
        self._current_group = 0

    def sample_random(self):
        raise NotImplementedError(
            "If your methods contains a random-sampling method, provide it here (only used for some algorithms)."
        )

    @property
    def num_dimensions(self) -> int:
        return self._dimensions

    @property
    def objective(self) -> Objective:
        return self._objective

    @property
    def current_group(self):
        return self._current_group

    @current_group.setter
    def current_group(self, key):
        assert key.__hash__
        self._current_group = key

    def has_finished(self) -> bool:
        raise NotImplementedError()

    def start(self, objective: Objective):
        self._objective = objective

    def step(self):
        raise NotImplementedError()

    def end(self):
        raise NotImplementedError()

    def __str__(self):
        raise NotImplementedError(
            "You should give your strategy a (possibly parameterized) name."
        )
