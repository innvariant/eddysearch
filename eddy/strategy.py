import numpy as np


class Search(object):
    def __init__(self, objective, strategy, soft_evaluation_limit=1000, hard_evaluation_limit=1100):
        self._objective = objective
        self._strategy = strategy
        self._soft_evaluation_limit = soft_evaluation_limit
        self._hard_evaluation_limit = hard_evaluation_limit

    def run(self):
        self._num_evaluations = 0
        self._minimum_eval = np.inf
        self._minimum_arg = None
        self._search_path = []

        def eval_objective(x):
            if self._num_evaluations >= self._hard_evaluation_limit:
                raise StopIteration()

            self._search_path.append(x)
            self._num_evaluations += 1
            eval = self._objective(x)
            if eval < self._minimum_eval:
                self._minimum_eval = eval
                self._minimum_arg = x
            return eval

        self._strategy.start(eval_objective)

        while not self._strategy.has_finished() and self._num_evaluations < self._soft_evaluation_limit:
            self._strategy.step()

        self._strategy.end()


class Strategy(object):
    def has_finished(self) -> bool:
        raise NotImplementedError()

    def start(self, objective):
        raise NotImplementedError()

    def step(self):
        raise NotImplementedError()

    def end(self):
        raise NotImplementedError()
