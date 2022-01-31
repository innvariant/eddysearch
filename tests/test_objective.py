import numpy as np

from eddysearch.objective import CrossInTrayObjective
from eddysearch.objective import EggholderObjective
from eddysearch.objective import HimmelblauObjective
from eddysearch.objective import LeviN13Objective
from eddysearch.objective import RastriginObjective
from eddysearch.objective import RosenbrockObjective
from eddysearch.objective import Stier2020A1Objective
from eddysearch.objective import Stier2020A2Objective
from eddysearch.objective import Stier2020BObjective


objectives = [
    RastriginObjective,
    RosenbrockObjective,
    LeviN13Objective,
    HimmelblauObjective,
    CrossInTrayObjective,
    EggholderObjective,
    Stier2020A1Objective,
    Stier2020A2Objective,
    Stier2020BObjective,
]


def test_construct_default_objectives():
    for clazz in objectives:
        obj = clazz()
        p = np.random.uniform(
            obj.search_bounds[:, 0] * obj.dims, obj.search_bounds[:, 1] * obj.dims
        )

        obj(p)


def test_objective_counts_evaluations():
    num_evals = np.random.randint(1, 10)

    for clazz in objectives:
        obj = clazz()

        for eval_step in range(num_evals):
            p = np.random.uniform(
                obj.search_bounds[:, 0] * obj.dims, obj.search_bounds[:, 1] * obj.dims
            )
            obj(p)

        assert obj.num_evaluations == num_evals


def test_objective_counts_evaluations_but_ignores_vis_evals():
    num_evals = np.random.randint(1, 10)

    for clazz in objectives:
        obj = clazz()

        for eval_step in range(num_evals):
            p = np.random.uniform(
                obj.search_bounds[:, 0] * obj.dims, obj.search_bounds[:, 1] * obj.dims
            )
            obj(p)

        for noisy_evaluations in range(np.random.randint(1, 10)):
            p = np.random.uniform(
                obj.search_bounds[:, 0] * obj.dims, obj.search_bounds[:, 1] * obj.dims
            )
            obj.evaluate_visual(p)
            obj.evaluate_raw(p)

        assert obj.num_evaluations == num_evals
