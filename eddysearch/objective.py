import matplotlib.colors
import numpy as np


class Objective(object):
    """
    An interface for n-dimensional objective functions.
    It tracks the number of evaluations of the objective and -- if desired -- the argument history.
    Parameterized objectives should be freshly instantiated and parameterized in their initialization (constructor) method.
    For analytical functions with known minima it should provide those minima.
    The objective should contain recommendations for cube bounds for plotting.
    """

    _found_minimum = np.inf
    _found_args = None
    _num_evaluations = 0
    _track_history = False
    _history_args = []
    _visualize_normalizer = matplotlib.colors.NoNorm()
    _visualize_z_shift = 0

    @property
    def dims(self) -> int:
        """
        :return: Number of input dimensions for the objective function.
        """
        raise NotImplementedError()

    @property
    def found_minimum(self):
        return self._found_minimum

    @property
    def found_args(self):
        return self._found_args

    @property
    def num_evaluations(self):
        return self._num_evaluations

    @property
    def history(self):
        return self._track_history

    @property
    def track_history(self):
        return self._track_history

    @track_history.setter
    def track_history(self, flag):
        self._track_history = True if flag else False

    @property
    def search_bounds(self):
        """
        Returns the search boundaries for the concrete objective. It must return an array with shape (d, 2).
        E.g. if the objective is in 2d space, it returns an array with shape (2, 2) such as [[-5, 5], [-3.5, 6]].
        This specifies the first dimension (often called `x`) to be searched between -5 and 5 and the second dimension
        to be searched between -3.5 and 6.

        :return: np.ndarray
        """
        raise NotImplementedError()

    @property
    def color_normalizer(self):
        return self._visualize_normalizer

    @property
    def visualization_z_shift(self):
        return self._visualize_z_shift

    @visualization_z_shift.setter
    def visualization_z_shift(self, val):
        self._visualize_z_shift = val

    @property
    def visualization_bounds(self):
        """
        Returns the recommended lower and upper bound for each dimension, thus an array shaped (d, 2).
        If the objective is only available in a 2d space it returns an array shaped (2, 2).

        :rtype: np.ndarray
        :return:
        """
        raise NotImplementedError()

    @property
    def minima(self):
        """
        Returns an array shaped (b, d+1) of d-dimensional points which are all arguments for minimums of the objective.
        The last entry (d+1) is the value of the minimum.

        :rtype: np.ndarray
        :return:
        """
        raise NotImplementedError()

    def _call(self, x: np.ndarray, *args, **kwargs):
        """
        This should be the main implementation of the objective function.
        It should not be called directly.
        Single objective evaluation for a x with shape (d,). Usually d is two.

        :param x:
        :return:
        """
        raise NotImplementedError()

    def _tracked_evaluation(self, x):
        value = self._call(x)

        self._num_evaluations += 1
        if value < self._found_minimum:
            self._found_minimum = value
            self._found_args = x

        if self._track_history:
            self._history_args.append(x)

        return value

    def _checked_call(self, x, callback):
        if not isinstance(x, np.ndarray):
            x = np.array(x)

        if x.ndim == 0:
            x = np.array([x])

        if x.ndim == 1:
            return callback(x)
        elif x.ndim == 2:
            return np.array([callback(x1d) for x1d in x])
        else:
            raise ValueError(
                'Too many number of dimensions given to objective function "%s"' % self
            )

    def evaluate_visual(self, x, z_shift=None):
        """
        Evaluation function without tracking which can be used for a common call for visualization purposes in which a
        z-shift (output value) is needed to visualize it properly.

        :param x: The usual parameter; same as in __call__(self, x)
        :param z_shift: Optional value to perform a local z-shift instead of one which is defined on the whole objective.
        :return:
        """
        z_correction = self.visualization_z_shift if z_shift is None else z_shift
        return self._checked_call(x, self._call) + z_correction

    def evaluate_raw(self, x):
        """
        Use this function for un-tracked evaluations of this particular objective. For example to generate points for
        visualizations etc. To actually evaluate the objective with tracking functionality, use objective() (the built-
        in objective call).

        :param x:
        :return:
        """
        return self._checked_call(x, self._call)

    def __call__(self, x):
        return self._checked_call(x, self._tracked_evaluation)

    def __str__(self):
        raise NotImplementedError(
            "You should give your objective function a (possibly parameterized) name."
        )


class RastriginObjective(Objective):
    def __init__(self, n_dim=2, A=10):
        self._A = A
        self._n_dim = n_dim

        self._search_bounds = np.array([[-5, 5]] * self._n_dim)
        self._visualization_bounds = np.array([[-5, 5]] * self._n_dim)
        self._visualize_normalizer = matplotlib.colors.LogNorm()

        # There is exactly one global minimum at f(0, ..) = 0
        self._minima = np.array([[0] * self._n_dim + [0]])

    @property
    def dims(self) -> int:
        return self._n_dim

    @property
    def search_bounds(self):
        return self._search_bounds

    @property
    def visualization_bounds(self):
        return self._visualization_bounds

    @property
    def minima(self):
        return self._minima

    def _call(self, x: np.ndarray, *args, **kwargs):
        assert (
            x.shape[0] == self._n_dim
        ), "Rastrigin was defined with dim=%s but was called with shape %s" % (
            self._n_dim,
            x.shape,
        )
        n = self._n_dim
        return self._A * n + sum(
            x[d] ** 2 - self._A * np.cos(2 * np.pi * x[d]) for d in range(n)
        )

    def __str__(self):
        return "Rastrigin(dim=%s, A=%s)" % (self._n_dim, self._A)


class GenericObjective(Objective):
    def __init__(
        self,
        fn,
        search_bounds,
        minima,
        visualization_bounds=None,
        color_normalizer=matplotlib.colors.NoNorm(),
        n_dim=2,
    ):
        self._fn = fn
        self._n_dim = n_dim

        self._search_bounds = search_bounds
        self._visualization_bounds = (
            visualization_bounds if visualization_bounds is not None else search_bounds
        )
        self._visualize_normalizer = color_normalizer

        # Compute all minima given the minima arguments and the evaluation function
        self._minima = np.array([np.concatenate([arg, [fn(arg)]]) for arg in minima])

    @property
    def dims(self) -> int:
        return self._n_dim

    @property
    def search_bounds(self):
        return self._search_bounds

    @property
    def visualization_bounds(self):
        return self._visualization_bounds

    @property
    def minima(self):
        return self._minima

    def _call(self, x: np.ndarray, *args, **kwargs):
        assert (
            x.shape[0] is self._n_dim
        ), "Objective was defined with dim=%s but was called with shape %s" % (
            self._n_dim,
            x.shape,
        )
        return self._fn(x)

    def __str__(self):
        return "%s(dim=%s)" % (self._fn.__name__, self._n_dim)


def rastrigin1d(x, A=10):
    assert isinstance(x, (np.ndarray, np.generic))
    assert x.ndim == 1
    n = x.shape[0]
    return A * n + sum(x[d] ** 2 - A * np.cos(2 * np.pi * x[d]) for d in range(n))


def rastrigin(x, A=10):
    return _batched(rastrigin1d, x, A)


def rosenbrock1d(x, a=1, b=100):
    assert isinstance(x, (np.ndarray, np.generic))
    assert x.ndim == 1

    return (a - x[0]) ** 2 + b * (x[1] - x[0]) ** 2


def rosenbrock(x, a=1, b=100):
    return _batched(rosenbrock1d, x, a, b)


class RosenbrockObjective(GenericObjective):
    def __init__(
        self,
        search_bounds=np.array([[-2, 2], [-2, 2]]),
        color_normalizer=matplotlib.colors.LogNorm(),
    ):
        ndim = len(search_bounds)
        minima = np.array([[1] * ndim])
        super().__init__(rosenbrock1d, search_bounds, minima, ndim, color_normalizer)


def goldstein_price1d(z):
    assert isinstance(z, (np.ndarray, np.generic))
    assert z.ndim == 1

    x = z[0]
    y = z[1]

    return (
        1
        + (x + y + 1) ** 2
        * (19 - 14 * x + 3 * x ** 2 - 14 * y + 6 * x * y + 3 * y ** 2)
    ) * (
        30
        + (2 * x - 3 * y) ** 2
        * (18 - 32 * x + 12 * x ** 2 + 48 * y - 36 * x * y + 27 * y ** 2)
    )


def goldstein_price(x):
    """
    Minimum at goldstein_price(np.array([0, -1])) = 3

    :param x:
    :return:
    """
    return _batched(goldstein_price1d, x)


class GoldsteinPriceObjective(GenericObjective):
    def __init__(
        self,
        search_bounds=np.array([[-2, 2], [-2, 2]]),
        color_normalizer=matplotlib.colors.LogNorm(),
    ):
        ndim = len(search_bounds)
        minima = np.array([[0, -1]])
        super().__init__(
            goldstein_price1d, search_bounds, minima, ndim, color_normalizer
        )


def levi_n13_1d(z):
    assert isinstance(z, (np.ndarray, np.generic))
    assert z.ndim == 1

    x = z[0]
    y = z[1]

    return (
        np.sin(3 * np.pi * x) ** 2
        + (x - 1) ** 2 * (1 + np.sin(3 * np.pi * y) ** 2)
        + (y - 1) ** 2 * (1 + np.sin(2 * np.pi * y) ** 2)
    )


def levi_n13(x):
    """
    Minimum at levi_n13(np.array([1, 1])) = 0

    :param x:
    :return:
    """
    return _batched(levi_n13, x)


class LeviN13Objective(GenericObjective):
    def __init__(
        self,
        search_bounds=np.array([[-8, 8], [-8, 8]]),
        color_normalizer=matplotlib.colors.LogNorm(),
    ):
        ndim = len(search_bounds)
        minima = np.array([[1, 1]])
        super().__init__(levi_n13_1d, search_bounds, minima, ndim, color_normalizer)


def himmelblau1d(z):
    assert isinstance(z, (np.ndarray, np.generic))
    assert z.ndim == 1

    x = z[0]
    y = z[1]

    return (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2


def himmelblau(x):
    """
    Minimum at [3,2], [-2.805118, 3.131312], [-3.779310, -3.283186], [3.584428, -1.848126] -> 0

    :param x:
    :return:
    """
    return _batched(himmelblau1d, x)


class HimmelblauObjective(GenericObjective):
    def __init__(
        self,
        search_bounds=np.array([[-5, 5], [-5, 5]]),
        color_normalizer=matplotlib.colors.LogNorm(),
    ):
        ndim = len(search_bounds)
        minima = np.array(
            [
                [3, 2],
                [-2.805118, 3.131312],
                [-3.779310, -3.283186],
                [3.584428, -1.848126],
            ]
        )
        super().__init__(himmelblau1d, search_bounds, minima, ndim, color_normalizer)


def crossintray1d(z):
    assert isinstance(z, (np.ndarray, np.generic))
    assert z.ndim == 1

    x = z[0]
    y = z[1]

    return (
        -0.0001
        * (
            abs(
                np.sin(x)
                * np.sin(y)
                * np.exp(abs(100 - np.sqrt(x ** 2 + y ** 2) / np.pi))
            )
            + 1
        )
        ** 0.1
    )


def crossintray(x):
    """

    :param x:
    :return:
    """
    return _batched(crossintray1d, x)


class CrossInTrayObjective(GenericObjective):
    def __init__(
        self,
        search_bounds=np.array([[-5, 5], [-5, 5]]),
        color_normalizer=matplotlib.colors.NoNorm(),
    ):
        ndim = len(search_bounds)
        minima = np.array(
            [
                [1.3491, -1.34941],
                [1.34941, 1.34941],
                [-1.34941, 1.34941],
                [-1.34941, -1.34941],
            ]
        )
        super().__init__(crossintray1d, search_bounds, minima, ndim, color_normalizer)


def eggholder1d(z):
    assert isinstance(z, (np.ndarray, np.generic))
    assert z.ndim == 1

    x = z[0]
    y = z[1]

    return -(y + 47) * np.sin(np.sqrt(abs(x / 2 + (y + 47)))) - x * np.sin(
        np.sqrt(abs(x - (y + 47)))
    )


def eggholder(x):
    """

    :param x:
    :return:
    """
    return _batched(eggholder1d, x)


class EggholderObjective(GenericObjective):
    def __init__(
        self,
        search_bounds=np.array([[-550, 550], [-420, 420]]),
        color_normalizer=matplotlib.colors.NoNorm(),
    ):
        ndim = len(search_bounds)
        minima = np.array([[512, 404.2319]])
        super().__init__(eggholder1d, search_bounds, minima, ndim, color_normalizer)


def _batched(fn, x, *args):
    assert isinstance(x, (np.ndarray, np.generic))

    if x.ndim == 1:
        return fn(x, *args)
    elif x.ndim == 2:
        return np.array([fn(x1d, *args) for x1d in x])
    else:
        raise ValueError(
            'Too many number of dimensions given to objective function "%s"'
            % fn.__name__
        )


def stier2020A1_1d(z):
    assert isinstance(z, (np.ndarray, np.generic))
    assert z.ndim == 1

    x = z[0]
    y = z[1]

    return 1421 + 2 * x ** 2 - 4 * x * y ** 2 + 2 * y + (x / 3) ** 6 + (y / 3) ** 6


class Stier2020A1Objective(GenericObjective):
    def __init__(
        self,
        search_bounds=np.array([[-12, 12], [-12, 12]]),
        color_normalizer=matplotlib.colors.LogNorm(),
    ):
        ndim = len(search_bounds)
        minima = np.array([[8.31373, -9.48875] * ndim])  # Not analytically proven
        super().__init__(stier2020A1_1d, search_bounds, minima, ndim, color_normalizer)


def stier2020A2_1d(z):
    assert isinstance(z, (np.ndarray, np.generic))
    assert z.ndim == 1

    x = z[0]
    y = z[1]

    return 1700 - 4 * x * y ** 2 + (0.5 * y) ** 3 + (x / 3) ** 6 + (y / 3) ** 6


class Stier2020A2Objective(GenericObjective):
    def __init__(
        self,
        search_bounds=np.array([[-15, 15], [-15, 15]]),
        color_normalizer=matplotlib.colors.LogNorm(),
    ):
        ndim = len(search_bounds)
        minima = np.array([[8.31373, -9.48875] * ndim])  # Not analytically proven
        super().__init__(stier2020A2_1d, search_bounds, minima, ndim, color_normalizer)


def stier2020B1d(z):
    assert isinstance(z, (np.ndarray, np.generic))
    assert z.ndim == 1

    x = z[0]
    y = z[1]

    # return np.sin(x)-x*y+(x/4)**4+(y/4)**4
    # return 2+np.sin(10*x)-np.sin(x)*y+(x/4)**4+(y/4)**4
    # return 10-np.sin(x)*y+(x/4)**4+(y/4)**4
    # return 8+2*np.sin(x+2*y)*y+(x/4)**4+(y/4)**4
    # return 20 + x - 1.5*(y-5) + 2 * np.sin(x + 2 * y) * y + (x / 4) ** 4 + (y / 4) ** 4
    return (
        20 + x - 1.8 * (y - 5) + 3 * np.sin(x + 2 * y) * y + (x / 4) ** 4 + (y / 4) ** 4
    )


class Stier2020BObjective(GenericObjective):
    def __init__(
        self,
        search_bounds=np.array([[-10, 10], [-10, 10]]),
        color_normalizer=matplotlib.colors.LogNorm(),
    ):
        ndim = len(search_bounds)
        minima = np.array([[0.87902855, 5.05756791] * ndim])  # Not analytically proven
        super().__init__(stier2020B1d, search_bounds, minima, ndim, color_normalizer)


def bahar1d(z):
    assert isinstance(z, (np.ndarray, np.generic))
    assert z.ndim == 1

    x = z[0]
    y = z[1]

    # return np.maximum(0, x)
    return np.maximum(0, x) ** 2 + (x / 4) ** 4 + (y / 4) ** 4


class BaharObjective(GenericObjective):
    def __init__(
        self,
        search_bounds=np.array([[-10, 10], [-10, 10]]),
        color_normalizer=matplotlib.colors.NoNorm(),
    ):
        ndim = len(search_bounds)
        minima = (
            np.array([[0.87902855, 5.05756791] * ndim]),
        )  # Not analytically proven
        super().__init__(bahar1d, search_bounds, minima, ndim, color_normalizer)
