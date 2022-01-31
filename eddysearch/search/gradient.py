import numpy as np

from eddysearch.strategy import SearchStrategy


def derivative(fn, a, method="central", h=0.001):
    """Compute the difference formula for f'(a) with step size h.

    Parameters
    ----------
    fn : function
        Vectorized function of multiple variables
    a : number
        Compute derivative at x = a
    method : string
        Difference formula: 'forward', 'backward' or 'central'
    h : number
        Step size in difference formula

    Returns
    -------
    float
        Difference formula:
            central: f(a+h) - f(a-h))/2h
            forward: f(a+h) - f(a))/h
            backward: f(a) - f(a-h))/h
    """
    if hasattr(a, "ndim"):
        if a.ndim > 2:
            raise ValueError(
                "Input for derivative() computation might have at max two dimensions (first batch, then input dimensions to objective)."
            )
        if a.ndim == 2:
            return np.array([derivative(fn, p, method, h) for p in a])

    diffs = np.eye(len(a)) * h
    params = np.stack([a] * len(a))
    if method == "central":
        # return (fn(a + h) - fn(a - h))/(2*h)
        return np.array(
            [(fn(p + d) - fn(p - d)) / (2 * h) for p, d in zip(params, diffs)]
        )
    elif method == "forward":
        # return (fn(a + h) - fn(a))/h
        return np.array([(fn(p + d) - fn(p)) / h for p, d in zip(params, diffs)])
    elif method == "backward":
        # return (fn(a) - fn(a - h))/h
        return np.array([(fn(p) - fn(p - d)) / h for p, d in zip(params, diffs)])
    else:
        raise ValueError("Method must be 'central', 'forward' or 'backward'.")


class GradientSearch(SearchStrategy):
    _updates = []
    _current_step = 0
    _last_update = 0
    _current_pos = None
    _track_updates = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def derivative(self, a, h=10e-8, method="central"):
        return derivative(self.objective, a, method=method, h=h)

    def initialize(self):
        raise NotImplementedError(
            "Your gradient descent algorithm has to specify an initial position."
        )

    def compute_update(self):
        raise NotImplementedError(
            "Your gradient descent algorithm has to implement an update routine for the current search status."
        )

    @property
    def track_updates(self):
        return self._track_updates

    @track_updates.setter
    def track_updates(self, flag):
        self._track_updates = True if flag else False

    @property
    def last_update(self):
        return self._last_update

    @property
    def current_step(self):
        return self._current_step

    @current_step.setter
    def current_step(self, value):
        self._current_step = int(value)

    @property
    def position(self):
        return self._current_pos

    def _reset_values(self):
        self._updates = []
        self._current_step = 0
        self._last_update = 0
        self._current_pos = None

    def start(self, *args, **kwargs):
        super().start(*args, **kwargs)

        self._reset_values()

        self._current_pos = self.initialize()

        # Do an initial evaluation of the objective for the initial position
        self.objective(self._current_pos)
        print("Starting at %s" % self._current_pos)

    def step(self):
        self.current_step += 1

        # Compute gradients and perform an update
        update = self.compute_update()

        # Track last update value and possibly also track it in a history list of updates
        self._last_update = update
        if self.track_updates:
            self._updates.append(update)

        # Perform the update-step -- updates are subtracted, not added! Thus the last_update is usually positive
        self._current_pos -= update

        # Make sure our position is not running out of allowed bounds
        # self._current_pos = np.minimum(np.maximum(self._current_pos, self._lower), self._upper)

    def has_finished(self) -> bool:
        return False  # never stop

    def end(self):
        pass

    def __str__(self):
        return "GradientSearch()"


class SGDSearch(GradientSearch):
    def __init__(self, *args, learning_rate=0.01, **kwargs):
        super().__init__(*args, **kwargs)
        self._learning_rate = learning_rate

    def initialize(self):
        return np.random.uniform(self._lower, self._upper)

    def compute_update(self):
        grads = self.derivative(self._current_pos)
        return self._learning_rate * grads

    def __str__(self):
        return "SGDSearch(lr=%s)" % self._learning_rate


class MomentumSGDSearch(SGDSearch):
    def __init__(self, *args, momentum=0.9, **kwargs):
        super().__init__(*args, **kwargs)
        self._momentum = momentum

    def compute_update(self):
        grads = self.derivative(self._current_pos)
        update = self._momentum * self.last_update + self._learning_rate * grads
        return -update

    def __str__(self):
        return "MomentumSGDSearch(lr=%s, momentum=%s)" % (
            self._learning_rate,
            self._momentum,
        )


class NesterovMomentumSGDSearch(SGDSearch):
    def __init__(self, *args, momentum=0.9, **kwargs):
        super().__init__(*args, **kwargs)
        self._momentum = momentum

    def compute_update(self):
        current_momentum = self._momentum * self.last_update
        grads = self.derivative(self._current_pos - current_momentum)
        update = current_momentum + self._learning_rate * grads
        return update

    def __str__(self):
        return "NesterovMomentumSGDSearch(lr=%s, momentum=%s)" % (
            self._learning_rate,
            self._momentum,
        )


class AdamSGDSearch(SGDSearch):
    def __init__(self, *args, beta1=0.9, beta2=0.999, epsilon=10e-8, **kwargs):
        super().__init__(*args, **kwargs)
        self._beta1 = beta1
        self._beta2 = beta2
        self._epsilon = epsilon
        self.track_updates = True

    def start(self, *args, **kwargs):
        super().start(*args, **kwargs)
        self._first_moment_estimate = 0
        self._second_moment_estimate = 0

    def compute_update(self):
        grads = self.derivative(self._current_pos)
        self._first_moment_estimate = (
            self._beta1 * self._first_moment_estimate + (1 - self._beta1) * grads
        )
        self._second_moment_estimate = self._beta2 * self._second_moment_estimate + (
            1 - self._beta2
        ) * (grads ** 2)
        bias_correction1 = 1 - self._beta1 ** self.current_step
        bias_correction2 = 1 - self._beta2 ** self.current_step

        # first_moment_avg = self._first_moment_estimate / bias_correction1
        # second_moment_avg = self._second_moment_estimate / bias_correction2
        # update = (self._learning_rate * first_moment_avg) / (np.sqrt(second_moment_avg) + self._epsilon)
        # Using trick from https://github.com/pytorch/pytorch/blob/cd9b27231b51633e76e28b6a34002ab83b0660fc/torch/optim/adam.py
        step_size = self._learning_rate * np.sqrt(bias_correction2) / bias_correction1
        update = step_size + self._first_moment_estimate / (
            np.sqrt(self._second_moment_estimate) + self._epsilon
        )

        return update

    def __str__(self):
        return "AdamSGDSearch(lr=%s, beta1=%s, beta2=%s, epsilon=%s)" % (
            self._learning_rate,
            self._beta1,
            self._beta2,
            self._epsilon,
        )
