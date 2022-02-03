import numpy as np
from eddysearch.search.gradient import derivative


def test_construct_default_objectives():
    # Arrange
    p_0 = 2

    def fn_square(x):
        return x**2

    # Act
    res = derivative(fn_square, p_0)

    # Assert
    assert res is not None
    assert res > 0
