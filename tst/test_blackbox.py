import numpy as np
from blackbox import BlackboxOffline


def test_blackbox():
    n = 20
    dim = 2
    X_test = np.random.rand(n, dim)
    y_test = np.random.rand(n, 1)
    blackbox = BlackboxOffline(
        X=X_test,
        y=y_test,
    )
    for x, y in zip(X_test, y_test):
        assert np.allclose(blackbox(x), y)
