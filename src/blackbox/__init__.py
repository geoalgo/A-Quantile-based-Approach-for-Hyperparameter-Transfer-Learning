from typing import Callable

import numpy as np


class Blackbox:
    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            eval_fun: Callable[[np.array], np.array],
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.eval_fun = eval_fun

    def __call__(self, x: np.array) -> np.array:
        """
        :param x: shape (input_dim,)
        :return: shape (output_dim,)
        """
        assert x.shape == (self.input_dim,)
        y = self.eval_fun(x)
        assert y.shape == (self.output_dim,)
        return y


class BlackboxOffline(Blackbox):
    def __init__(
            self,
            X: np.array,
            y: np.array,
    ):
        """
        A blackbox whose evaluations are already known.
        To evaluate a new point, we return the value of the closest known point.
        :param input_dim:
        :param output_dim:
        :param X: list of arguments evaluated, shape (n, input_dim)
        :param y: list of outputs evaluated, shape (n, output_dim)
        """
        assert len(X) == len(y)
        n, input_dim = X.shape
        n, output_dim = y.shape

        from sklearn.neighbors import KNeighborsRegressor
        proj = KNeighborsRegressor(n_neighbors=1).fit(X, y)

        super(BlackboxOffline, self).__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            eval_fun=lambda x: proj.predict(x.reshape(1, -1))[0]
        )