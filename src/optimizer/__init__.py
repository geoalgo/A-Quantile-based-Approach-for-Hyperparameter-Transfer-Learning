from typing import Optional, Tuple, List

import numpy as np


class Optimizer:
    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            bounds: Optional[np.array] = None,
            evaluations_other_tasks: Optional[List[Tuple[np.array, np.array]]] = None,
    ):
        """
        :param input_dim: input dimensions of blackbox arguments
        :param output_dim: output dimensions of blackbox output
        :param bounds: bounds on the space to sample with shape (2, input_dim), if not specified all coordinates are constrained to [-1, 1]
        :param evaluations_other_tasks: List of tuple X, y with shape (num_evaluations, input_dim) and
         (num_evaluations, output_dim). We pass as a separate list as the optimizer may need to group evaluations, for
         instance for normalizing the data.
        :param candidates:
        """
        self.input_dim = input_dim
        self.output_dim = output_dim

        if bounds is None:
            self.bounds = np.stack([
                np.ones(input_dim) * -1,
                np.ones(input_dim)
            ])
        else:
            self.bounds = bounds
        assert self.bounds.shape == (2, input_dim)

        if evaluations_other_tasks is not None:
            self.num_tasks = len(evaluations_other_tasks)
            for X, y in evaluations_other_tasks:
                assert len(X) == len(y)
                assert X.shape[1] == input_dim
                assert y.shape[1] == output_dim

    def sample(self, candidates: Optional[np.array] = None) -> np.array:
        """
        :param candidates: optionally a list of candidates when performing constrained search
        todo ensure that sampling happens inside this range
        :return: sample point with shape (input_dim,)
        """
        if candidates is not None:
            assert candidates.shape[1] == self.input_dim
        x = self._sample(candidates)
        assert x.shape == (self.input_dim,)
        return x

    def _sample(self, candidates: Optional[np.array] = None) -> np.array:
        return "override me"

    def observe(self, x: np.array, y: np.array):
        """
        Update the state after seeing an observation
        :param x: shape (input_dim,)
        :param y: shape (output_dim,)
        """
        assert x.shape == (self.input_dim,)
        assert y.shape == (self.output_dim,)
        self._observe(x, y)

    def _observe(self, x: np.array, y: np.array):
        pass
