from typing import Optional, List, Tuple
import numpy as np

from optimizer import Optimizer


class RS(Optimizer):
    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            bounds: Optional[np.array] = None,
            evaluations_other_tasks: Optional[List[Tuple[np.array, np.array]]] = None,
    ):
        super(RS, self).__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            bounds=bounds,
            evaluations_other_tasks=evaluations_other_tasks,
        )

    def _sample(self, candidates: Optional[np.array] = None) -> np.array:
        # if candidates are given, then pick a random one, else draw uniformly from domain
        if candidates is not None:
            return candidates[np.random.randint(low=0, high=len(candidates))]
        else:
            a, b = self.bounds
            random_draw = (b - a) * np.random.random(self.input_dim, ) + a
            return random_draw