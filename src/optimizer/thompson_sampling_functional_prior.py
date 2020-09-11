import logging
from typing import Optional, List, Tuple
import numpy as np

from constants import num_gradient_updates
from optimizer import Optimizer
from optimizer.normalization_transforms import from_string
from optimizer.random_search import RS
from prior.mlp_pytorch import ParametricPrior
from prior.mlp_sklearn import ParametricPriorSklearn


class TS(Optimizer):

    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            bounds: Optional[np.array] = None,
            evaluations_other_tasks: Optional[List[Tuple[np.array, np.array]]] = None,
            num_gradient_updates: int = num_gradient_updates,
            normalization: str = "standard",
            prior: str = "pytorch",
    ):
        super(TS, self).__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            evaluations_other_tasks=evaluations_other_tasks,
            bounds=bounds,
        )
        # todo add option for data transform
        assert evaluations_other_tasks is not None

        X_train = np.concatenate([X for X, y in evaluations_other_tasks], axis=0)
        normalizer = from_string(normalization)
        z_train = np.concatenate([normalizer(y).transform(y) for X, y in evaluations_other_tasks], axis=0)

        prior_dict = {
            "sklearn": ParametricPriorSklearn,
            "pytorch": ParametricPrior,
        }

        logging.info(f"fit prior {prior}")
        self.prior = prior_dict[prior](
            X_train=X_train,
            y_train=z_train,
            num_gradient_updates=num_gradient_updates,
        )
        logging.info("prior fitted")

    def _sample(self, candidates: Optional[np.array] = None) -> np.array:
        if candidates is None:
            num_random_candidates = 10000
            # since Thompson Sampling selects from discrete set of options,
            # when no candidates are given we draw random candidates
            candidates = self.draw_random_candidates(num_random_candidates)

        mu_pred, sigma_pred = self.prior.predict(candidates)
        samples = np.random.normal(loc=mu_pred, scale=sigma_pred)
        return candidates[np.argmin(samples)]

    def draw_random_candidates(self, num_random_candidates: int):
        random_sampler = RS(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            bounds=self.bounds,
        )
        candidates = np.stack([random_sampler.sample() for _ in range(num_random_candidates)])
        return candidates
