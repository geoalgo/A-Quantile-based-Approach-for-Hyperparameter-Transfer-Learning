import logging
from typing import Optional

import numpy as np
import torch
from botorch import fit_gpytorch_model
from botorch.acquisition import ExpectedImprovement
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from botorch.utils.transforms import normalize
from gpytorch import ExactMarginalLogLikelihood
from gpytorch.constraints import GreaterThan
from gpytorch.likelihoods import GaussianLikelihood

from blackbox import Blackbox, BlackboxOffline
from constants import num_initial_random_draws
from misc import set_seed
from misc.artificial_data import artificial_task1
from optimizer import Optimizer
from optimizer.normalization_transforms import from_string
from optimizer.random_search import RS


class GP(Optimizer):

    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            bounds: Optional[np.array] = None,
            normalization: str = "standard",
            evaluations_other_tasks=None,
    ):
        super(GP, self).__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            evaluations_other_tasks=evaluations_other_tasks,
            bounds=bounds,
        )
        # maintains observations
        # (num_observations, input_dim)
        self.X_observed = torch.empty(size=(0, input_dim))
        # (num_observations, output_dim)
        self.y_observed = torch.empty(size=(0, output_dim))
        self.num_initial_random_draws = num_initial_random_draws
        self.normalizer = from_string(normalization)
        self.initial_sampler = RS(
            input_dim=input_dim,
            output_dim=output_dim,
            bounds=bounds,
        )
        self.bounds_tensor = torch.Tensor(self.bounds)

    def expected_improvement(self, model, best_f):
        return ExpectedImprovement(
            model=model,
            best_f=best_f,
            maximize=False,
        )

    def transform_outputs(self, y: np.array):
        psi = self.normalizer(y)
        z = psi.transform(y)
        return z

    def _sample(self, candidates: Optional[np.array] = None) -> np.array:
        if len(self.X_observed) < self.num_initial_random_draws:
            return self.initial_sampler.sample(candidates=candidates)
        else:
            z_observed = torch.Tensor(self.transform_outputs(self.y_observed.numpy()))

            # build and fit GP
            gp = SingleTaskGP(
                train_X=self.X_observed,
                train_Y=z_observed,
                # special likelihood for numerical Cholesky errors, following advice from
                # https://www.gitmemory.com/issue/pytorch/botorch/179/506276521
                likelihood=GaussianLikelihood(noise_constraint=GreaterThan(1e-3)),
            )
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            fit_gpytorch_model(mll)

            acq = self.expected_improvement(
                model=gp,
                best_f=z_observed.min(dim=0).values,
            )

            if candidates is None:
                candidate, acq_value = optimize_acqf(
                    acq, bounds=self.bounds_tensor, q=1, num_restarts=5, raw_samples=100,
                )
                return candidate[0]
            else:
                # (N,)
                ei = acq(torch.Tensor(candidates).unsqueeze(dim=-2))
                return torch.Tensor(candidates[ei.argmax()])

    def _observe(self, x: np.array, y: np.array):
        # remark, we could fit the GP there so that sampling several times avoid the cost of refitting the GP
        self.X_observed = torch.cat((self.X_observed, torch.Tensor(x).unsqueeze(dim=0)), dim=0)
        self.y_observed = torch.cat((self.y_observed, torch.Tensor(y).unsqueeze(dim=0)), dim=0)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    num_evaluations = 10

    Xy_train, X_test, y_test = artificial_task1(seed=0)

    print(y_test[0])
    set_seed(0)

    blackbox = BlackboxOffline(
        X=X_test,
        y=y_test,
    )

    optimizer = GP(
        input_dim=blackbox.input_dim,
        output_dim=blackbox.output_dim,
    )

    candidates = X_test

    for i in range(num_evaluations):
        #x = optimizer.sample(candidates)
        x = optimizer.sample()
        y = blackbox(x)
        logging.info(f"criterion {y} for arguments {x}")
        optimizer.observe(x=x, y=y)
