from typing import Optional, Tuple, Callable, Union, List
import logging

import numpy as np
import torch
from gpytorch import ExactMarginalLogLikelihood
from gpytorch.constraints import GreaterThan
from gpytorch.likelihoods import GaussianLikelihood
from torch import Tensor
from torch.distributions import Normal
from botorch import fit_gpytorch_model
from botorch.acquisition import ExpectedImprovement, ScalarizedObjective
from botorch.models import SingleTaskGP
from botorch.models.model import Model
from botorch.optim import optimize_acqf
from botorch.utils.transforms import t_batch_mode_transform

from blackbox import Blackbox
from constants import num_gradient_updates
from misc.artificial_data import artificial_task1
from optimizer.gaussian_process import GP
from optimizer.thompson_sampling_functional_prior import TS


def residual_transform(y, mu_pred, sigma_pred):
    return (y - mu_pred) / sigma_pred


def residual_transform_inv(z, mu_pred, sigma_pred):
    return z * sigma_pred + mu_pred


def scale_posterior(mu_posterior, sigma_posterior, mu_est, sigma_est):
    mean = mu_posterior * sigma_est + mu_est
    sigma = (sigma_posterior * sigma_est)
    return mean, sigma


class ShiftedExpectedImprovement(ExpectedImprovement):
    """
    Applies ExpectedImprovement taking care to shift residual posterior with the predicted
    prior mean and variance
    :param model:
    :param best_f: best value observed (not residual but actual value)
    :param mean_std_predictor:
    :param objective:
    :param maximize:
    """

    def __init__(
            self,
            model: Model,
            best_f: Union[float, Tensor],
            mean_std_predictor: Callable[[np.array], Tuple[np.array, np.array]],
            objective: Optional[ScalarizedObjective] = None,
            maximize: bool = True,
    ) -> None:
        super(ShiftedExpectedImprovement, self).__init__(model=model, best_f=best_f, objective=objective,
                                                         maximize=maximize)
        self.mean_std_predictor = mean_std_predictor

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor) -> Tensor:
        """
        :param X: A (..., 1, input_dim) batched tensor of input_dim design points.
                Expected Improvement is computed for each point individually,
                i.e., what is considered are the marginal posteriors, not the
                joint.
        :return:  A (...) tensor of Expected Improvement values at the
            given design points `X`.
        """

        with torch.no_grad():
            # both (..., 1,)
            # (..., input_dim)
            X_features = X.detach().numpy().squeeze(1)
            mu_est, sigma_est = self.mean_std_predictor(X_features)

            # both (..., 1, 1)
            mu_est = torch.Tensor(mu_est).unsqueeze(1)
            sigma_est = torch.Tensor(sigma_est).unsqueeze(1)

        posterior = self._get_posterior(X=X)

        mean, sigma = scale_posterior(
            mu_posterior=posterior.mean,
            sigma_posterior=posterior.variance.clamp_min(1e-6).sqrt(),
            mu_est=mu_est,
            sigma_est=sigma_est,
        )

        u = (mean - self.best_f.expand_as(mean)) / sigma
        if not self.maximize:
            u = -u
        normal = Normal(torch.zeros_like(u), torch.ones_like(u))
        ucdf = normal.cdf(u)
        updf = torch.exp(normal.log_prob(u))
        ei = sigma * (updf + u * ucdf)

        return ei.squeeze(dim=-1).squeeze(dim=-1)


class ShiftedThompsonSampling(ExpectedImprovement):
    """
    Applies Thompson sampling taking care to shift residual posterior with the predicted
    prior mean and variance
    :param model:
    :param best_f:
    :param mean_std_predictor:
    :param objective:
    :param maximize:
    """

    def __init__(
            self,
            model: Model,
            best_f: Union[float, Tensor],
            mean_std_predictor: Callable[[np.array], Tuple[np.array, np.array]],
            objective: Optional[ScalarizedObjective] = None,
            maximize: bool = True,
    ) -> None:
        super(ShiftedThompsonSampling, self).__init__(model=model, best_f=best_f, objective=objective,
                                                         maximize=maximize)
        self.mean_std_predictor = mean_std_predictor

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor) -> Tensor:
        """
        :param X: A `... x 1 x d`-dim batched tensor of `d`-dim design points.
                Expected Improvement is computed for each point individually,
                i.e., what is considered are the marginal posteriors, not the
                joint.
        :return:  A `...` tensor of Expected Improvement values at the
            given design points `X`.
        """

        with torch.no_grad():
            # both (..., 1,)
            mu_est, sigma_est = self.mean_std_predictor(X)

        posterior = self._get_posterior(X=X)

        mean, sigma = scale_posterior(
            mu_posterior=posterior.mean,
            sigma_posterior=posterior.variance.clamp_min(1e-9).sqrt(),
            mu_est=mu_est,
            sigma_est=sigma_est,
        )
        normal = Normal(torch.zeros_like(mean), torch.ones_like(mean))
        u = normal.sample() * sigma + mean
        if not self.maximize:
            u = -u
        return u.squeeze(dim=-1).squeeze(dim=-1)


class G3P(GP):

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
        super(G3P, self).__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            bounds=bounds,
            normalization=normalization,
        )

        self.initial_sampler = TS(
            input_dim=input_dim,
            output_dim=output_dim,
            evaluations_other_tasks=evaluations_other_tasks,
            num_gradient_updates=num_gradient_updates,
            normalization=normalization,
            prior=prior,
        )

    def _sample(self, candidates: Optional[np.array] = None) -> np.array:
        if len(self.X_observed) < self.num_initial_random_draws:
            return self.initial_sampler.sample(candidates=candidates)
        else:
            z_observed = torch.Tensor(self.transform_outputs(self.y_observed.numpy()))

            with torch.no_grad():
                # both (n, 1)
                #mu_pred, sigma_pred = self.thompson_sampling.prior(self.X_observed)
                mu_pred, sigma_pred = self.initial_sampler.prior.predict(self.X_observed)
                mu_pred = torch.Tensor(mu_pred)
                sigma_pred = torch.Tensor(sigma_pred)

            # (n, 1)
            r_observed = residual_transform(z_observed, mu_pred, sigma_pred)

            # build and fit GP on residuals
            gp = SingleTaskGP(
                train_X=self.X_observed,
                train_Y=r_observed,
                likelihood=GaussianLikelihood(noise_constraint=GreaterThan(1e-3)),
            )
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            fit_gpytorch_model(mll)

            acq = ShiftedExpectedImprovement(
                model=gp,
                best_f=z_observed.min(dim=0).values,
                mean_std_predictor=self.initial_sampler.prior.predict,
                maximize=False,
            )

            if candidates is None:
                candidate, acq_value = optimize_acqf(
                    acq,
                    bounds=self.bounds_tensor,
                    q=1,
                    num_restarts=5,
                    raw_samples=100,
                )
                # import matplotlib.pyplot as plt
                # x = torch.linspace(-1, 1).unsqueeze(dim=-1)
                # x = torch.cat((x, x * 0), dim=1)
                # plt.plot(x[:, 0].flatten().tolist(), acq(x.unsqueeze(dim=1)).tolist())
                # plt.show()
                return candidate[0]
            else:
                # (N,)
                ei = acq(torch.Tensor(candidates).unsqueeze(dim=-2))
                return torch.Tensor(candidates[ei.argmax()])


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    num_evaluations = 10

    Xy_train, X_test, y_test = artificial_task1()

    blackbox = Blackbox(
        input_dim=2,
        output_dim=1,
        eval_fun=lambda x: x.sum(axis=-1, keepdims=True),
    )

    optimizer = G3P(
        input_dim=blackbox.input_dim,
        output_dim=blackbox.output_dim,
        evaluations_other_tasks=Xy_train,
        num_gradient_updates=2,
    )

    candidates = X_test

    for i in range(num_evaluations):
        x = optimizer.sample(candidates)
        #x = optimizer.sample()
        y = blackbox(x)
        logging.info(f"criterion {y} for arguments {x}")
        optimizer.observe(x=x, y=y)
