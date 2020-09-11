import logging
import random
from functools import partial

import numpy as np
import pytest
import torch

from blackbox import Blackbox, BlackboxOffline
from misc import set_seed
from misc.artificial_data import artificial_task1
from optimizer.gaussian_process import GP
from optimizer.gaussian_process_functional_prior import G3P
from optimizer.normalization_transforms import StandardTransform, GaussianTransform
from optimizer.thompson_sampling_functional_prior import TS
from optimizer.random_search import RS


Xy_train, X_test, y_test = artificial_task1()


@pytest.mark.parametrize("blackbox", [
    Blackbox(
        input_dim=2,
        output_dim=1,
        eval_fun=lambda x: x.sum(axis=-1, keepdims=True),
    ),
    BlackboxOffline(
        X=X_test,
        y=y_test,
    )
])
def test_blackbox_works_with_optimization(blackbox: Blackbox):
    logging.basicConfig(level=logging.INFO)
    seed = 3
    num_evaluations = 5
    optimizer_cls = RS

    set_seed(seed)

    optimizer = optimizer_cls(
        input_dim=blackbox.input_dim,
        output_dim=blackbox.output_dim,
        evaluations_other_tasks=Xy_train,
    )

    candidates = X_test

    for i in range(num_evaluations):
        x = optimizer.sample(candidates)
        y = blackbox(x)
        logging.info(f"criterion {y} for arguments {x}")
        optimizer.observe(x=x, y=y)


@pytest.mark.parametrize("optimizer_cls", [
    RS,
    # 5 gradient updates to makes it faster as we are only smoke-checking
    partial(TS, num_gradient_updates=5, normalization="standard"),
    partial(TS, num_gradient_updates=5, normalization="gaussian"),
    partial(GP, normalization="standard"),
    partial(GP, normalization="gaussian"),
    partial(G3P, num_gradient_updates=5, normalization="standard"),
])
@pytest.mark.parametrize("constrained_search", [False, True])
def test_smoke_optimizers(optimizer_cls, constrained_search: bool):
    logging.basicConfig(level=logging.INFO)
    num_evaluations = 10

    blackbox = Blackbox(
        input_dim=2,
        output_dim=1,
        eval_fun=lambda x: x.sum(axis=-1, keepdims=True),
    )

    optimizer = optimizer_cls(
        input_dim=blackbox.input_dim,
        output_dim=blackbox.output_dim,
        evaluations_other_tasks=Xy_train,
    )

    candidates = X_test

    for i in range(num_evaluations):
        if constrained_search:
            x = optimizer.sample(candidates)
        else:
            x = optimizer.sample()
        y = blackbox(x)
        logging.info(f"criterion {y} for arguments {x}")
        optimizer.observe(x=x, y=y)
