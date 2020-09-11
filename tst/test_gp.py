import logging

import pytest

from blackbox import Blackbox
from misc.artificial_data import artificial_task1
from optimizer.gaussian_process import GP


@pytest.mark.parametrize("constrained_search", [False, True])
@pytest.mark.parametrize("normalization", ["standard", "gaussian"])
def test_gp(constrained_search: bool, normalization: str):
    logging.basicConfig(level=logging.INFO)
    num_evaluations = 8

    Xy_train, X_test, y_test = artificial_task1()

    blackbox = Blackbox(
        input_dim=2,
        output_dim=1,
        eval_fun=lambda x: x.sum(axis=-1, keepdims=True),
    )

    optimizer = GP(
        input_dim=blackbox.input_dim,
        output_dim=blackbox.output_dim,
        normalization=normalization,
    )

    candidates = X_test

    for i in range(num_evaluations):
        x = optimizer.sample(candidates) if constrained_search else optimizer.sample()
        y = blackbox(x)
        logging.info(f"criterion {y} for arguments {x}")
        optimizer.observe(x=x, y=y)
