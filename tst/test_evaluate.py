import pytest

from experiments.evaluate_optimizer_task import evaluate


@pytest.mark.parametrize("optimizer", [
    "RS",
    "GP",
    "GCP",
    # slow:
    # "TS",
    "CTS",
    # "GP+prior",
    "GCP+prior",
])
def test_evaluate(optimizer: str):
    evaluate(
        optimizer=optimizer,
        task="electricity",
        num_seeds=2,
        num_evaluations=10,
        output_folder="/tmp/",
        prior="sklearn",
    )