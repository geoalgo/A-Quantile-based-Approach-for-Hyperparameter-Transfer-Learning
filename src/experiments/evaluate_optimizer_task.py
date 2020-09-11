import argparse
import logging
import os
from functools import partial
from pathlib import Path

import pandas as pd
import numpy as np

from blackbox import BlackboxOffline
from blackbox.load_utils import evaluation_split_from_task, blackbox_from_task
from optimizer.benchmark import benchmark
from optimizer.gaussian_process import GP
from optimizer.gaussian_process_functional_prior import G3P
from optimizer.random_search import RS
from optimizer.thompson_sampling_functional_prior import TS


def evaluate(
        task: str,
        optimizer: str,
        prior: str,
        num_seeds: int,
        num_evaluations: int,
        output_folder: str,
):
    optimizers = {
        "GP": partial(GP, normalization="standard"),
        "GCP": partial(GP, normalization="gaussian"),
        "RS": RS,
        "GP+prior": partial(G3P, normalization="standard", prior=prior),
        "GCP+prior": partial(G3P, normalization="gaussian", prior=prior),
        "TS": partial(TS, normalization="standard", prior=prior),
        "CTS": partial(TS, normalization="gaussian", prior=prior),
    }

    logging.info(f"Evaluating {optimizer} on {task} with {num_seeds} seeds and {num_evaluations} evaluations.")

    Xys_train, (X_test, y_test) = evaluation_split_from_task(test_task=task)
    candidates = X_test

    blackbox = BlackboxOffline(
        X=X_test,
        y=y_test,
    )

    X = np.vstack([X for X, _ in Xys_train] + [X_test])
    bounds = np.vstack([X.min(axis=0), X.max(axis=0)])

    optimizer_factory = partial(
        optimizers[optimizer],
        bounds=bounds,
        input_dim=blackbox.input_dim,
        output_dim=blackbox.output_dim,
        evaluations_other_tasks=Xys_train,
    )

    # (num_seeds, num_evaluations, dim)
    X, y = benchmark(
        optimizer_factory=optimizer_factory,
        blackbox=blackbox,
        candidates=candidates,
        num_seeds=num_seeds,
        num_evaluations=num_evaluations,
        verbose=False,
    )

    # (num_seeds, num_evaluations,)
    y = y.squeeze(axis=-1)

    df = pd.DataFrame([
        {"seed": seed, "iteration": iteration, "value": y[seed, iteration]}
        for seed in range(num_seeds)
        for iteration in range(num_evaluations)
    ])
    df["blackbox"] = blackbox_from_task(task)
    df["task"] = task
    df["optimizer"] = optimizer
    df.to_csv(Path(output_folder) / "result.csv.zip", index=False)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True)
    parser.add_argument('--optimizer', type=str, required=True)
    parser.add_argument('--prior', type=str, default="sklearn")
    parser.add_argument('--num_seeds', type=int, default=30)
    parser.add_argument('--num_evaluations', type=int, default=100)
    parser.add_argument('--output_folder', type=str)

    args = parser.parse_args()

    if args.output_folder is not None:
        output_folder = args.output_folder
    else:
        output_folder = os.getenv("SLURMAKER_JOBPATH")
        assert output_folder is not None, \
                "if you dont pass an output folder as argument, " \
            "you must set it with SLURMAKER_JOBPATH environment variable"

    logging.info(f"evaluating: {args}")

    for key, val in args.__dict__.items():
        logging.info(f"[{key}]:{val}")

    evaluate(
        task=args.task,
        optimizer=args.optimizer,
        num_seeds=args.num_seeds,
        num_evaluations=args.num_evaluations,
        output_folder=output_folder,
        prior=args.prior,
    )
