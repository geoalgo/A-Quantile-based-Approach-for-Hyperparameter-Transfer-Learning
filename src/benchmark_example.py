import logging
from functools import partial

import numpy as np
import matplotlib.pyplot as plt

from blackbox import BlackboxOffline
from blackbox.load_utils import evaluation_split_from_task
from optimizer.benchmark import benchmark
from optimizer.gaussian_process import GP
from optimizer.random_search import RS
from optimizer.thompson_sampling_functional_prior import TS


if __name__ == '__main__':
    task = "electricity"
    logging.basicConfig(level=logging.INFO)
    num_seeds = 20
    num_evaluations = 50

    Xys_train, (X_test, y_test) = evaluation_split_from_task(task)
    candidates = X_test

    blackbox = BlackboxOffline(
        X=X_test,
        y=y_test,
    )

    optimizers = {
        #"GP + prior": partial(G3P, normalization="standard"),
        #"GCP + prior": partial(G3P, normalization="gaussian"),
        "RS": RS,
        "TS": TS,
        "GP": GP,
    }

    res = {}
    for name, Optimizer_cls in optimizers.items():
        logging.info(f"evaluate {name}")
        optimizer_factory = partial(
            Optimizer_cls,
            input_dim=blackbox.input_dim,
            output_dim=blackbox.output_dim,
            evaluations_other_tasks=Xys_train,
        )
        X, y = benchmark(
            optimizer_factory=optimizer_factory,
            blackbox=blackbox,
            candidates=candidates,
            num_seeds=num_seeds,
            num_evaluations=num_evaluations,
            verbose=False,
        )
        res[name] = X, y
    print(res)

    fig, ax = plt.subplots()
    for name, (X, y) in res.items():
        y_best = np.minimum.accumulate(y, axis=1)
        mean = y_best.mean(axis=0)[:, 0]
        std = y_best.std(axis=0)[:, 0]
        ax.plot(mean, label=name)
        ax.fill_between(range(len(mean)), mean - std, mean + std, alpha=0.2)
    plt.legend()
    plt.show()
    plt.savefig(f"optimizer-comparison-{task}.pdf")
