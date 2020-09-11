import gc
import logging
import sys
import traceback
from typing import Tuple, Callable

import numpy as np
from tqdm import tqdm

from blackbox import Blackbox
from misc import set_seed
from optimizer import Optimizer


def benchmark(
        num_evaluations: int,
        optimizer_factory: Callable[[], Optimizer],
        blackbox: Blackbox,
        candidates: np.array,
        num_seeds: int,
        verbose: bool = False,
) -> Tuple[np.array]:
    """
    For each seed, the optimizer is run 'num_evaluations'.
    :param num_evaluations:
    :param optimizer_factory:
    :param blackbox:
    :param candidates:
    :param num_seeds:
    :param verbose:
    :return: two tensors of shape (num_seeds, num_evaluations, X) where X = [input_dim, output_dim]
    """
    seeds = range(num_seeds)
    #if verbose:
    #    seeds = tqdm(seeds)
    seeds = tqdm(seeds)

    Xs = np.empty((num_seeds, num_evaluations, blackbox.input_dim))
    Xs[:] = np.nan
    ys = np.empty((num_seeds, num_evaluations, blackbox.output_dim))
    ys[:] = np.nan

    for seed in seeds:
        try:
            set_seed(seed)
            optimizer = optimizer_factory()
            for i in range(num_evaluations):
                x = optimizer.sample(candidates)
                y = blackbox(x)
                if verbose:
                    logging.info(f"criterion {y} for arguments {x}")
                optimizer.observe(x=x, y=y)
                Xs[seed, i] = x
                ys[seed, i] = y
                # memory leaks without gc, not sure why, perhaps a reference cycle
                gc.collect()
            del optimizer
        except Exception:
            print("seed evaluation failed")
            traceback.print_exc(file=sys.stdout)
            pass
    return Xs, ys