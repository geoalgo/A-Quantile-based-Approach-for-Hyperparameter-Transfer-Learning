from typing import Optional

import pandas as pd
from pathlib import Path

from blackbox.offline import evaluations_df
from blackbox.load_utils import error_metric

path = Path(__file__).parent


def postprocess_results(df):
    # keeps only 70 iteration for NAS and 100 for other blackboxes as described in the paper
    # in case where optimizer fails, we put their evaluation value to the maximum of the task (note that when computing
    # the rolling best, this is equivalent of forward filling with the best value observed)
    task_max = df.groupby('task').max()['value']

    missing_mask = df.loc[:, "value"].isna()
    if sum(missing_mask) > 0:
        df.loc[missing_mask, 'value'] = df.loc[missing_mask, 'task'].apply(lambda task: task_max[task])
    # only keep 100 iteration
    df = df[(df.iteration < 100) & (df.seed < 30)]
    # for NAS, not more than 70 iteration as explained in the paper
    df = df[(df.blackbox != "nas_bench102") | (df.iteration < 70)]

    return df


def min_max_tasks():
    """
    :return: two series mapping task name to min and max respectively.
    """
    res = []
    for bb, metric in error_metric.items():
        offline_evals = evaluations_df(bb)
        res.append(offline_evals.groupby('task').agg(['min', 'max'])[metric])
    y_min = pd.concat([x['min'] for x in res])
    y_max = pd.concat([x['max'] for x in res])
    return y_min, y_max


def add_adtm(df):
    """
    :param df:
    :return: dataframe with a column ADTM added measuring (best - min_task) / (max_task - min_task)
    """
    df.loc[:, 'best'] = df.groupby(['task', 'optimizer', 'seed']).cummin().loc[:, 'value']
    y_min, y_max = min_max_tasks()
    df = df.join(other=y_min, on='task', lsuffix='dataset_')
    df = df.join(other=y_max, on='task', lsuffix='dataset_')
    df.loc[:, "ADTM"] = (df.loc[:, "best"] - df.loc[:, "min"]) / (df.loc[:, "max"] - df.loc[:, "min"])
    return df


def load_results(file):
    df = pd.read_csv(file)
    df = postprocess_results(df)
    return df


def load_results_paper(do_add_adtm: bool = True):
    df = load_results(path / "results_paper.csv.zip")
    if do_add_adtm:
        df = add_adtm(df)
    return df


def load_results_reimplem(filename: str = "results_reimplem.csv.zip"):
    return load_results(path / filename)

