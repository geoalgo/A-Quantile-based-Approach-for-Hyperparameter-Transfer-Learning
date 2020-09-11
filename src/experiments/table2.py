from typing import List, Optional

import pandas as pd
import numpy as np
from pathlib import Path

from blackbox.offline import deepar, fcnet, xgboost, nas102
from experiments.load_results import load_results_paper
from experiments.optimizer_names import names

path = Path(__file__).parent


def adtm_scores(df, optimizers_to_plot = None, baseline: Optional[str] = "RS"):
    # return adtm table per blackbox and per dataset

    scores_df = df.groupby(["blackbox", "task", "optimizer", "iteration"])[
        "ADTM"
    ].mean().reset_index().pivot_table(
        values='ADTM',
        columns=['optimizer'],
        index=['blackbox', 'task', 'iteration'],
    )

    rel_scores = (scores_df[[baseline]].values - scores_df.values) / scores_df[[baseline]].values

    rel_scores_df = pd.DataFrame(rel_scores, index=scores_df.index, columns=scores_df.columns).reset_index(
        level=2).drop(
        columns='iteration')

    scores_per_task = rel_scores_df.groupby(['blackbox', 'task']).mean()

    avg_scores_per_blackbox = rel_scores_df.groupby(['blackbox']).mean()

    if optimizers_to_plot is not None:
        avg_scores_per_blackbox = avg_scores_per_blackbox[optimizers_to_plot]
        scores_per_task = scores_per_task[optimizers_to_plot]

    scores_per_blackbox = avg_scores_per_blackbox.T[["DeepAR", "FCNET", "XGBoost", "nas_bench102"]]

    return scores_per_blackbox, scores_per_task


def rank(scores_per_task: pd.DataFrame, blackboxes: List[str]):
    ranks = {}
    for b in blackboxes:
        ranks[b] = scores_per_task.transpose()[b].rank(ascending=False).mean(axis=1)
    return pd.DataFrame(ranks)


if __name__ == '__main__':

    df_paper = load_results_paper()

    print(df_paper.head())
    baseline = names.RS
    renamed_baseline = f"{names.RS} (baseline)"
    df_paper.optimizer = df_paper.optimizer.apply(lambda name: renamed_baseline if name == baseline else name)

    optimizers_to_plot = [
        renamed_baseline,
        names.TS_prior,
        names.CTS_prior,
        names.GP_prior,
        names.GCP,
        names.GCP_prior,
        names.GP,
        names.AUTORANGE_GP,
        names.WS_BEST,
        names.ABLR,
        names.ABLR_COPULA,
        names.SGPT,
        names.SGPT_COPULA,
        names.BOHB,
        names.REA,
        names.REINFORCE,
    ]

    scores_per_blackbox, scores_per_task = adtm_scores(
        df_paper,
        optimizers_to_plot,
        baseline=renamed_baseline,
    )

    print(scores_per_blackbox.to_string())
    print(scores_per_blackbox.to_latex(float_format='%.2f', na_rep='-'))

    rank_df = rank(scores_per_task=scores_per_task, blackboxes=[deepar, fcnet, xgboost, nas102])
    print(rank_df.to_string())
    print(rank_df.to_latex(float_format='%.1f', na_rep='-'))

    # generates "dtm (rank)" numbers dataframe so that it can be exported easily in latex
    dtm_and_rank_values = []
    for x, y in zip(scores_per_blackbox.values.reshape(-1), rank_df.values.reshape(-1)):
        dtm_and_rank_values.append("{:.2f}".format(x) + " (" + "{:.1f}".format(y) + ")")
    dtm_and_rank = pd.DataFrame(
        np.array(dtm_and_rank_values).reshape(rank_df.shape),
        index=rank_df.index,
        columns=rank_df.columns
    )
    print(dtm_and_rank.to_latex())



