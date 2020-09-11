from pathlib import Path
from typing import List

import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

from experiments.load_results import load_results_paper
from experiments.optimizer_names import names
from experiments.optimizer_styles import optimizer_style
from experiments.table2 import adtm_scores

path = Path(__file__).parent


def plot_per_task(scores_per_task: pd.DataFrame, optimizers_to_plot: List[str]):
    import seaborn as sns
    from matplotlib.patches import Patch
    sns.set()
    sns.set_style("white")

    # load RMSEs from csv
    rmses = pd.read_csv(
        Path(__file__).parent / 'rmse.csv',
        header=None, names=['task', 'rmse']
    ).set_index('task')['rmse']

    # show task in x, ADTM improvement over RS on the y-axis
    cols = {'rmse': rmses}
    for method in optimizers_to_plot:
        cols[method] = scores_per_task[method].reset_index()[['task', method]].set_index('task')[method]
    dd = pd.DataFrame(cols).sort_values(by='rmse').reset_index().rename(columns={'index': 'task'})
    dd['task_and_rmse'] = dd.apply(lambda x: f"{x.task} (%.2f)" % x.rmse, axis=1)
    styles, colors = zip(*[optimizer_style(method) for method in optimizers_to_plot])

    hatches = tuple(['///' if 'Copula' in m else None for m in optimizers_to_plot])

    fig, axes = plt.subplots(3, 9, figsize=(20, 5), sharex=True, sharey='row')
    axes = np.ravel(axes)
    for i, row in dd.iterrows():
        y = [row[m] for m in optimizers_to_plot]
        bars = axes[i].bar(x=range(len(colors)), height=y, color=colors, label=optimizers_to_plot)
        for bar, h in zip(bars, hatches):
            bar.set_hatch(h)

        axes[i].set_xlabel(row['task_and_rmse'], fontsize=14)
        axes[i].set_ylim([-1, 1])

    # plot legend on the last subplots
    custom_lines = []
    for c, h in zip(colors, hatches):
        p = Patch(facecolor=c, hatch=h)
        custom_lines.append(p)

    axes[-1].spines['right'].set_visible(False)
    axes[-1].spines['top'].set_visible(False)
    axes[-1].spines['left'].set_visible(False)
    axes[-1].spines['bottom'].set_visible(False)
    axes[-1].legend(custom_lines, optimizers_to_plot, fontsize=10, loc='center')

    plt.subplots_adjust(wspace=0.0)
    plt.xticks([], [])
    plt.tight_layout(h_pad=0, w_pad=0)

    filename = Path(__file__).parent / f'hpo/figures/ADTM_per_task.pdf'
    os.makedirs(filename.parent, exist_ok=True)
    print(filename)
    plt.savefig(str(filename))
    plt.show()


if __name__ == '__main__':
    df_paper = load_results_paper()

    optimizers_to_plot = [
        names.GCP_prior,
        names.CTS_prior,
        names.WS_BEST,
        names.AUTORANGE_GP,
        names.ABLR,
        names.ABLR_COPULA,
        names.SGPT,
        names.SGPT_COPULA,
    ]

    scores_per_blackbox, scores_per_task = adtm_scores(df_paper, optimizers_to_plot)

    plot_per_task(scores_per_task=scores_per_task, optimizers_to_plot=optimizers_to_plot)

