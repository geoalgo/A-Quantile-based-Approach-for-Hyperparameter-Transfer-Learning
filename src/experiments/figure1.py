from pathlib import Path

import matplotlib.pyplot as plt

from blackbox.offline import deepar, fcnet, xgboost, nas102
from experiments.load_results import load_results_paper
from experiments.optimizer_names import names
from experiments.optimizer_styles import optimizer_style

path = Path(__file__).parent


def plot_optimizers(df, ax, blackbox, optimizers, legend: bool = False):
    df_plot = df.loc[df.optimizer.isin(optimizers), :]

    pivot_df = df_plot.loc[df_plot.blackbox == blackbox, :].groupby(
        ['blackbox', 'optimizer', 'iteration']
    )['ADTM'].mean().reset_index().pivot_table(
        index='iteration', columns='optimizer', values='ADTM'
    ).dropna()

    # reorder optimizers to original list order
    optimizers = [m for m in optimizers if m in pivot_df]
    style, color = zip(*[optimizer_style(optimizer) for optimizer in optimizers])
    pivot_df[optimizers].plot(
        ax=ax,
        title=blackbox,
        color=list(color),
        style=[a + b for a, b in style],
        # marker=list(marker),
        markevery=20,
        alpha=0.8,
        lw=2.5,
    )
    ax.grid()
    if blackbox == 'DeepAR':
        ax.set_ylim([None, 1e-2])
    if blackbox == 'fcnet':
        ax.set_ylim([None, 0.3])
    if blackbox == 'xgboost':
        ax.set_ylim([1e-2, 0.3])

    if blackbox == 'NAS':
        ax.set_xlim([None, 65])
        # ax.set_ylim([0.001, None])
    ax.set_yscale('log')
    ax.set_ylabel('ADTM')
    if not legend:
        ax.get_legend().remove()
    else:
        ax.legend(loc="upper right")


if __name__ == '__main__':
    df = load_results_paper()

    blackboxes = [deepar, fcnet, xgboost, nas102]

    optimizers_to_plot = [
        [
            names.RS,
            names.GP,
            names.AUTORANGE_GP,
            names.WS_BEST,
            names.ABLR,
            names.CTS_prior,
            names.GCP_prior,
            # 'BOHB', 'R-EA', 'REINFORCE',
        ],
        [
            names.GP,
            names.GP_prior,
            names.GCP,
            names.GCP_prior,
            names.TS_prior,
            names.CTS_prior,
        ]
    ]
    fig, axes = plt.subplots(4, 2, figsize=(10, 12), sharex='row', sharey='row')
    for i, blackbox in enumerate(blackboxes):
        for j, optimizers in enumerate(optimizers_to_plot):
            plot_optimizers(df, blackbox=blackbox, ax=axes[i, j], optimizers=optimizers, legend=(i == 0))
    plt.savefig("adtm.pdf")

    plt.show()
