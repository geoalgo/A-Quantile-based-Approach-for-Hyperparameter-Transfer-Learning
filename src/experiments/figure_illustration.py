import seaborn as sns
import matplotlib.pyplot as plt

from pathlib import Path

import pandas as pd
import numpy as np

from optimizer.normalization_transforms import GaussianTransform


from blackbox.offline import evaluations_df, deepar
df = evaluations_df(deepar)

df = df[df.task.isin(["traffic", "electricity", "solar"])]

df["hp_learning_rate"] = df.hp_learning_rate_log.apply(np.exp)
df["hp_context_length_ratio"] = df.hp_context_length_ratio_log.apply(np.exp)
df["hp_num_batches_per_epoch"] = df.hp_num_batches_per_epoch_log.apply(np.exp)


#fig, axes = plt.subplots(1, 3)

# plot learning rate vs CRPS
#ax = sns.lmplot(x="hp_learning_rate", y="metric_CRPS", hue="task", data=df,)
#ax = sns.scatterplot(data=df, x='hp_learning_rate', y='metric_CRPS', hue='task')
#ax.set(xscale="log")
#ax.set_xlabel("x (learning rate)")
#ax.set_ylabel("y")

height = 4
aspect = 1.2
ax = sns.lmplot(
    x="hp_learning_rate", y="metric_CRPS", hue="task", ci=None,
    data=df, height=height, aspect=aspect, legend_out=False,
    fit_reg=False
)
ax.set(xscale="log", yscale="log")
ax.ax.set_ylim(0.02,)
ax.ax.set_xlabel("x (learning rate)")
ax.ax.set_ylabel("y")

plt.tight_layout()
plt.savefig("y_plot.jpg")
plt.show()

# plot learning rate vs CRPS mapped through psi = Phi^{-1} o F
for task in df.task.unique():
    y = df.loc[df.loc[:, "task"] == task, "metric_CRPS"].values.reshape(-1, 1)
    z = GaussianTransform(y).transform(y)
    df.loc[df.loc[:, "task"] == task, "z"] = z.reshape(-1)

#ax = sns.scatterplot(data=df, x='hp_learning_rate', y='z', hue='task')
#ax.set_ylabel("z = Psi(y)")
ax = sns.lmplot(
    x="hp_learning_rate",
    y="z",
    hue="task",
    legend=False,
    data=df,
    ci=None,
    height=height,
    aspect=aspect
)
ax.set(xscale="log")
ax.ax.set_xlabel("x (learning rate)")
ax.ax.set_ylabel("z")

plt.tight_layout()
plt.savefig("z_plot.jpg")
plt.show()


ax = sns.lmplot(
    x="hp_learning_rate",
    y="z",
    hue="task",
    legend=False,
    data=df,
    ci=None,
    height=height,
    aspect=aspect,
    fit_reg=False,
)
ax.set(xscale="log")
ax.ax.set_xlabel("x (learning rate)")
ax.ax.set_ylabel("z")

plt.tight_layout()
plt.savefig("z_scatter.jpg")
plt.show()
