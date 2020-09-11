from pathlib import Path

import pandas as pd
import numpy as np

deepar = 'DeepAR'
fcnet = 'FCNET'
xgboost = 'XGBoost'
nas102 = 'nas_bench102'
metric_error = 'metric_error'
metric_time = 'metric_time'


def evaluations_df(blackbox: str) -> pd.DataFrame:
    """
    :returns a dataframe where each row corresponds to one hyperparameter evaluated for one task.
    The hyperparamers columns are all prefixed by 'hp_', the metric columns (error, time, etc) are
    prefixed by 'metric_' and dataset information are prefixed by 'dataset_' (only available for
    DeepAR). Two columns 'task' and 'blackbox' contains the name of the task and of the blackbox.

    ## DeepAR
    Hyperparameters:
    * num_layers
    * num_cells
    * context_length_ratio, context_length_ratio = context_length / prediction_length
    * dropout_rate
    * learning_rate
    * num_batches_per_epoch

    Constants:
    * epochs = 100
    * early_stopping_patience = 5

    Dataset specific:
    * time_freq
    * prediction_length

    Metrics:
    * CRPS
    * train_loss
    * throughput
    * RMSE

    ## FCNET
    """
    assert blackbox in [deepar, fcnet, xgboost, nas102]
    df = pd.read_csv(Path(__file__).parent / f"offline_evaluations/{blackbox}.csv.zip")
    return df


if __name__ == '__main__':

    df = evaluations_df(deepar)

    import seaborn as sns
    import matplotlib.pyplot as plt
    df["hp_learning_rate"] = df.hp_learning_rate_log.apply(np.exp)
    df["hp_context_length_ratio"] = df.hp_context_length_ratio_log.apply(np.exp)
    df["hp_num_batches_per_epoch"] = df.hp_num_batches_per_epoch_log.apply(np.exp)

    ax = sns.scatterplot(data=df, x='hp_learning_rate', y='metric_CRPS', hue='task')
    plt.show()

    ax = sns.scatterplot(data=df, x='hp_learning_rate', y='metric_CRPS', hue='task')
    ax.set(xscale="log", yscale="log")
    plt.show()

    ax = sns.scatterplot(data=df, x='hp_context_length_ratio', y='metric_CRPS', hue='task')
    ax.set(yscale="log")
    plt.show()

    ax = sns.scatterplot(data=df, x='hp_num_batches_per_epoch', y='metric_time', hue='task')
    ax.set(xscale="log", yscale="log")
    plt.show()