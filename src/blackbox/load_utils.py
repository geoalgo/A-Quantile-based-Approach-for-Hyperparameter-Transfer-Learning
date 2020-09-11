import logging
from typing import Tuple, List

import numpy as np

from blackbox.offline import evaluations_df, deepar, fcnet, nas102, xgboost

blackbox_tasks = {
    nas102: [
        'cifar10',
        'cifar100',
        'ImageNet16-120'
    ],
    fcnet: [
        'naval',
        'parkinsons',
        'protein',
        'slice',
    ],
    deepar: [
        'm4-Hourly',
        'm4-Daily',
        'm4-Weekly',
        'm4-Monthly',
        'm4-Quarterly',
        'm4-Yearly',
        'electricity',
        'exchange-rate',
        'solar',
        'traffic',
    ],
    xgboost: [
        'a6a',
        'australian',
        'german.numer',
        'heart',
        'ijcnn1',
        'madelon',
        'skin_nonskin',
        'spambase',
        'svmguide1',
        'w6a'
    ],
}

error_metric = {
    deepar: 'metric_CRPS',
    fcnet: 'metric_error',
    nas102: 'metric_error',
    xgboost: 'metric_error',
}

tasks = [task for bb, tasks in blackbox_tasks.items() for task in tasks]


def evaluations_np(
        blackbox: str,
        test_task: str,
        metric_cols: List[str],
        min_max_features: bool = False
) -> Tuple[List[Tuple[np.array, np.array]], Tuple[np.array, np.array]] :
    """
    :param blackbox:
    :param test_task:
    :param metric_cols:
    :param min_max_features: whether to apply min-max scaling on input features
    :return: list of features/evaluations on train task and features/evaluations of the test task.
    """
    logging.info(f"retrieving metrics {metric_cols} of blackbox {blackbox} for test-task {test_task}")
    df = evaluations_df(blackbox=blackbox)

    assert test_task in df.task.unique()
    for c in metric_cols:
        assert c in df.columns

    Xy_dict = {}
    for task in sorted(df.task.unique()):
        mask = df.loc[:, 'task'] == task
        hp_cols = [c for c in sorted(df.columns) if c.startswith("hp_")]
        X = df.loc[mask, hp_cols].values
        y = df.loc[mask, metric_cols].values
        Xy_dict[task] = X, y

    # todo it would be better done as a post-processing step
    if blackbox in [fcnet, nas102]:
        # applies onehot encoding to *all* hp columns as all hps are categories for those two blackboxes
        # it would be nice to detect column types or pass it as an argument
        from sklearn.preprocessing import OneHotEncoder
        enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
        hp_cols = [c for c in sorted(df.columns) if c.startswith("hp_")]
        enc.fit(df.loc[:, hp_cols])
        for task, (X, y) in Xy_dict.items():
            X_features = enc.transform(X)
            Xy_dict[task] = X_features, y

    if min_max_features:
        # min-max scaling of input features
        from sklearn.preprocessing import MinMaxScaler
        X = np.vstack([X for (X, y) in Xy_dict.values()])
        scaler = MinMaxScaler().fit(X)
        Xy_dict = {t: (scaler.transform(X), y) for (t, (X, y)) in Xy_dict.items()}

    Xys_train = [Xy_dict[t] for t in df.task.unique() if t != test_task]
    Xy_test = Xy_dict[test_task]

    return Xys_train, Xy_test


def blackbox_from_task(task: str) -> str:

    for bb, tasks in blackbox_tasks.items():
        if task in tasks:
            return bb
    assert f"unknown task {task}"


def evaluation_split_from_task(test_task: str, min_max_features: bool = True) -> Tuple[np.array, np.array]:
    """
    :param test_task:
    :param min_max_features: whether inputs are maped to [0, 1] with min-max scaling
    :return: list of features/evaluations on train task and features/evaluations of the test task.
    """
    blackbox = blackbox_from_task(test_task)

    Xys_train, Xy_test = evaluations_np(
        blackbox=blackbox,
        test_task=test_task,
        metric_cols=[error_metric[blackbox]],
        min_max_features=min_max_features
    )
    return Xys_train, Xy_test


if __name__ == '__main__':
    Xys_train, (X_test, y_test) = evaluation_split_from_task("a6a")

    for task in [
        'electricity',
        'cifar10',
        'australian',
        'parkinsons',
    ]:
        Xys_train, (X_test, y_test) = evaluation_split_from_task(task)
        print(len(Xys_train), X_test.shape)