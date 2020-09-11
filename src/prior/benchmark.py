import numpy as np
import pandas as pd

from blackbox.load_utils import evaluation_split_from_task, tasks
from optimizer.normalization_transforms import from_string
from prior.mlp_pytorch import ParametricPrior
from prior.mlp_sklearn import ParametricPriorSklearn

normalization = "gaussian"

rows = []
#tasks = [
#    'electricity',
#    # 'australian',
#    #'m4-Hourly',
#    #'m4-Daily',
#]
for task in tasks:

    Xys_train, (X_test, y_test) = evaluation_split_from_task(task)

    X_train = np.concatenate([X for X, y in Xys_train], axis=0)
    normalizer = from_string(normalization)
    z_train = np.concatenate([normalizer(y).transform(y) for X, y in Xys_train], axis=0)

    # y_test is only used for measuring RMSE on the prior as mentioned in the paper
    z_test = normalizer(y_test).transform(y_test)

    # todo normalization inside prior
    prior = ParametricPrior(
        X_train=X_train,
        y_train=z_train,
        num_gradient_updates=2000,
        num_decays=2,
        num_layers=3,
        num_hidden=50,
        dropout=0.1,
        lr=0.001,
    )

    mu_pred, sigma_pred = prior.predict(X_test)

    rmse = np.sqrt(np.square(mu_pred - z_test).mean())
    mae = np.abs(mu_pred - z_test).mean()
    row = {"task": task, "rmse": rmse, "mae": mae}
    rows.append(row)
    print(row)

df = pd.DataFrame(rows)
print(df.to_string())