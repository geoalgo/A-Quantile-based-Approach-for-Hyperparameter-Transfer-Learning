import numpy as np
from prior.mlp_pytorch import ParametricPrior

num_train_examples = 10000
num_test_examples = num_train_examples
dim = 2
num_gradient_updates = 200
lr = 1e-2


def make_random_X_y(num_examples: int, dim: int, noise_std: float):
    X = np.random.rand(num_examples, dim)
    noise = np.random.normal(scale=noise_std, size=(num_examples, 1))
    y = X.sum(axis=-1, keepdims=True) + noise
    return X, y


def test_mu_fit():
    # test that parametric prior can recover a simple linear function for the mean
    noise_std = 0.01
    X_train, y_train = make_random_X_y(num_examples=num_train_examples, dim=dim, noise_std=noise_std)
    prior = ParametricPrior(
        X_train=X_train,
        y_train=y_train,
        num_gradient_updates=num_gradient_updates,
        num_decays=1,
        # smaller network for UT speed
        num_layers=2,
        num_hidden=20,
        dropout=0.0,
        lr=lr
    )
    X_test, y_test = make_random_X_y(num_examples=num_test_examples, dim=dim, noise_std=noise_std)
    mu_pred, sigma_pred = prior.predict(X_test)

    mu_l1_error = np.abs(mu_pred - y_test).mean()
    print(mu_l1_error)
    assert mu_l1_error < 0.3


def test_sigma_fit():
    # test that parametric prior can recover a simple constant function for the variance
    noise_std = 0.5
    X_train, y_train = make_random_X_y(num_examples=num_train_examples, dim=dim, noise_std=noise_std)

    prior = ParametricPrior(
        X_train=X_train,
        y_train=y_train,
        num_gradient_updates=num_gradient_updates,
        num_decays=1,
        num_layers=2,
        num_hidden=20,
        dropout=0.0,
        lr=lr
    )

    X_test, y_test = make_random_X_y(num_examples=num_test_examples, dim=dim, noise_std=noise_std)
    mu_pred, sigma_pred = prior.predict(X_test)

    sigma_l1_error = (sigma_pred.mean() - noise_std)
    assert sigma_l1_error < 0.05
