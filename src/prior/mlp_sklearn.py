import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from constants import num_gradient_updates
from prior import Prior


class ParametricPriorSklearn(Prior):
    def __init__(
            self,
            X_train: np.array,
            y_train: np.array,
            num_gradient_updates: int = num_gradient_updates,
    ):
        self.estimator = MLPRegressor(
            activation='relu',
            hidden_layer_sizes=(50, 50, 50),
            learning_rate='adaptive',
            verbose=False,
            max_iter=num_gradient_updates,
            tol=1e-6,
            early_stopping=True,
        )
        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(X_train)
        self.estimator.fit(X, y_train.ravel())

    def predict(self, X):
        X = self.scaler.transform(X)
        mu = self.estimator.predict(X).reshape((-1, 1))
        sigma = np.ones_like(mu)
        return mu, sigma


if __name__ == '__main__':

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


    # test that parametric prior can recover a simple linear function for the mean
    noise_std = 0.01
    X_train, y_train = make_random_X_y(num_examples=num_train_examples, dim=dim, noise_std=noise_std)
    prior = ParametricPriorSklearn(
        X_train=X_train,
        y_train=y_train,
        #num_gradient_updates=num_gradient_updates,
        #num_decays=2,
        # smaller network for UT speed
        #num_layers=2,
        #num_hidden=20,
        #dropout=0.0,
        #lr=lr
    )
    X_test, y_test = make_random_X_y(num_examples=num_test_examples, dim=dim, noise_std=noise_std)
    mu_pred, sigma_pred = prior.predict(X_test)

    mu_l1_error = np.abs(mu_pred - y_test).mean()
    print(mu_l1_error)
    assert mu_l1_error < 0.2
