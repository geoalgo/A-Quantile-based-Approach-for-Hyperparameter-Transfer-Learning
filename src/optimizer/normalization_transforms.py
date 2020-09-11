import numpy as np
from scipy import stats


class GaussianTransform:
    """
    Transform data into Gaussian by applying psi = Phi^{-1} o F where F is the truncated ECDF.
    :param y: shape (n, dim)
    """

    def __init__(self, y: np.array):
        assert y.ndim == 2
        self.dim = y.shape[1]
        self.sorted = y.copy()
        self.sorted.sort(axis=0)

    @staticmethod
    def z_transform(series, values_sorted=None):
        # applies truncated ECDF then inverse Gaussian CDF.
        if values_sorted is None:
            values_sorted = sorted(series)

        def winsorized_delta(n):
            return 1.0 / (4.0 * n ** 0.25 * np.sqrt(np.pi * np.log(n)))

        delta = winsorized_delta(len(series))

        def quantile(values_sorted, values_to_insert, delta):
            res = np.searchsorted(values_sorted, values_to_insert) / len(values_sorted)
            return np.clip(res, a_min=delta, a_max=1 - delta)

        quantiles = quantile(
            values_sorted,
            series,
            delta
        )

        quantiles = np.clip(quantiles, a_min=delta, a_max=1 - delta)

        return stats.norm.ppf(quantiles)

    def transform(self, y: np.array):
        """
        :param y: shape (n, dim)
        :return: shape (n, dim), distributed along a normal
        """
        assert y.shape[1] == self.dim
        # compute truncated quantile, apply gaussian inv cdf
        return np.stack([
            self.z_transform(y[:, i], self.sorted[:, i])
            for i in range(self.dim)
        ]).T


class StandardTransform:

    def __init__(self, y: np.array):
        assert y.ndim == 2
        self.dim = y.shape[1]
        self.mean = y.mean(axis=0, keepdims=True)
        self.std = y.std(axis=0, keepdims=True)

    def transform(self, y: np.array):
        z = (y - self.mean) / np.clip(self.std, a_min=0.001, a_max=None)
        return z


def from_string(name: str):
    assert name in ["standard", "gaussian"]
    mapping = {
        "standard": StandardTransform,
        "gaussian": GaussianTransform,
    }
    return mapping[name]