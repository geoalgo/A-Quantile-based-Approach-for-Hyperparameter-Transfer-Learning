import numpy as np
import pytest

from optimizer.normalization_transforms import GaussianTransform, StandardTransform


@pytest.mark.parametrize("psi_cls", [GaussianTransform, StandardTransform])
def test_gaussian_transform(psi_cls):
    n = 1000
    tol = 0.05
    dim = 2
    y = np.random.uniform(size=(n, dim))
    psi = psi_cls(y)
    z = psi.transform(y)

    assert np.allclose(z.mean(axis=0), np.zeros((dim,)), rtol=tol, atol=tol)
    assert np.allclose(z.std(axis=0), np.ones((dim,)), rtol=tol)