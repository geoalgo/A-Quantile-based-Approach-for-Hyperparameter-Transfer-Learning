from typing import Tuple

import numpy as np
from prior import Prior


class UnitPrior(Prior):
    def __init__(
            self,
            X_train: np.array,
            y_train: np.array
    ):
        super(UnitPrior, self).__init__(
            X_train=X_train,
            y_train=y_train,
        )

    def predict(self, X_test: np.array) -> Tuple[np.array, np.array]:
        return np.zeros_like(X_test[..., 0]), np.ones_like(X_test[..., 1])