from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator


class DummyBaseline(BaseEstimator):
    """Randomly selects products from the product catalog.

    A baseline model that ignores all input features and predicts
    by uniformly sampling from the catalog of all known products.
    """

    def __init__(self, random_state: int = 42, num_predictions: int = 5):
        self.random_state = random_state
        self.num_predictions = num_predictions

    def fit(self, order_products: pd.DataFrame, y: pd.Series | None = None) -> DummyBaseline:
        self.catalog_ = np.sort(order_products["product_id"].unique())
        return self

    def predict(self, known_products: np.ndarray | None = None) -> np.ndarray:
        rng = np.random.default_rng(self.random_state)
        exclude = known_products if known_products is not None else np.array([], dtype=int)
        candidates = np.setdiff1d(self.catalog_, exclude)
        n = min(self.num_predictions, len(candidates))
        return rng.choice(candidates, size=n, replace=False)

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {"random_state": self.random_state, "num_predictions": self.num_predictions}

    def set_params(self, **params) -> DummyBaseline:
        for k, v in params.items():
            setattr(self, k, v)
        return self
