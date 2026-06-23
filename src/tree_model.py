from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from .features import (
    FEATURE_COLS,
    build_prediction_features,
    compute_product_stats,
    generate_training_features,
)


class TreeBaseline:
    def __init__(self, random_state: int = 42, num_predictions: int = 5,
                 max_depth: int = 10, n_estimators: int = 100,
                 min_samples_leaf: int = 50):
        self.random_state = random_state
        self.num_predictions = num_predictions
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.min_samples_leaf = min_samples_leaf
        self.model: RandomForestClassifier | None = None
        self.product_stats: pd.DataFrame | None = None
        self.catalog_: np.ndarray | None = None

    @property
    def _model_params(self) -> dict[str, Any]:
        return {
            "random_state": self.random_state,
            "num_predictions": self.num_predictions,
            "max_depth": self.max_depth,
            "n_estimators": self.n_estimators,
            "min_samples_leaf": self.min_samples_leaf,
        }

    def fit(self, order_products: pd.DataFrame | None = None) -> TreeBaseline:
        print("Computing product statistics...")
        self.product_stats = compute_product_stats()
        self.catalog_ = self.product_stats["product_id"].values

        print("Generating training features (this may take a few minutes)...")
        train_df = generate_training_features(
            product_stats=self.product_stats,
            n_negatives=1,
            max_orders=100_000,
        )

        X = train_df[FEATURE_COLS].to_numpy(dtype=np.float32)
        y = train_df["label"].to_numpy(dtype=np.int32)
        n_pos = int(y.sum())
        n_neg = len(y) - n_pos
        print(f"Training set: {len(train_df)} rows ({n_pos} positive, {n_neg} negative)")

        print("Training RandomForest...")
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
            n_jobs=-1,
            verbose=1,
        )
        self.model.fit(X, y)
        return self

    def predict(self, known_products: np.ndarray | None = None,
                order_context: dict[str, Any] | None = None) -> np.ndarray:
        if self.model is None or self.product_stats is None or self.catalog_ is None:
            raise RuntimeError("Model not trained. Call fit() first.")

        rng = np.random.default_rng(self.random_state)
        exclude = known_products if known_products is not None else np.array([], dtype=int)

        n_candidates = min(1000, len(self.catalog_))
        popular = self.product_stats.sort_values("popularity", ascending=False)
        candidates = popular[~popular["product_id"].isin(exclude)]["product_id"].values[:n_candidates]

        ctx = order_context or {}
        feat_df = build_prediction_features(ctx, candidates, self.product_stats)
        X_pred = feat_df[FEATURE_COLS].to_numpy(dtype=np.float32)

        probs = self.model.predict_proba(X_pred)[:, 1]
        best_idx = np.argsort(probs)[::-1][:self.num_predictions]
        return candidates[best_idx]

    def save(self, path: str | Path) -> None:
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, out / "model.joblib")
        self.product_stats.to_parquet(out / "product_stats.parquet", index=False)
        np.save(out / "catalog.npy", self.catalog_)
        with open(out / "config.json", "w") as f:
            json.dump(self._model_params, f, indent=2)
        print(f"Model saved to {out}/")

    def load(self, path: str | Path) -> TreeBaseline:
        out = Path(path)
        self.model = joblib.load(out / "model.joblib")
        self.product_stats = pd.read_parquet(out / "product_stats.parquet")
        self.catalog_ = np.load(out / "catalog.npy")
        with open(out / "config.json") as f:
            cfg = json.load(f)
            for k, v in cfg.items():
                setattr(self, k, v)
        print(f"Model loaded from {out}/")
        return self
