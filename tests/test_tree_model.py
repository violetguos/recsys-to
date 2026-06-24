from pathlib import Path

import numpy as np
import pytest

from src.tree_model import TreeBaseline

MODEL_DIR = Path(__file__).resolve().parent.parent / "outputs" / "tree"
_REQUIRED = [MODEL_DIR / "model.joblib", MODEL_DIR / "catalog.npy",
             MODEL_DIR / "product_stats.parquet"]


pytestmark = pytest.mark.skipif(
    any(not p.exists() for p in _REQUIRED),
    reason="Trained tree model artifacts not found — run `uv run python main.py train-tree` first",
)


def test_tree_model_loads_and_predicts():
    model = TreeBaseline().load(MODEL_DIR)

    known = np.array([1, 2, 3], dtype=int)
    ctx = {"order_dow": 2, "order_hour_of_day": 10,
           "days_since_prior_order": 7.0, "order_number": 5}

    preds = model.predict(known_products=known, order_context=ctx)

    assert isinstance(preds, np.ndarray), f"expected ndarray, got {type(preds)}"
    assert preds.dtype == np.int64, f"expected int64, got {preds.dtype}"
    assert len(preds) == model.num_predictions, (
        f"expected {model.num_predictions} predictions, got {len(preds)}"
    )
    assert len(set(preds)) == len(preds), "predictions contain duplicates"
    assert not set(known) & set(preds), "predicted a known product"
    assert all(p in model.catalog_ for p in preds), "predicted unknown product IDs"
