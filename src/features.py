from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_csv(name: str, **kwargs: Any) -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / f"{name}.csv", low_memory=False, **kwargs)


def compute_product_stats() -> pd.DataFrame:
    op = load_csv("order_products__prior", usecols=["product_id", "reordered", "add_to_cart_order"])
    products = load_csv("products", usecols=["product_id", "aisle_id", "department_id"])

    stats = op.groupby("product_id", sort=False).agg(
        popularity=("product_id", "count"),
        reorder_rate=("reordered", "mean"),
        avg_cart_pos=("add_to_cart_order", "mean"),
    ).reset_index()

    stats = products.merge(stats, on="product_id", how="left").fillna(0)

    stats["popularity"] = stats["popularity"].astype(int)
    return stats


def generate_training_features(
    product_stats: pd.DataFrame,
    n_negatives: int = 1,
    max_orders: int | None = 200_000,
    output_dir: str | Path | None = None,
) -> pd.DataFrame:
    orders = load_csv("orders", usecols=["order_id", "order_dow", "order_hour_of_day",
                                          "days_since_prior_order", "order_number"])
    op = load_csv("order_products__prior", usecols=["order_id", "product_id"])

    prior_order_ids = orders["order_id"].unique()
    op = op[op["order_id"].isin(prior_order_ids)]

    if max_orders is not None:
        rng = np.random.default_rng(42)
        sample_ids = rng.choice(prior_order_ids, size=min(max_orders, len(prior_order_ids)), replace=False)
        op = op[op["order_id"].isin(sample_ids)]

    pos = op.merge(orders, on="order_id", how="left")
    pos = pos.merge(product_stats, on="product_id", how="left")
    pos["label"] = 1
    pos["days_since_prior_order"] = pos["days_since_prior_order"].fillna(-1).astype(float)

    order_to_products = op.groupby("order_id")["product_id"].apply(np.array).to_dict()
    all_pids = product_stats["product_id"].values
    stats_lookup = product_stats.set_index("product_id")
    rng = np.random.default_rng(42)

    neg_rows: list[dict[str, Any]] = []

    for oid, pos_arr in order_to_products.items():
        neg_candidates = np.setdiff1d(all_pids, pos_arr, assume_unique=True)
        n_neg = min(len(pos_arr) * n_negatives, len(neg_candidates))
        neg_pids = rng.choice(neg_candidates, size=n_neg, replace=False)

        ord_row = orders[orders["order_id"] == oid].iloc[0]
        dspo = ord_row["days_since_prior_order"] if pd.notna(ord_row["days_since_prior_order"]) else -1

        for pid in neg_pids:
            ps = stats_lookup.loc[pid]
            neg_rows.append({
                "order_dow": ord_row["order_dow"],
                "order_hour_of_day": ord_row["order_hour_of_day"],
                "days_since_prior_order": float(dspo),
                "order_number": ord_row["order_number"],
                "product_id": int(pid),
                "aisle_id": int(ps["aisle_id"]),
                "department_id": int(ps["department_id"]),
                "popularity": int(ps["popularity"]),
                "reorder_rate": float(ps["reorder_rate"]),
                "avg_cart_pos": float(ps["avg_cart_pos"]),
                "label": 0,
            })

    neg = pd.DataFrame(neg_rows)

    feature_cols = ["order_dow", "order_hour_of_day", "days_since_prior_order",
                    "order_number", "product_id", "aisle_id", "department_id",
                    "popularity", "reorder_rate", "avg_cart_pos", "label"]
    features = pd.concat([pos[feature_cols], neg[feature_cols]], ignore_index=True)

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        features.to_parquet(out / "train_features.parquet", index=False)

    return features


FEATURE_COLS = [
    "order_dow", "order_hour_of_day", "days_since_prior_order",
    "order_number", "aisle_id", "department_id",
    "popularity", "reorder_rate", "avg_cart_pos",
]


def build_prediction_features(
    order_context: dict[str, Any],
    candidate_ids: np.ndarray,
    product_stats: pd.DataFrame,
) -> pd.DataFrame:
    stats_lookup = product_stats.set_index("product_id")
    rows: list[dict[str, Any]] = []

    for pid in candidate_ids:
        ps = stats_lookup.loc[pid]
        rows.append({
            "order_dow": order_context.get("order_dow", 0),
            "order_hour_of_day": order_context.get("order_hour_of_day", 12),
            "days_since_prior_order": order_context.get("days_since_prior_order", -1),
            "order_number": order_context.get("order_number", 1),
            "aisle_id": int(ps["aisle_id"]),
            "department_id": int(ps["department_id"]),
            "popularity": int(ps["popularity"]),
            "reorder_rate": float(ps["reorder_rate"]),
            "avg_cart_pos": float(ps["avg_cart_pos"]),
        })

    return pd.DataFrame(rows)
