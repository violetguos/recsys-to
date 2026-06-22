from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .models import DummyBaseline


@dataclass
class EvalResult:
    model_name: str
    config: dict[str, Any]
    metrics: dict[str, float] = field(default_factory=dict)
    per_order: list[dict[str, Any]] = field(default_factory=list)
    num_orders: int = 0
    total_products_in_orders: int = 0
    avg_order_size: float = 0.0


def recall_at_k(predictions: set[int], targets: set[int], k: int) -> float:
    if len(targets) == 0:
        return 0.0
    hits = len(predictions & targets)
    return hits / min(k, len(targets))


def precision_at_k(predictions: set[int], targets: set[int], k: int) -> float:
    if k == 0:
        return 0.0
    hits = len(predictions & targets)
    return hits / k


METRIC_REGISTRY: dict[str, Callable[[set[int], set[int], int], float]] = {
    "recall_at_k": recall_at_k,
    "precision_at_k": precision_at_k,
}


def split_order_products(
    row: pd.Series,
    known_ratio: float = 0.5,
    split_by: str = "add_to_cart_order",
) -> tuple[np.ndarray, np.ndarray]:
    products = np.asarray(row["products"])
    if split_by == "add_to_cart_order":
        order = np.asarray(row["add_to_cart_order"])
        sorter = np.argsort(order)
    else:
        rng = np.random.default_rng(42)
        sorter = rng.permutation(len(products))

    sorted_products = products[sorter]
    split_idx = max(1, int(len(sorted_products) * known_ratio))
    known = sorted_products[:split_idx]
    target = sorted_products[split_idx:]
    return known, target


def evaluate_baseline(
    model: DummyBaseline,
    order_products: pd.DataFrame,
    known_ratio: float = 0.5,
    split_by: str = "add_to_cart_order",
    num_predictions: int = 5,
    metrics: list[str] | None = None,
) -> EvalResult:
    if metrics is None:
        metrics = ["recall_at_k", "precision_at_k"]

    grouped = (
        order_products.groupby("order_id")
        .agg(products=("product_id", list), add_to_cart_order=("add_to_cart_order", list))
        .reset_index()
    )

    result = EvalResult(
        model_name=model.__class__.__name__,
        config={
            "known_ratio": known_ratio,
            "split_by": split_by,
            "num_predictions": num_predictions,
            "metrics": metrics,
        },
        num_orders=len(grouped),
    )

    all_metrics_acc: dict[str, float] = {m: 0.0 for m in metrics}
    k = num_predictions

    for _, row in grouped.iterrows():
        known, target = split_order_products(row, known_ratio, split_by)
        preds = set(model.predict(known_products=known))
        target_set = set(target)

        row_result = {
            "order_id": row["order_id"],
            "num_known": len(known),
            "num_target": len(target),
            "predictions": sorted(preds),
            "target": sorted(target_set),
        }

        for metric_name in metrics:
            fn = METRIC_REGISTRY[metric_name]
            val = fn(preds, target_set, k)
            all_metrics_acc[metric_name] += val
            row_result[metric_name] = val

        result.per_order.append(row_result)

    for metric_name in metrics:
        result.metrics[metric_name] = round(all_metrics_acc[metric_name] / len(grouped), 6)

    all_targets = sum(len(r["target"]) for r in result.per_order)
    result.total_products_in_orders = all_targets
    result.avg_order_size = round(all_targets / len(grouped), 2) if len(grouped) > 0 else 0.0

    return result


def print_eval_report(result: EvalResult) -> None:
    sep = "=" * 60
    print(sep)
    print(f"  Evaluation Report: {result.model_name}")
    print(sep)
    print(f"  Config: {result.config}")
    print(f"  Orders evaluated: {result.num_orders}")
    print(f"  Total target products: {result.total_products_in_orders}")
    print(f"  Avg order size: {result.avg_order_size}")
    print()
    for metric, value in result.metrics.items():
        print(f"  {metric}: {value}")
    print(sep)
