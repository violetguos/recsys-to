from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_quality import run_quality_checks
from src.evaluation import evaluate_baseline, print_eval_report
from src.models import DummyBaseline
from src.tree_model import TreeBaseline

DATA_DIR = Path(__file__).parent / "data"
CONFIG_DIR = Path(__file__).parent / "configs"


def cmd_quality(args: argparse.Namespace) -> None:
    ok = run_quality_checks(sample_size=args.sample)
    sys.exit(0 if ok else 1)


def _load_model_cfg(config_name: str) -> dict:
    config_path = CONFIG_DIR / f"{config_name}.json"
    with open(config_path) as f:
        return json.load(f)


def cmd_train(args: argparse.Namespace) -> None:
    config = _load_model_cfg(args.config)
    sample = args.sample or config.get("data", {}).get("sample_size")

    order_products = pd.read_csv(
        DATA_DIR / "order_products__prior.csv",
        usecols=["order_id", "product_id"],
        nrows=sample,
    )

    model_cfg = config.get("model", {})
    valid_params = {k: v for k, v in model_cfg.items() if k in ("random_state", "num_predictions")}
    model = DummyBaseline(**valid_params)
    model.fit(order_products)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "catalog.npy", model.catalog_)
    with open(out_dir / "config.json", "w") as f:
        json.dump({"model": model.get_params(), "catalog_size": len(model.catalog_)}, f, indent=2)

    print(f"Trained {model.__class__.__name__}: catalog_size={len(model.catalog_)}")
    print(f"Saved to {out_dir}/")


def cmd_evaluate(args: argparse.Namespace) -> None:
    config = _load_model_cfg(args.config)
    sample = args.sample or config.get("data", {}).get("sample_size")
    eval_cfg = config.get("evaluation", {})
    model_cfg = config.get("model", {})

    orders = pd.read_csv(
        DATA_DIR / "orders.csv", usecols=["order_id", "eval_set"], nrows=sample,
    )
    train_order_ids = orders.loc[orders["eval_set"] == "train", "order_id"]
    if len(train_order_ids) == 0:
        print("No train orders found in sample. Increase --sample or remove it for full scan.")
        sys.exit(1)

    chunk_iter = pd.read_csv(DATA_DIR / "order_products__train.csv", chunksize=500_000)
    order_products = pd.concat(
        [chunk[chunk["order_id"].isin(train_order_ids)] for chunk in chunk_iter],
        ignore_index=True,
    )

    valid_params = {k: v for k, v in model_cfg.items() if k in ("random_state", "num_predictions")}
    model = DummyBaseline(**valid_params)
    prior = pd.read_csv(
        DATA_DIR / "order_products__prior.csv", usecols=["order_id", "product_id"], nrows=sample,
    )
    model.fit(prior)

    result = evaluate_baseline(
        model=model, order_products=order_products,
        known_ratio=eval_cfg.get("known_ratio", 0.5),
        split_by=eval_cfg.get("split_by", "add_to_cart_order"),
        num_predictions=eval_cfg.get("num_predictions", model_cfg.get("num_predictions", 5)),
        metrics=eval_cfg.get("metrics", ["recall_at_k", "precision_at_k"]),
    )
    print_eval_report(result)

    if args.out:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "eval_result.json").write_text(
            json.dumps({
                "model_name": result.model_name, "config": result.config,
                "metrics": result.metrics, "num_orders": result.num_orders,
                "avg_order_size": result.avg_order_size,
            }, indent=2)
        )


def cmd_train_tree(args: argparse.Namespace) -> None:
    config = _load_model_cfg(args.config)
    model_cfg = config.get("model", {})
    valid_keys = {"random_state", "num_predictions", "max_depth", "n_estimators", "min_samples_leaf"}
    params = {k: v for k, v in model_cfg.items() if k in valid_keys}

    model = TreeBaseline(**params)
    model.fit()

    out_dir = Path(args.out)
    model.save(out_dir)


def cmd_evaluate_tree(args: argparse.Namespace) -> None:
    config = _load_model_cfg(args.config)
    sample = args.sample or config.get("data", {}).get("sample_size")
    eval_cfg = config.get("evaluation", {},)
    num_preds = eval_cfg.get("num_predictions",
                             config.get("model", {}).get("num_predictions", 5))

    model_dir = Path(args.model_dir)
    model = TreeBaseline().load(model_dir)

    orders = pd.read_csv(
        DATA_DIR / "orders.csv", usecols=["order_id", "eval_set"], nrows=sample,
    )
    train_order_ids = orders.loc[orders["eval_set"] == "train", "order_id"]
    if len(train_order_ids) == 0:
        print("No train orders found in sample.")
        sys.exit(1)

    chunk_iter = pd.read_csv(DATA_DIR / "order_products__train.csv", chunksize=500_000)
    order_products = pd.concat(
        [chunk[chunk["order_id"].isin(train_order_ids)] for chunk in chunk_iter],
        ignore_index=True,
    )
    order_info = pd.read_csv(
        DATA_DIR / "orders.csv",
        usecols=["order_id", "order_dow", "order_hour_of_day",
                 "days_since_prior_order", "order_number"],
        nrows=sample,
    )

    grouped = order_products.groupby("order_id")["product_id"].apply(list).reset_index()
    grouped.columns = ["order_id", "products"]
    grouped = grouped.merge(order_info, on="order_id", how="left")

    hits = 0
    total_target = 0
    k = num_preds

    for _, row in grouped.iterrows():
        all_prods = np.array(row["products"])
        split = max(1, len(all_prods) // 2)
        known = all_prods[:split]
        target = set(all_prods[split:])

        ctx = {
            "order_dow": int(row["order_dow"]),
            "order_hour_of_day": int(row["order_hour_of_day"]),
            "days_since_prior_order": (float(row["days_since_prior_order"])
                                       if pd.notna(row["days_since_prior_order"]) else -1),
            "order_number": int(row["order_number"]),
        }

        preds = set(model.predict(known_products=known, order_context=ctx))
        n_target = min(k, len(target))
        if n_target > 0:
            hits += len(preds & target) / n_target
        total_target += 1

    avg_recall = hits / total_target if total_target > 0 else 0
    print(f"\nEvaluated {total_target} orders")
    print(f"recall_at_{k}: {avg_recall:.6f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Instacart - Data Quality, Dummy & Tree Models")
    sub = parser.add_subparsers(dest="command", required=True)

    dq = sub.add_parser("quality", help="Run data quality checks")
    dq.add_argument("--sample", type=int, default=None)
    dq.set_defaults(func=cmd_quality)

    train = sub.add_parser("train", help="Train dummy baseline")
    train.add_argument("--config", default="baseline_config")
    train.add_argument("--sample", type=int, default=None)
    train.add_argument("--out", default="outputs/baseline")
    train.set_defaults(func=cmd_train)

    eval_ = sub.add_parser("evaluate", help="Evaluate dummy baseline")
    eval_.add_argument("--config", default="baseline_config")
    eval_.add_argument("--sample", type=int, default=None)
    eval_.add_argument("--out", default=None)
    eval_.set_defaults(func=cmd_evaluate)

    tt = sub.add_parser("train-tree", help="Train tree model (generates features, trains RandomForest)")
    tt.add_argument("--config", default="tree_config")
    tt.add_argument("--out", default="outputs/tree")
    tt.set_defaults(func=cmd_train_tree)

    et = sub.add_parser("evaluate-tree", help="Evaluate tree model on train orders")
    et.add_argument("--config", default="tree_config")
    et.add_argument("--sample", type=int, default=None)
    et.add_argument("--model-dir", default="outputs/tree")
    et.set_defaults(func=cmd_evaluate_tree)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
