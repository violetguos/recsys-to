from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from src.data_quality import run_quality_checks
from src.evaluation import evaluate_baseline, print_eval_report
from src.models import DummyBaseline

DATA_DIR = Path(__file__).parent / "data"
CONFIG_DIR = Path(__file__).parent / "configs"


def cmd_quality(args: argparse.Namespace) -> None:
    ok = run_quality_checks(sample_size=args.sample)
    sys.exit(0 if ok else 1)


def cmd_train(args: argparse.Namespace) -> None:
    config_path = CONFIG_DIR / f"{args.config}.json"
    with open(config_path) as f:
        config = json.load(f)

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

    np_path = out_dir / "catalog.npy"
    import numpy as np
    np.save(np_path, model.catalog_)

    cfg_path = out_dir / "config.json"
    with open(cfg_path, "w") as f:
        json.dump({"model": model.get_params(), "catalog_size": len(model.catalog_)}, f, indent=2)

    print(f"Trained {model.__class__.__name__}: catalog_size={len(model.catalog_)}")
    print(f"Saved to {out_dir}/")


def cmd_evaluate(args: argparse.Namespace) -> None:
    config_path = CONFIG_DIR / f"{args.config}.json"
    with open(config_path) as f:
        config = json.load(f)

    sample = args.sample or config.get("data", {}).get("sample_size")
    eval_cfg = config.get("evaluation", {})
    model_cfg = config.get("model", {})

    orders = pd.read_csv(
        DATA_DIR / "orders.csv",
        usecols=["order_id", "eval_set"],
        nrows=sample,
    )
    train_order_ids = orders.loc[orders["eval_set"] == "train", "order_id"]

    if len(train_order_ids) == 0:
        print("No train orders found in sample. Increase --sample or remove it for full scan.")
        sys.exit(1)

    chunk_iter = pd.read_csv(
        DATA_DIR / "order_products__train.csv",
        chunksize=500_000,
    )
    order_products = pd.concat(
        [chunk[chunk["order_id"].isin(train_order_ids)] for chunk in chunk_iter],
        ignore_index=True,
    )

    valid_params = {k: v for k, v in model_cfg.items() if k in ("random_state", "num_predictions")}
    model = DummyBaseline(**valid_params)
    prior = pd.read_csv(
        DATA_DIR / "order_products__prior.csv",
        usecols=["order_id", "product_id"],
        nrows=sample,
    )
    model.fit(prior)

    result = evaluate_baseline(
        model=model,
        order_products=order_products,
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
                "model_name": result.model_name,
                "config": result.config,
                "metrics": result.metrics,
                "num_orders": result.num_orders,
                "avg_order_size": result.avg_order_size,
            }, indent=2)
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Instacart - Data Quality & Baseline Model")
    sub = parser.add_subparsers(dest="command", required=True)

    dq = sub.add_parser("quality", help="Run data quality checks")
    dq.add_argument("--sample", type=int, default=None)
    dq.set_defaults(func=cmd_quality)

    train = sub.add_parser("train", help="Train baseline model on prior orders")
    train.add_argument("--config", default="baseline_config", help="Config name (without .json)")
    train.add_argument("--sample", type=int, default=None)
    train.add_argument("--out", default="outputs/baseline", help="Output directory")
    train.set_defaults(func=cmd_train)

    eval_ = sub.add_parser("evaluate", help="Evaluate baseline on train orders")
    eval_.add_argument("--config", default="baseline_config", help="Config name (without .json)")
    eval_.add_argument("--sample", type=int, default=None)
    eval_.add_argument("--out", default=None, help="Directory to save results")
    eval_.set_defaults(func=cmd_evaluate)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
