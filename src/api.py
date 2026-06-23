from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .tree_model import TreeBaseline

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "outputs" / "baseline"
TREE_DIR = ROOT / "outputs" / "tree"
DATA_DIR = ROOT / "data"


class PredictRequest(BaseModel):
    known_products: list[int] = []
    num_predictions: int | None = None
    model: str = "dummy"
    order_context: dict[str, Any] | None = None


class ProductInfo(BaseModel):
    product_id: int
    product_name: str
    aisle: str
    department: str


class PredictResponse(BaseModel):
    predictions: list[ProductInfo]
    model: str
    num_predictions: int
    catalog_size: int


def _load_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / f"{name}.csv")


def build_app(model_dir: Path = MODEL_DIR, tree_dir: Path = TREE_DIR,
              data_dir: Path = DATA_DIR) -> FastAPI:
    if not model_dir.exists():
        raise FileNotFoundError(f"Dummy model directory not found: {model_dir}")

    catalog = np.load(model_dir / "catalog.npy")
    config = json.loads((model_dir / "config.json").read_text())
    default_k = config.get("model", {}).get("num_predictions", 5)

    tree_model: TreeBaseline | None = None
    if tree_dir.exists() and (tree_dir / "model.joblib").exists():
        tree_model = TreeBaseline().load(tree_dir)
        print(f"Tree model loaded (catalog_size={len(tree_model.catalog_)})")

    products = _load_csv("products")
    aisles = _load_csv("aisles")
    departments = _load_csv("departments")
    products = products.merge(aisles, on="aisle_id", how="left")
    products = products.merge(departments, on="department_id", how="left")
    product_lookup = products.set_index("product_id")

    app = FastAPI(title="RecSys-Baseline", version="0.1.0")

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "catalog_size": len(catalog),
            "tree_model_loaded": tree_model is not None,
            "model": config.get("model", {}),
        }

    @app.get("/products/{product_id}")
    def get_product(product_id: int) -> ProductInfo:
        try:
            row = product_lookup.loc[product_id]
        except KeyError:
            raise HTTPException(404, f"Product {product_id} not found")
        return ProductInfo(
            product_id=int(row.name),
            product_name=str(row["product_name"]),
            aisle=str(row["aisle"]),
            department=str(row["department"]),
        )

    @app.post("/predict", response_model=PredictResponse)
    def predict(req: PredictRequest) -> PredictResponse:
        k = req.num_predictions or default_k
        model_name = req.model

        if model_name == "tree":
            if tree_model is None:
                raise HTTPException(400, "Tree model not loaded. Train it first with `python main.py train-tree`")
            chosen = tree_model.predict(
                known_products=np.array(req.known_products, dtype=int),
                order_context=req.order_context,
            )
            cat_size = len(tree_model.catalog_)
        else:
            rng = np.random.default_rng()
            exclude = np.array(req.known_products, dtype=int)
            candidates = np.setdiff1d(catalog, exclude)
            if len(candidates) == 0:
                raise HTTPException(400, "No candidates available after excluding known products")
            n = min(k, len(candidates))
            chosen = rng.choice(candidates, size=n, replace=False)
            cat_size = len(catalog)

        predictions = []
        for pid in chosen:
            try:
                row = product_lookup.loc[pid]
                predictions.append(ProductInfo(
                    product_id=int(row.name),
                    product_name=str(row["product_name"]),
                    aisle=str(row["aisle"]),
                    department=str(row["department"]),
                ))
            except KeyError:
                predictions.append(ProductInfo(
                    product_id=int(pid),
                    product_name="unknown", aisle="unknown", department="unknown",
                ))

        return PredictResponse(
            predictions=predictions,
            model="TreeBaseline" if model_name == "tree" else "DummyBaseline",
            num_predictions=len(predictions),
            catalog_size=cat_size,
        )

    return app


app = build_app()
