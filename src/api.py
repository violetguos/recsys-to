from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "outputs" / "baseline"
DATA_DIR = ROOT / "data"


class PredictRequest(BaseModel):
    known_products: list[int] = []
    num_predictions: int | None = None


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


class PredictError(BaseModel):
    detail: str


def _load_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / f"{name}.csv")


def build_app(model_dir: Path = MODEL_DIR, data_dir: Path = DATA_DIR) -> FastAPI:
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    catalog = np.load(model_dir / "catalog.npy")
    config = json.loads((model_dir / "config.json").read_text())
    default_k = config.get("model", {}).get("num_predictions", 5)
    random_state = config.get("model", {}).get("random_state", 42)

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
        rng = np.random.default_rng()
        k = req.num_predictions or default_k

        exclude = np.array(req.known_products, dtype=int)
        candidates = np.setdiff1d(catalog, exclude)

        if len(candidates) == 0:
            raise HTTPException(400, "No candidates available after excluding known products")

        n = min(k, len(candidates))
        chosen = rng.choice(candidates, size=n, replace=False)

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
                    product_name="unknown",
                    aisle="unknown",
                    department="unknown",
                ))

        return PredictResponse(
            predictions=predictions,
            model="DummyBaseline",
            num_predictions=n,
            catalog_size=len(catalog),
        )

    return app


app = build_app()
