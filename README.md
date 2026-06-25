# recsys-to

In roder to build a recommendation system into the city's system, we need to prepare the infrastructure and for data ingestion, feature engineering, training, serving, monitoring, governance, and deployment.

I discussed how I designed for the following components in this demo. This demo focuses on the training lifecycle, shared infrastructure, shared storage, model parameter storage and configurations, reproducibility and production readiness through a lifecycle of two simple models.


## Setup

Detailed reasons and why certain tools were selected, see [tool_selection.md](./architecture/tool_selection.md)

```bash
# install packages
uv sync
source .venv/bin/activate
```

## Docker

```bash
# Build
docker build -t recsys-to .

# Run (mount local outputs/data for live artifacts)
docker run -p 8000:8000 \
  -v "$(pwd)/outputs:/app/outputs" \
  -v "$(pwd)/data:/app/data" \
  recsys-to

# Or with compose (includes hot-reload)
docker compose up
```

## Commands

```bash
# Data quality (24 checks)
uv run python main.py quality

# Train dummy baseline (random)
uv run python main.py train

# Evaluate dummy
uv run python main.py evaluate

# Train tree model (RandomForest — generates features, trains on 100K orders)
uv run python main.py train-tree

# Evaluate tree
uv run python main.py evaluate-tree --sample 20000

# API server (loads both models if available)
uv run uvicorn src.api:app --reload
```

## Overall results - Model Comparison

| Model | Recall@5 | Description |
|-------|----------|-------------|
| DummyBaseline (random) | ~0.0001 | Uniform random sampling from 50K products |
| TreeBaseline (RandomForest) | **0.0369** | Learns product popularity + order context features |

## Tree Model — Features

Features computed per (order, product) pair:

| Feature | Source | Description |
|---------|--------|-------------|
| `order_dow` | orders | Day of week (0–6) |
| `order_hour_of_day` | orders | Hour (0–23) |
| `days_since_prior_order` | orders | Days since last order (NaN → -1) |
| `order_number` | orders | Order sequence for this user |
| `popularity` | aggregated | How many prior orders contain this product |
| `reorder_rate` | aggregated | Fraction of orders where product was reordered |
| `avg_cart_pos` | aggregated | Average add-to-cart position |
| `aisle_id` | products | Product aisle |
| `department_id` | products | Product department |

Generated via `src/features.py` — product stats are pre-computed, then positive + negative sampled examples are merged. Training features are stored to `outputs/tree/features/train_features.parquet`.

## API - model deployment

```bash
# Test health
curl http://localhost:8000/health

# Predict with dummy model (default)
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"known_products": [1, 2, 3], "num_predictions": 3}'

# Predict with tree model
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"known_products": [1, 2, 3], "model": "tree", "order_context": {"order_dow": 2, "order_hour_of_day": 10}}'

# Product lookup
curl http://localhost:8000/products/1
```

Both models are served from the same `/predict` endpoint — switch with `"model": "dummy"` (default) or `"model": "tree"`.

## CI

Push/PR to `main` triggers GitHub Actions (`.github/workflows/ci.yml`):
1. **lint** — `ruff check src/ tests/`
2. **test** — `pytest` (model-dependent tests skip gracefully if artifacts absent)
3. **docker** — build & push `ghcr.io/<repo>` image, skip gracefully if artifacts absent

```bash
# Run CI tests locally
uv sync --group dev
uv run ruff check src/ tests/
uv run pytest tests/ -v
```

## Infra setup (AWS / Terraform)

| File | Purpose |
| ---- | ------- |
| `main.tf` | Provider, required version, backend placeholder |
| `variables.tf` | Region, project name, instance type, etc. |
| `s3.tf` | Data bucket (raw CSVs) + Artifacts bucket (model outputs) |
| `iam.tf` | SageMaker execution role with S3/ECR/CloudWatch access |
| `sagemaker.tf` | ECR repo, Notebook instance, SageMaker Model + Endpoint, Model Package Group |
| `outputs.tf` | Bucket ARNs, role ARN, ECR URL, endpoint name |

```bash
cd infra/terraform
terraform plan   # preview without spending
```

## Detailed design and architecture decisions
Please refer to [solution_architecture.md](./architecture/solution_architecture.md), [tools](./architecture/tool_selection.md), [decisions](./architecture/decisions_log.md), and [operational scenarios](./architecture/operational_scenarios.md)
