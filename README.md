# recsys-to

Data quality checks and baseline model for the [Instacart Market Basket Analysis](https://www.kaggle.com/c/instacart-market-basket-analysis) dataset.

**Problem:** Given an order with a subset of known products, predict the remaining products in the order.

## Setup

```bash
uv sync
source .venv/bin/activate
```

## Commands

```bash
# Run data quality checks on the full dataset
uv run python main.py quality

# Quick quality check (sample 100K rows per table)
uv run python main.py quality --sample 100000

# Train the dummy baseline model
uv run python main.py train

# Evaluate on train orders
uv run python main.py evaluate

# Quick eval with 10K sample
uv run python main.py evaluate --sample 10000

# Start the API server (requires trained model in outputs/baseline/)
uv run uvicorn src.api:app --reload
```

## Baseline Model

`DummyBaseline` randomly samples products from the full catalog, ignoring all input features. This provides a lower-bound reference for future models.

**Training:** Extracts the set of all unique `product_id` values from `order_products__prior.csv`.

**Evaluation:** For each order in `order_products__train.csv`:
1. Sort products by `add_to_cart_order` (simulating a basket-filling sequence)
2. Split the first `known_ratio` fraction as known, the rest as target
3. Predict `num_predictions` products
4. Compute `recall_at_k` and `precision_at_k`

## Configuration

Model and evaluation parameters are stored in `configs/` as JSON files:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model.random_state` | 42 | RNG seed for reproducibility |
| `model.num_predictions` | 5 | Number of products to predict per order |
| `evaluation.known_ratio` | 0.5 | Fraction of products treated as "known" in each order |
| `evaluation.split_by` | `add_to_cart_order` | Sorting key for known/target split |
| `evaluation.metrics` | `[recall_at_k, precision_at_k]` | Metrics to compute |

## Data Quality Checks

| Check | Blocking | Description |
|-------|----------|-------------|
| Schema Validation | Yes | Expected columns match |
| Missing Identifiers | Yes | No nulls in key columns |
| Duplicate Records | No | Logs dupe count, non-blocking |
| Invalid Timestamps | Yes | `order_dow` [0-6], `order_hour_of_day` [0-23], `days_since_prior_order` ≥ 0 |
| Invalid Interaction Values | Yes | `add_to_cart_order` ≥ 1, `reordered` ∈ {0,1}, `order_number` ≥ 1 |
| Reference Integrity | Yes | FK checks across all tables |

## Model API

Example request after starting the local API
```
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"known_products": [1, 2, 3], "num_predictions": 3}'
```

Response
```
{"predictions":[{"product_id":41981,"product_name":"Triphala, Vegetarian Capsules","aisle":"digestion","department":"personal care"},{"product_id":11654,"product_name":"Blood Orange Meyer Lemon Ginger Ale","aisle":"soft drinks","department":"beverages"},{"product_id":13053,"product_name":"Whole  Cashews","aisle":"nuts seeds dried fruit","department":"snacks"}],"model":"DummyBaseline","num_predictions":3,"catalog_size":49677}
```

```
curl -X GET http://localhost:8000/health
{"status":"ok","catalog_size":49677,"model":{"random_state":42,"num_predictions":5}}
```



## Infra setup
| File |	Purpose |
| ---- | -------- |
| main.tf |	Provider, required version, backend placeholder |
| variables.tf |	Region, project name, instance type, etc. |
| s3.tf |	recsys-baseline-dev-data bucket (raw CSVs) — recsys-baseline-dev-artifacts bucket (model outputs) — versioning + encryption |
| iam.tf	| SageMaker execution role with S3 read/write, ECR pull, CloudWatch logs |
| sagemaker.tf	| ECR repo (training image), Notebook instance (dev exploration), SageMaker Model + EndpointConfig + Endpoint (serving), Model Package Group (registry), null_resource with CLI template for triggering training |
| outputs.tf	| Bucket ARNs, role ARN, ECR URL, endpoint name, model name, notebook name |

Data flow:
1. CSVs uploaded to s3://...-data/data/ via aws_s3_object
2. Training container image pushed to ECR (built separately)
3. User triggers aws sagemaker create-training-job (command template in null_resource)
4. Training reads from data bucket, writes catalog.npy + config.json → artifacts bucket
5. SageMaker Model + Endpoint serve inference from those artifacts
6. Model versions tracked via SageMaker Model Package Group
```
cd infra/terraform
terraform plan   # preview (no apply = no AWS spend)
```

## Engineering decision log

| Decision | alternatives | selected option | justification | tradeoff |
| -------  | ------------| --------------- | --------------- | --------|
| build a baseline model | build a full machine learning model | baseline model | Amazon setup cost | included a terraform file to define data access and job |
