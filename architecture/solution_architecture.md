# Solution architecture

## Overall Design

In roder to build a recommendation system into the city's system, we need to prepare the infrastructure and for data ingestion, feature engineering, training, serving, monitoring, governance, and deployment.

I discussed how I designed for the following components in this demo. This demo focuses on the training lifecycle, shared infrastructure, shared storage, model parameter storage and configurations, reproducibility and production readiness through a lifecycle of two simple models.


### Modeling demo:
This repo will use open source e commerce data from Instacart. We will outlined the full data ingestion plan below.

For the demo, we will use [Instacart open source grocery](https://www.kaggle.com/c/instacart-market-basket-analysis) dataset.

**Problem:** Given an order with a subset of known products, predict the remaining products in the order.
- given an order with multiple products, predict product given a subset of products in the order. For example, if I ordered shampoo and conditioner, I will mask shampoo, and train the model to predict conditioner.

### Data Ingestion - Enrichment

In order to collect useful data for the city's website usage, we need to collect user interactions with the City's system. In online advertising, this is commonly referred to as first party (1P) data. For example, the system wants to track users who browsed and clicked on children's swimming classes, so the model can be trained to recommend more children's sports.

However, the city works with multiple divisions, and if there is no shared infrastructure, it essentially becomes a third party data ingestion (3P). This can be implemented after 1P, and generally, needs collaboration with the other divisions to enable data sharing across different platforms. This will be done in a way that is respectful of user privacy.

Other than click and impression events described above, we may also extend data ingestion into omni-channel. In e-commerce, this often refers to in store activities. Continuing with the recreation example above, there are many classes that requires signing up by phone calls. We can enrich the events data by combining offline activies, such as phone calls.

Historical activities, such as the class participant's list from previous years, can be used to solve the cold start problem in recommendataion systems. 

### Feature engineering

A primitive way to store and encode features is one hot encoding. It indicates a binary outcome, which is compatible with tree based models. Tree based models are lightweight and performant, and still remains a solid choice.

More advanced machine learning requires a more continuous encoding approach. This often involves transforming discrete values into a continuous, multi-dimensional vector. This technique originated in Natural Language Processing (NLP). It has since been deployed as a generic strategy to model discrete value using a probablistic interpretation. The vectors are trained using measures of similarity, e.g. swimming classes (both the activity and the language semantics) are closer to "volleyball classes" then "dispute your parking ticket."

Traditional and lightweight feature generation can be stored in a generic SQL or NoSQL database.

Vectors are generally higher in dimenaionality and more abstract. Newer databases (such as [pinecone](https://www.pinecone.io/learn/vector-database/)) uses similarity based indexing to ensure that "swimming classes and "volleyball classes" can be queries and returned together.

Features should be stored in a centralized database, and the paths to the storage buckets can be granted for individual model's using the model's configs.

### Model training

I propose a true north star metric to be something outside of the modeling world, such as a resident survey conducted for different cohorts. Their overall feedback is something we optimize for, instead of treating this as a purely mathematical problem.

As a proxy for their experience, we can define a baseline similarity measure for the recommendations. We can use a standard similarity or probablity measure for all the possible combinations. The baseline model could simply be a random generator, which picks any UI popup at random for a match with "swimming class". If there are 100 activities, and all activies are euqally probably to be recommended, the Shannon entropy is maximal, where H = 6.64.

Our models should be able to learn the difference between "pay your fine" and "volleyball classes", so it should have a strong preference over certain choices over others, so some activies should be highly likely to be recommended, and others are very unlikely to be recommended.

Using historical datasets, we should be able to divide it into training, validation, and testing.

Model architectures can be recorded using a combination of code, configurations, and open source libraries. Each variation should be archived and assigned a unique identifier. The hyperparameters can also be identified and retrieved using this approach.

### Recommendation serving

The model artifact can sit behing an api endpoint, and accessed 
```json
curl -X GET api/recommend \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
```

The response is generated using the session info like so

```json
{
    "id": 1,
    "data": {
        "title": "volleyball",
        "location": "north york community centre",
    }
}
```

### Monitoring and operations

The model performance should be evaluated using live sessions data and compared with the evaluation results on historical data to catch any significant drift.

A platform wide infrastructure should be enabled to toggle traffic between different models. Note that this experimentation and release infrastructure can be used by other teams that work on the city's website.

If there is a drift, we can toggle the traffic off, and investigate.

If there is an incident, we can immediately revert to the previous working model.

### Security and governance

The data source will have access control list.
Encryption applies for identity management and session management.
Training data should be scrubbed so there should be no PII presence.
we should have background jobs to delete any user data that has been requested for a deletion.

In terms of auditability, we can produce documents on what types of result will be rendered by the web or made available for any city employee. Sometimes, a tree based model is highly favoured by regulated industries, such as insurance.

### Deployment

Promotion from staging to production should go through a review process, similar to a pull request. Once approved, it should follow a gradual ramp up schedule, starting at a low percentage of traffic.

We can also whitelist some users if we can organize a beta test. This can be done by the employees, and also recrutied test participants.

## Solution overview

Baseline:
We will use a baseline dummy model that randomly select from all possible products.
Use scikitlearn, and store all model parameters in `configs/` using jsons.

Baseline results (full dataset, 49,677 products):
- recall@5: 0.000111
- precision@5: 0.000101

Tree based model:

We will use a simple tree based model that's trained locally for the same problem. Random forest is a classic machine learning model, and can be a precursor to the XGBoost and other deep learning models. Tree based models are computationally lightweight, and highly performant, and highly transparent in audits.

| File | Purpose |
| ---- | ------- |
| src/features.py	| Product stats aggregation + training feature generation + prediction feature builder. Stored locally as parquet at outputs/tree/features/train_features.parquet |
| src/tree_model.py |	TreeBaseline — RandomForest (100 trees, depth 10, 1.9M training rows) with fit/predict/save/load |
| configs/tree_config.json |	Model params + feature config |
| main.py |	Added train-tree and evaluate-tree subcommands
src/api.py	Both models served from /predict — switch with "model": "dummy" or "model": "tree"; tree accepts optional order_context |
| pyproject.toml | Added pyarrow (parquet support for archiving features) |
| README.md | Full documentation |

Results
| Model | Recall@5	| vs Random |
| ---- | ---------- | --------- |
| DummyBaseline | 0.0001 | 1× |
| TreeBaseline | 0.0369	| ~370× |

The tree learns mostly global product popularity (bananas, organic avocados, strawberries at the top), with some adjustment from order context features. See this [detailed report](https://github.com/violetguos/recsys-to/pull/3) for model features and result interpretation

### API usage - Tree model with order context
```
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"known_products": [1,2,3], "model": "tree", "order_context": {"order_dow": 2, "order_hour_of_day": 10}}'
```
Results
```
{"predictions":[{"product_id":24852,"product_name":"Banana","aisle":"fresh fruits","department":"produce"},{"product_id":13176,"product_name":"Bag of Organic Bananas","aisle":"fresh fruits","department":"produce"},{"product_id":47209,"product_name":"Organic Hass Avocado","aisle":"fresh fruits","department":"produce"},{"product_id":21137,"product_name":"Organic Strawberries","aisle":"fresh fruits","department":"produce"},{"product_id":47766,"product_name":"Organic Avocado","aisle":"fresh fruits","department":"produce"}],"model":"TreeBaseline","num_predictions":5,"catalog_size":49688}
```


### Data Quality Checks

| Check | Blocking | Description |
|-------|----------|-------------|
| Schema Validation | Yes | Expected columns match |
| Missing Identifiers | Yes | No nulls in key columns |
| Duplicate Records | No | Logs dupe count, non-blocking |
| Invalid Timestamps | Yes | `order_dow` [0-6], `order_hour_of_day` [0-23], `days_since_prior_order` ≥ 0 |
| Invalid Interaction Values | Yes | `add_to_cart_order` ≥ 1, `reordered` ∈ {0,1}, `order_number` ≥ 1 |
| Reference Integrity | Yes | FK checks across all tables |


### Scaling Notes

Training uses 100K prior orders (~3% of 3.2M) with 1:1 negative sampling → 1.9M training rows. The full dataset (32M prior rows) would produce ~64M training rows. Feature generation uses vectorized merges for positives and per-order negative sampling with `numpy.setdiff1d` + `set_index` lookups. For full-scale training, increase `max_orders_for_training` in `configs/tree_config.json` and tune `n_negatives_per_positive`.

For future versions of models involving more vector based features and deep learning models, we will use more horizontal scaling and sharding storages. See [operationa_scenarios](./operational_scenarios.md), [cost estimates](./operational_scenarios.md) and [decisions_log](./decisions_log.md)
