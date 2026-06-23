## Problem statement

In roder to build a recommendation system into the city's system, we need to prepare the infrastructure and for data ingestion, feature engineering, training, serving, monitoring, governance, and deployment.

### Data Ingestion

In order to collect useful data for the city's website usage, we need to collect user interactions with the City's system. In online advertising, this is commonly referred to as first party (1P) data. For example, the system wants to track users who browsed and clicked on children's swimming classes, so the model can be trained to recommend more children's sports.

However, the city works with multiple divisions, and if there is no shared infrastructure, it essentially becomes a third party data ingestion (3P). This can be implemented after 1P, and generally, needs collaboration with the other divisions to enable data sharing across different platforms.

Other than click and impression events described above, we may also extend data ingestion into omni-channel. In e-commerce, this often refers to in store activities. Continuing with the recreation example above, there are many classes that requires signing up by phone calls. We can enrich the events data by combining offline activies, such as phone calls.

Historical activities, such as the class participant's list, can be used to solve the cold start problem in recommendataion systems. 

### Feature engineering

A primitive way to store and encode features is one hot encoding. It indicates a binary outcome, which is compatible with tree based models. Tree based models are lightweight and performant, and still remains a solid choice.

More advanced machine learning requires a more continuous encoding approach. This often involves transforming discrete values into a continuous, multi-dimensional vector. This technique originated in Natural Language Processing (NLP). It has since been deployed as a generic strategy to model discrete value using a probablistic interpretation. The vectors are trained using measures of similarity, e.g. swimming classes (both the activity and the language semantics) are closer to "volleyball classes" then "dispute your parking ticket."

Traditional and lightweight feature generation can be stored in a generic SQL or NoSQL database.

Vectors are generally higher in dimenaionality and more abstract. Newer databases (such as [pinecone](https://www.pinecone.io/learn/vector-database/)) uses similarity based indexing to ensure that "swimming classes and "volleyball classes" can be queries and returned together.

Features should be stored in a centralized database, and the paths to the storage buckets can be granted for individual model's using the model's configs.

### Model training

I propose a true north star metric to be something outside of the modeling world, such as a resident survey conducted throughout different cohorts. Their overall feedback is something we optimize for, instead of treating this as a purely mathematical problem.

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

A platform wide infrastructure should be enabled to toggle traffic between different models. Note that this infrastructure can be used by other teams that work with the city in any capcity.

If there is a drift, we can toggle the traffic to 100% go to the better model's endpoint, and investigate.

If there is an incident, we should immediately revert to the previous working model.

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

### Data

This repo will use open source e commerce data from Instacart. We will omit the full data ingestion since that requires collecting clicks and impressions across the city's websites.

See [README.md](README.md) section Data Quality Checks

### Modeling

Problem statement:
- I retrieved the Instacart opensource grocery dataset
- given an order with multiple products, predict product given a subset of products in the order. For example, if I ordered shampoo and conditioner, I will mask shampoo, and train the model to predict conditioner.


Baseline:
We will use a baseline dummy model that randomly select from all possible products.
Use scikitlearn, and store all model parameters in configs/ using jsons.

Baseline results (full dataset, 49,677 products):
- recall@5: 0.000111
- precision@5: 0.000101

Tree based model:

We will use a simple tree based model that's trained locally for the same problem. Random forest is a classic machine learning model, and can be a precursor to the XGBoost and other deep learning models. Tree based models are computationally lightweight, and highly performant, and highly transparent in audits.

New files
| File | Purpose |
| ---- | ------- |
| src/features.py	| Product stats aggregation + training feature generation + prediction feature builder. Stored locally as parquet at outputs/tree/features/train_features.parquet |
| src/tree_model.py |	TreeBaseline — RandomForest (100 trees, depth 10, 1.9M training rows) with fit/predict/save/load |
| configs/tree_config.json |	Model params + feature config |

Updated files
| File |	Change |
| ---- | ------- |
| main.py |	Added train-tree and evaluate-tree subcommands
src/api.py	Both models served from /predict — switch with "model": "dummy" or "model": "tree"; tree accepts optional order_context |
| pyproject.toml | Added pyarrow (parquet support) |
| README.md | Full documentation |

Results
| Model | Recall@5	| vs Random |
| ---- | ---------- | --------- |
| DummyBaseline | 0.0001 | 1× |
| TreeBaseline | 0.0369	| ~370× |

The tree learns mostly global product popularity (bananas, organic avocados, strawberries at the top), with some adjustment from order context features.
API usage
### Tree model with order context
```
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"known_products": [1,2,3], "model": "tree", "order_context": {"order_dow": 2, "order_hour_of_day": 10}}'
```
Results
```
{"predictions":[{"product_id":24852,"product_name":"Banana","aisle":"fresh fruits","department":"produce"},{"product_id":13176,"product_name":"Bag of Organic Bananas","aisle":"fresh fruits","department":"produce"},{"product_id":47209,"product_name":"Organic Hass Avocado","aisle":"fresh fruits","department":"produce"},{"product_id":21137,"product_name":"Organic Strawberries","aisle":"fresh fruits","department":"produce"},{"product_id":47766,"product_name":"Organic Avocado","aisle":"fresh fruits","department":"produce"}],"model":"TreeBaseline","num_predictions":5,"catalog_size":49688}
```