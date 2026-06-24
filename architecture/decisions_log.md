# Decisions log

| Scenario | alternative / north star | decision |
| ------- | ----------- |  -------- |
| feature engineering | Cloud storage | For this demo, I stored it locally, and I provided terraform files where I provision storage that outlines the plan for cloud storage |
| Model training | deep learning | I used random forest model with limited hyperparameter tuning and trained the model locally. It can be reproduced using Sagemaker with the parameters we have in [config](../configs/tree_config.json) |
| recommendation serving | deploy endpoint | for this demo, I used fastapi to create a local endpoint to access the modeling results. |
| data quality | jobs on top of data warehouse | for this demo, I included [quality check](../src/data_quality.py) locally. it can be transformed into a DAG, which runs on top of our analytics infrastructure |
| monitoring | cloud based solutions, e.g. datadog | for this demo, I used fastapi's console logging to monitor application health. When we have the volume, we should investigate cloud monitoring solutions, e.g. AWS or Datadog, and archive logs for auditing purposes |
| security, governance | implement a separate privacy infrastructure | In prod, we need to scrub PII from training data. The endpoint should also implement authentication, session, rate limiting |
| deployment | gradual release platform | this demo has a local endpoint. In prod, we should have infrastructure for a gradual ramp up of user traffic to any newly released model |
