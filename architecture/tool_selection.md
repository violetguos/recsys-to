# Tool selection

This repo will focus on the infrastructure for data ingestion, mdel registry, and serving.

This means assuming that we have data to work with, and not building the full data ingestion since that requires collecting clicks and impressions across the city's websites.

- 01_load_data - loader using parquest and numpy
- 02_validate_data - `data_quality.py`
- 03_generate_features - shared feature processing in `features.py`
- 04_train_model - implemented in model classes
- 05_evaluate_model - implemented in `evaluation.py`
- 06_generate_recommendations - implemented in fastapi
- 07_publish_results - implemented in fastapi

I included terraform files to provision AWS resources, and models are trained using numpy and scikitlearn for demostrative purposes. I chose not to implement deep learning frameworks due to the compute that they require. This project sets up a good practice to evolve into deep learning.

Cost estimate for a random forest model

| Resource |	Full-catalog training (once) |
| ------- | ------------------ |
| SageMaker Training (ml.c5.4xlarge, 2 hrs) |	~$1.40 |
| SageMaker Training (ml.m5.4xlarge, 1 hr)	| ~$0.90 |
| SageMaker Endpoint (ml.m5.large, 24/7) |	~$90/mo |
| Feature storage (S3, ~500 MB parquet) |	~$0.01/mo |
