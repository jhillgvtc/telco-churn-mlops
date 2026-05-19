# Case Study: Productionizing a Broadband Churn Model

## Problem

Broadband churn work often stalls after the first model. A CSV of risk scores is useful once, but it does not become an operating system for retention unless the model can be refreshed, monitored, served, and explained.

This project shows the production path around a churn model: synthetic customer data flows through validation, feature engineering, training, scoring, monitoring, and serving.

## Pipeline

```text
synthetic telco data
-> validation and leakage checks
-> feature engineering
-> LightGBM training
-> local experiment logging
-> batch scoring
-> risk tier and action routing
-> FastAPI serving
-> Streamlit dashboard
-> drift monitoring
```

## Business Logic

The score answers who is likely to churn. The routing logic answers what to do next.

- Stable-address, medium-to-high risk customers route to retention outreach.
- Rental proxy and MDU customers route to winback or property-aware outreach.
- Low-risk customers stay in monitor status.

This distinction matters because not all churn is the same. Some customers are save candidates; others are address lifecycle opportunities.

## MLOps Lessons

- Data validation is part of the product, not a pre-flight nicety.
- Leakage checks should be explicit because churn datasets often contain post-outcome fields.
- Model quality should be measured with lift and capture, not AUC alone.
- A batch score without monitoring is a decaying asset.
- Portfolio projects are stronger when they show the operating system around the model.

## Public-Safety Design

The repo uses synthetic data only. The schema and behavior are realistic enough to demonstrate the engineering workflow, but no private customer data, addresses, exports, or proprietary records are included.

