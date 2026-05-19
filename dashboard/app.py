from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
SCORED = ROOT / "artifacts" / "scored_customers.csv"
METRICS = ROOT / "artifacts" / "metrics.json"

st.set_page_config(page_title="Telco Churn MLOps", layout="wide")
st.title("Telco Churn MLOps")

if not SCORED.exists():
    st.warning("Run `python -m telco_churn.cli score` before opening the dashboard.")
    st.stop()

df = pd.read_csv(SCORED)
metrics = json.loads(METRICS.read_text(encoding="utf-8")) if METRICS.exists() else {}

col1, col2, col3, col4 = st.columns(4)
col1.metric("Customers scored", f"{len(df):,}")
col2.metric("Avg risk", f"{df['churn_risk_score'].mean() * 100:.1f}%")
col3.metric("High+ risk", f"{df[df['risk_tier'].isin(['High', 'Very High'])].shape[0]:,}")
col4.metric("AUC", f"{metrics.get('auc_roc', 0):.3f}")

left, right = st.columns([1, 2])
with left:
    exchange = st.multiselect("Exchange", sorted(df["exchange"].unique()))
    tier = st.multiselect("Risk tier", ["Very Low", "Low", "Medium", "High", "Very High"])
    address_type = st.radio("Address type", ["All", "Stable", "Rental/MDU"])

filtered = df.copy()
if exchange:
    filtered = filtered[filtered["exchange"].isin(exchange)]
if tier:
    filtered = filtered[filtered["risk_tier"].isin(tier)]
if address_type == "Stable":
    filtered = filtered[(filtered["is_rental_proxy"] == 0) & (filtered["is_mdu"] == 0)]
elif address_type == "Rental/MDU":
    filtered = filtered[(filtered["is_rental_proxy"] == 1) | (filtered["is_mdu"] == 1)]

with right:
    st.subheader("Risk Tier Distribution")
    st.bar_chart(filtered["risk_tier"].value_counts())

st.subheader("Business Routing")
routing = (
    filtered.groupby(["risk_tier", "recommended_action"])
    .agg(customers=("customer_id", "size"), median_tenure_months=("tenure_months", "median"))
    .reset_index()
)
st.dataframe(
    routing,
    use_container_width=True,
    column_config={
        "risk_tier": "Risk tier",
        "recommended_action": "Recommended action",
        "customers": "Customers",
        "median_tenure_months": st.column_config.NumberColumn("Median tenure months", format="%.1f"),
    },
)

st.subheader("Top Scored Customers")
columns = [
    "customer_id",
    "exchange",
    "subdivision",
    "arpu",
    "tenure_months",
    "churn_risk_pct",
    "risk_tier",
    "top_risk_driver",
    "recommended_action",
]
st.dataframe(
    filtered.sort_values("churn_risk_score", ascending=False)[columns].head(100),
    use_container_width=True,
    column_config={
        "customer_id": "Customer ID",
        "exchange": "Exchange",
        "subdivision": "Subdivision",
        "arpu": st.column_config.NumberColumn("ARPU", format="$%.2f"),
        "tenure_months": st.column_config.NumberColumn("Customer tenure months", format="%.1f"),
        "churn_risk_pct": st.column_config.NumberColumn("Churn risk %", format="%.1f"),
        "risk_tier": "Risk tier",
        "top_risk_driver": "Top risk driver",
        "recommended_action": "Recommended action",
    },
)
