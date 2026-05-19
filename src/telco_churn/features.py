from __future__ import annotations

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "tenure_months",
    "arpu",
    "rgu_count",
    "internet_only",
    "has_phone",
    "has_cable",
    "has_security",
    "has_autopay",
    "is_competitive",
    "is_mdu",
    "customers_at_address",
    "is_rental_proxy",
    "exchange_churn_rate",
]

IDENTIFIER_COLUMNS = ["customer_id", "exchange", "subdivision"]
TARGET_COLUMN = "churned"


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    output = df.copy()
    output["internet_only"] = (
        (output["rgu_count"].astype(int) == 1) & (output["has_internet"].astype(int) == 1)
    ).astype(int)
    output["is_rental_proxy"] = (output["customers_at_address"].astype(int) >= 3).astype(int)
    output["exchange_churn_rate"] = output.groupby("exchange")["churned"].transform("mean")
    output["exchange_churn_rate"] = output["exchange_churn_rate"].fillna(output["churned"].mean())
    return output


def model_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    data = add_engineered_features(df)
    return data[FEATURE_COLUMNS], data[TARGET_COLUMN].astype(int)


def score_tier(score: float) -> str:
    if score < 0.10:
        return "Very Low"
    if score < 0.25:
        return "Low"
    if score < 0.50:
        return "Medium"
    if score < 0.75:
        return "High"
    return "Very High"


def recommended_action(row: pd.Series) -> str:
    tier = row["risk_tier"]
    rental = int(row["is_rental_proxy"]) == 1 or int(row["is_mdu"]) == 1
    if tier in {"Very Low", "Low"}:
        return "Monitor"
    if rental:
        return "Winback trigger / property-aware outreach"
    if tier == "Medium":
        return "Proactive retention outreach"
    return "Priority retention offer"


def top_risk_driver(row: pd.Series) -> str:
    candidates = {
        "short_tenure": max(0.0, (24.0 - float(row["tenure_months"])) / 24.0),
        "low_arpu": max(0.0, (95.0 - float(row["arpu"])) / 95.0),
        "address_turnover": min(float(row["customers_at_address"]) / 8.0, 1.0),
        "competitive_area": float(row["is_competitive"]),
        "manual_pay": 1.0 - float(row["has_autopay"]),
        "internet_only": float(row["internet_only"]),
    }
    return max(candidates, key=candidates.get)


def population_stability_index(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
    quantiles = np.linspace(0, 1, buckets + 1)
    breakpoints = np.unique(np.quantile(expected, quantiles))
    if len(breakpoints) <= 2:
        return 0.0
    expected_counts, _ = np.histogram(expected, bins=breakpoints)
    actual_counts, _ = np.histogram(actual, bins=breakpoints)
    expected_pct = np.maximum(expected_counts / max(expected_counts.sum(), 1), 0.0001)
    actual_pct = np.maximum(actual_counts / max(actual_counts.sum(), 1), 0.0001)
    return float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))

