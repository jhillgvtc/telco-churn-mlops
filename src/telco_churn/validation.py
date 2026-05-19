from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from telco_churn.features import FEATURE_COLUMNS, TARGET_COLUMN


REQUIRED_COLUMNS = {
    "customer_id",
    "customer_type",
    "exchange",
    "subdivision",
    "tenure_months",
    "arpu",
    "rgu_count",
    "has_internet",
    "has_phone",
    "has_cable",
    "has_security",
    "has_autopay",
    "is_competitive",
    "is_mdu",
    "customers_at_address",
    "internet_only",
    "is_rental_proxy",
    "exchange_churn_rate",
    "churned",
}
LEAKAGE_COLUMNS = {"disconnect_date", "churn_reason", "save_offer_accepted", "post_churn_status"}
BINARY_COLUMNS = [
    "has_internet",
    "has_phone",
    "has_cable",
    "has_security",
    "has_autopay",
    "is_competitive",
    "is_mdu",
    "internet_only",
    "is_rental_proxy",
    "churned",
]


@dataclass
class ValidationResult:
    passed: bool
    errors: list[str]


def validate_dataframe(df: pd.DataFrame) -> ValidationResult:
    errors: list[str] = []
    missing = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing:
        errors.append(f"Missing required columns: {', '.join(missing)}")

    leakage = sorted(LEAKAGE_COLUMNS & set(df.columns))
    if leakage:
        errors.append(f"Leakage-prone columns present: {', '.join(leakage)}")

    if not df.empty and not missing:
        if df["customer_id"].duplicated().any():
            errors.append("customer_id must be unique")
        if (df["tenure_months"] <= 0).any():
            errors.append("tenure_months must be positive")
        if not df["arpu"].between(0, 500).all():
            errors.append("arpu must be between 0 and 500")
        if not df["rgu_count"].between(1, 5).all():
            errors.append("rgu_count must be between 1 and 5")
        if (df["customers_at_address"] < 1).any():
            errors.append("customers_at_address must be at least 1")
        if not df["exchange_churn_rate"].between(0, 1).all():
            errors.append("exchange_churn_rate must be between 0 and 1")
        for column in BINARY_COLUMNS:
            if column in df and not set(df[column].dropna().unique()).issubset({0, 1}):
                errors.append(f"{column} must be binary 0/1")
        for column in FEATURE_COLUMNS + [TARGET_COLUMN]:
            if column in df and df[column].isna().any():
                errors.append(f"{column} contains null values")

    return ValidationResult(passed=not errors, errors=errors)


def raise_if_invalid(df: pd.DataFrame) -> None:
    result = validate_dataframe(df)
    if not result.passed:
        raise ValueError("; ".join(result.errors))

