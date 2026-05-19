from __future__ import annotations

import numpy as np
import pandas as pd

from telco_churn.config import ensure_parent, load_config
from telco_churn.features import add_engineered_features


# Fictional market labels only. Do not use real service-area place names in this
# public-safe synthetic dataset.
EXCHANGES = ["North Ridge", "Pine Valley", "Lakeview", "Hillcrest", "Riverbend", "South Mesa"]
SUBDIVISIONS = [
    "Maple Grove",
    "Cedar Crossing",
    "Juniper Bend",
    "Oak Hollow",
    "Silver Creek",
    "Copper Ridge",
    "Willow Park",
    "Summit Flats",
]


def generate_customers(n_customers: int = 12000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    customer_id = np.arange(100000, 100000 + n_customers)
    exchange = rng.choice(EXCHANGES, size=n_customers, p=[0.17, 0.18, 0.2, 0.16, 0.18, 0.11])
    subdivision = rng.choice(SUBDIVISIONS, size=n_customers)
    is_mdu = rng.binomial(1, np.where(np.isin(subdivision, ["Summit Flats"]), 0.85, 0.12))
    is_competitive = rng.binomial(1, np.where(np.isin(subdivision, ["Maple Grove", "Silver Creek"]), 0.65, 0.22))
    tenure_months = np.clip(rng.gamma(shape=2.4, scale=38, size=n_customers), 1, 260).round(1)
    rgu_count = rng.choice([1, 2, 3, 4], size=n_customers, p=[0.54, 0.28, 0.14, 0.04])
    has_internet = np.ones(n_customers, dtype=int)
    has_phone = rng.binomial(1, np.clip(0.18 + 0.16 * (rgu_count >= 2), 0, 1))
    has_cable = rng.binomial(1, np.clip(0.12 + 0.23 * (rgu_count >= 2), 0, 1))
    has_security = rng.binomial(1, np.clip(0.08 + 0.16 * (rgu_count >= 2), 0, 1))
    has_autopay = rng.binomial(1, np.clip(0.48 + 0.18 * (tenure_months > 36), 0, 0.88))
    customers_at_address = np.where(is_mdu == 1, rng.poisson(3.2, n_customers) + 1, rng.poisson(0.8, n_customers) + 1)
    base_arpu = rng.normal(92, 22, n_customers) + (rgu_count - 1) * rng.normal(25, 8, n_customers)
    arpu = np.clip(base_arpu, 38, 235).round(2)

    risk_logit = (
        -2.6
        - 0.022 * tenure_months
        - 0.015 * (arpu - 90)
        - 0.46 * (rgu_count - 1)
        - 0.55 * has_autopay
        + 0.84 * is_competitive
        + 0.66 * is_mdu
        + 0.24 * np.clip(customers_at_address - 1, 0, 8)
        + rng.normal(0, 0.38, n_customers)
    )
    churn_probability = 1 / (1 + np.exp(-risk_logit))
    churned = rng.binomial(1, churn_probability)

    df = pd.DataFrame(
        {
            "customer_id": customer_id,
            "customer_type": "Residential",
            "exchange": exchange,
            "subdivision": subdivision,
            "tenure_months": tenure_months,
            "arpu": arpu,
            "rgu_count": rgu_count,
            "has_internet": has_internet,
            "has_phone": has_phone,
            "has_cable": has_cable,
            "has_security": has_security,
            "has_autopay": has_autopay,
            "is_competitive": is_competitive,
            "is_mdu": is_mdu,
            "customers_at_address": customers_at_address,
            "churned": churned,
        }
    )
    return add_engineered_features(df)


def write_synthetic_data(config_path: str | None = None) -> pd.DataFrame:
    config = load_config(config_path) if config_path else load_config()
    df = generate_customers(config["n_customers"], config["seed"])
    raw_path = ensure_parent(config["paths"]["raw"])
    processed_path = ensure_parent(config["paths"]["processed"])
    df.to_csv(raw_path, index=False)
    df.to_csv(processed_path, index=False)
    return df
