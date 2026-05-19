from __future__ import annotations

import joblib
import pandas as pd

from telco_churn.config import ensure_parent, load_config, project_path
from telco_churn.features import FEATURE_COLUMNS, add_engineered_features, recommended_action, score_tier, top_risk_driver
from telco_churn.validation import raise_if_invalid


def load_model(config: dict):
    artifact = joblib.load(project_path(config["paths"]["model"]))
    return artifact["model"], artifact["features"], artifact.get("metrics", {})


def score_dataframe(df: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    cfg = config or load_config()
    data = add_engineered_features(df)
    raise_if_invalid(data)
    model, features, _ = load_model(cfg)
    scores = model.predict_proba(data[features])[:, 1]
    output = data.copy()
    output["churn_risk_score"] = scores
    output["churn_risk_pct"] = (scores * 100).round(1)
    output["risk_tier"] = output["churn_risk_score"].apply(score_tier)
    output["top_risk_driver"] = output.apply(top_risk_driver, axis=1)
    output["recommended_action"] = output.apply(recommended_action, axis=1)
    return output


def batch_score(config_path: str | None = None) -> pd.DataFrame:
    config = load_config(config_path) if config_path else load_config()
    df = pd.read_csv(project_path(config["paths"]["processed"]))
    scored = score_dataframe(df, config)
    output_path = ensure_parent(config["paths"]["scored"])
    scored.to_csv(output_path, index=False)
    return scored

