from __future__ import annotations

from typing import Any

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from telco_churn.config import load_config, project_path
from telco_churn.scoring import load_model, score_dataframe


app = FastAPI(title="Telco Churn MLOps API", version="0.1.0")


class CustomerPayload(BaseModel):
    customer_id: int = Field(..., examples=[100001])
    customer_type: str = "Residential"
    exchange: str = "North Ridge"
    subdivision: str = "Maple Grove"
    tenure_months: float = Field(..., gt=0)
    arpu: float = Field(..., ge=0, le=500)
    rgu_count: int = Field(..., ge=1, le=5)
    has_internet: int = Field(1, ge=0, le=1)
    has_phone: int = Field(..., ge=0, le=1)
    has_cable: int = Field(..., ge=0, le=1)
    has_security: int = Field(..., ge=0, le=1)
    has_autopay: int = Field(..., ge=0, le=1)
    is_competitive: int = Field(..., ge=0, le=1)
    is_mdu: int = Field(..., ge=0, le=1)
    customers_at_address: int = Field(..., ge=1)
    churned: int = Field(0, ge=0, le=1)


@app.get("/health")
def health() -> dict[str, Any]:
    config = load_config()
    model_path = project_path(config["paths"]["model"])
    exists = model_path.exists()
    metrics = {}
    if exists:
        _, _, metrics = load_model(config)
    return {"status": "ok" if exists else "model_missing", "model_path": str(model_path), "metrics": metrics}


@app.post("/predict")
def predict(payload: CustomerPayload) -> dict[str, Any]:
    scored = score_dataframe(pd.DataFrame([payload.model_dump()]))
    row = scored.iloc[0]
    return {
        "customer_id": int(row["customer_id"]),
        "churn_risk_score": float(row["churn_risk_score"]),
        "risk_tier": row["risk_tier"],
        "top_risk_driver": row["top_risk_driver"],
        "recommended_action": row["recommended_action"],
    }


@app.post("/batch-predict")
def batch_predict(payloads: list[CustomerPayload]) -> list[dict[str, Any]]:
    scored = score_dataframe(pd.DataFrame([payload.model_dump() for payload in payloads]))
    return [
        {
            "customer_id": int(row["customer_id"]),
            "churn_risk_score": float(row["churn_risk_score"]),
            "risk_tier": row["risk_tier"],
            "top_risk_driver": row["top_risk_driver"],
            "recommended_action": row["recommended_action"],
        }
        for _, row in scored.iterrows()
    ]
