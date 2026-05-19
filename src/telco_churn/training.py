from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split

from telco_churn.config import ensure_parent, load_config, project_path
from telco_churn.features import FEATURE_COLUMNS, model_frame
from telco_churn.validation import raise_if_invalid


def top_decile_lift(y_true: np.ndarray, scores: np.ndarray) -> float:
    cutoff = np.quantile(scores, 0.9)
    base_rate = y_true.mean()
    top_rate = y_true[scores >= cutoff].mean()
    return float(top_rate / base_rate) if base_rate > 0 else 0.0


def top_quartile_capture(y_true: np.ndarray, scores: np.ndarray) -> float:
    cutoff = np.quantile(scores, 0.75)
    total_churners = y_true.sum()
    captured = y_true[scores >= cutoff].sum()
    return float(captured / total_churners) if total_churners > 0 else 0.0


def _log_mlflow_or_fallback(metrics: dict[str, float], params: dict, model_path: Path, fallback_dir: Path) -> None:
    try:
        import mlflow

        mlflow.set_tracking_uri("file:mlruns")
        mlflow.set_experiment("telco-churn-mlops")
        with mlflow.start_run():
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            mlflow.log_artifact(str(model_path))
    except Exception:
        fallback_dir.mkdir(parents=True, exist_ok=True)
        with (fallback_dir / "last_run.json").open("w", encoding="utf-8") as handle:
            json.dump({"params": params, "metrics": metrics, "model_path": str(model_path)}, handle, indent=2)


def train_model(config_path: str | None = None) -> dict[str, float]:
    config = load_config(config_path) if config_path else load_config()
    df = pd.read_csv(project_path(config["paths"]["processed"]))
    raise_if_invalid(df)
    X, y = model_frame(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config["model"]["test_size"],
        random_state=config["model"]["random_state"],
        stratify=y,
    )
    params = {
        "n_estimators": config["model"]["n_estimators"],
        "learning_rate": config["model"]["learning_rate"],
        "max_depth": config["model"]["max_depth"],
        "num_leaves": config["model"]["num_leaves"],
        "random_state": config["model"]["random_state"],
        "n_jobs": 1,
        "verbosity": -1,
    }
    model = LGBMClassifier(**params)
    model.fit(X_train, y_train)
    scores = model.predict_proba(X_test)[:, 1]
    y_np = y_test.to_numpy()
    metrics = {
        "auc_roc": float(roc_auc_score(y_np, scores)),
        "average_precision": float(average_precision_score(y_np, scores)),
        "brier_score": float(brier_score_loss(y_np, scores)),
        "top_decile_lift": top_decile_lift(y_np, scores),
        "top_quartile_capture": top_quartile_capture(y_np, scores),
        "test_rows": float(len(y_test)),
    }
    model_path = ensure_parent(config["paths"]["model"])
    joblib.dump({"model": model, "features": FEATURE_COLUMNS, "metrics": metrics}, model_path)
    metrics_path = ensure_parent(config["paths"]["metrics"])
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    _log_mlflow_or_fallback(metrics, params, model_path, project_path(config["paths"]["mlflow_fallback"]))
    return metrics
