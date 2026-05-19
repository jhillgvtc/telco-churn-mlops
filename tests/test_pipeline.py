import json
import shutil

from telco_churn.data import write_synthetic_data
from telco_churn.monitoring import build_monitoring_report
from telco_churn.scoring import batch_score
from telco_churn.training import train_model


def test_end_to_end_small_pipeline():
    base = "artifacts/test-pipeline"
    shutil.rmtree(base, ignore_errors=True)
    config = {
        "seed": 9,
        "n_customers": 600,
        "paths": {
            "raw": f"{base}/raw.csv",
            "processed": f"{base}/processed.csv",
            "scored": f"{base}/scored.csv",
            "model": f"{base}/model.joblib",
            "metrics": f"{base}/metrics.json",
            "monitoring": f"{base}/monitoring.html",
            "mlflow_fallback": f"{base}/mlflow_fallback",
        },
        "model": {
            "test_size": 0.2,
            "n_estimators": 50,
            "learning_rate": 0.05,
            "max_depth": 5,
            "num_leaves": 31,
            "random_state": 9,
        },
        "monitoring": {"psi_alert_threshold": 0.25, "top_decile_lift_alert_threshold": 3.0},
    }
    from pathlib import Path

    Path(base).mkdir(parents=True, exist_ok=True)
    config_path = Path(base) / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    df = write_synthetic_data(str(config_path))
    metrics = train_model(str(config_path))
    scored = batch_score(str(config_path))
    monitoring = build_monitoring_report(str(config_path))

    assert len(df) == 600
    assert metrics["auc_roc"] > 0.6
    assert len(scored) == 600
    assert "risk_tier" in scored.columns
    assert monitoring["status"] in {"pass", "alert"}
