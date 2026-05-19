from __future__ import annotations

import json

import pandas as pd

from telco_churn.config import ensure_parent, load_config, project_path
from telco_churn.data import generate_customers
from telco_churn.features import FEATURE_COLUMNS, population_stability_index
from telco_churn.scoring import score_dataframe


def build_monitoring_report(config_path: str | None = None) -> dict[str, float | str]:
    config = load_config(config_path) if config_path else load_config()
    baseline = pd.read_csv(project_path(config["paths"]["processed"]))
    current = generate_customers(config["n_customers"], config["seed"] + 7)
    baseline_scored = score_dataframe(baseline, config)
    current_scored = score_dataframe(current, config)

    monitored_features = [column for column in FEATURE_COLUMNS if column != "exchange_churn_rate"]
    feature_psi = {
        column: population_stability_index(baseline_scored[column].to_numpy(), current_scored[column].to_numpy())
        for column in monitored_features
    }
    score_psi = population_stability_index(
        baseline_scored["churn_risk_score"].to_numpy(), current_scored["churn_risk_score"].to_numpy()
    )
    max_feature_psi = max(feature_psi.values())
    threshold = config["monitoring"]["psi_alert_threshold"]
    status = "alert" if max(max_feature_psi, score_psi) > threshold else "pass"
    summary = {
        "status": status,
        "score_psi": float(score_psi),
        "max_feature_psi": float(max_feature_psi),
        "threshold": float(threshold),
    }
    html = _render_html(summary, feature_psi)
    report_path = ensure_parent(config["paths"]["monitoring"])
    report_path.write_text(html, encoding="utf-8")
    json_path = report_path.with_suffix(".json")
    json_path.write_text(json.dumps({"summary": summary, "feature_psi": feature_psi}, indent=2), encoding="utf-8")
    return summary


def _render_html(summary: dict[str, float | str], feature_psi: dict[str, float]) -> str:
    rows = "\n".join(
        f"<tr><td>{feature}</td><td>{value:.4f}</td></tr>" for feature, value in sorted(feature_psi.items())
    )
    return f"""<!doctype html>
<html>
<head><meta charset="utf-8"><title>Telco Churn Monitoring</title></head>
<body>
  <h1>Telco Churn Monitoring Report</h1>
  <p>Status: <strong>{summary["status"]}</strong></p>
  <p>Score PSI: {summary["score_psi"]:.4f}</p>
  <p>Max feature PSI: {summary["max_feature_psi"]:.4f}</p>
  <p>Alert threshold: {summary["threshold"]:.2f}</p>
  <table border="1" cellpadding="6" cellspacing="0">
    <thead><tr><th>Feature</th><th>PSI</th></tr></thead>
    <tbody>{rows}</tbody>
  </table>
</body>
</html>
"""
