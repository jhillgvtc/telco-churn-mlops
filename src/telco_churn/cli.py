from __future__ import annotations

import argparse
import json

import pandas as pd

from telco_churn.config import load_config, project_path
from telco_churn.data import write_synthetic_data
from telco_churn.monitoring import build_monitoring_report
from telco_churn.scoring import batch_score
from telco_churn.training import train_model
from telco_churn.validation import raise_if_invalid


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthetic telco churn MLOps pipeline")
    parser.add_argument("command", choices=["data", "validate", "train", "score", "monitor"])
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    config = load_config(args.config) if args.config else load_config()

    if args.command == "data":
        df = write_synthetic_data(args.config)
        print(f"generated {len(df):,} rows")
    elif args.command == "validate":
        df = pd.read_csv(project_path(config["paths"]["processed"]))
        raise_if_invalid(df)
        print("validation passed")
    elif args.command == "train":
        print(json.dumps(train_model(args.config), indent=2))
    elif args.command == "score":
        scored = batch_score(args.config)
        print(f"scored {len(scored):,} rows")
    elif args.command == "monitor":
        print(json.dumps(build_monitoring_report(args.config), indent=2))


if __name__ == "__main__":
    main()

