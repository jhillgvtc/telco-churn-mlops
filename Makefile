.PHONY: data validate train score monitor api dashboard test all

data:
	PYTHONPATH=src python -m telco_churn.cli data

validate:
	PYTHONPATH=src python -m telco_churn.cli validate

train:
	PYTHONPATH=src python -m telco_churn.cli train

score:
	PYTHONPATH=src python -m telco_churn.cli score

monitor:
	PYTHONPATH=src python -m telco_churn.cli monitor

api:
	PYTHONPATH=src uvicorn telco_churn.api:app --reload --host 127.0.0.1 --port 8000

dashboard:
	streamlit run dashboard/app.py

test:
	PYTHONPATH=src pytest

all: data validate train score monitor test
