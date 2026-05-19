param(
  [Parameter(Mandatory=$true)]
  [ValidateSet("data","validate","train","score","monitor","api","dashboard","test","all")]
  [string]$Task
)

$ErrorActionPreference = "Stop"
$env:PYTHONPATH = "src"
$env:STREAMLIT_BROWSER_GATHER_USAGE_STATS = "false"

switch ($Task) {
  "data" { python -m telco_churn.cli data }
  "validate" { python -m telco_churn.cli validate }
  "train" { python -m telco_churn.cli train }
  "score" { python -m telco_churn.cli score }
  "monitor" { python -m telco_churn.cli monitor }
  "api" { python -m uvicorn telco_churn.api:app --reload --host 127.0.0.1 --port 8000 }
  "dashboard" { python -m streamlit run dashboard/app.py --server.headless true --browser.gatherUsageStats false }
  "test" { python -m pytest -p no:cacheprovider }
  "all" {
    python -m telco_churn.cli data
    python -m telco_churn.cli validate
    python -m telco_churn.cli train
    python -m telco_churn.cli score
    python -m telco_churn.cli monitor
    python -m pytest -p no:cacheprovider
  }
}
