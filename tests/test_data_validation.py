import pandas as pd
import pytest

from telco_churn.data import generate_customers
from telco_churn.validation import validate_dataframe, raise_if_invalid


def test_generate_customers_is_deterministic():
    left = generate_customers(100, seed=7)
    right = generate_customers(100, seed=7)
    pd.testing.assert_frame_equal(left, right)


def test_generated_customers_validate():
    df = generate_customers(250, seed=7)
    result = validate_dataframe(df)
    assert result.passed, result.errors


def test_validation_rejects_leakage_columns():
    df = generate_customers(50, seed=7)
    df["churn_reason"] = "Moved"
    with pytest.raises(ValueError, match="Leakage-prone"):
        raise_if_invalid(df)

