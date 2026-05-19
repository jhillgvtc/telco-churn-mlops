import numpy as np
import pandas as pd

from telco_churn.features import population_stability_index, recommended_action, score_tier, top_risk_driver


def test_score_tier_boundaries():
    assert score_tier(0.09) == "Very Low"
    assert score_tier(0.10) == "Low"
    assert score_tier(0.25) == "Medium"
    assert score_tier(0.50) == "High"
    assert score_tier(0.75) == "Very High"


def test_recommended_action_routes_rental_high_risk_to_winback():
    row = pd.Series({"risk_tier": "High", "is_rental_proxy": 1, "is_mdu": 0})
    assert "Winback" in recommended_action(row)


def test_top_risk_driver_returns_known_label():
    row = pd.Series(
        {
            "tenure_months": 3,
            "arpu": 120,
            "customers_at_address": 1,
            "is_competitive": 0,
            "has_autopay": 1,
            "internet_only": 0,
        }
    )
    assert top_risk_driver(row) == "short_tenure"


def test_psi_zero_for_identical_distributions():
    values = np.array([1, 2, 3, 4, 5, 6])
    assert population_stability_index(values, values) == 0

