from telco_churn.data import EXCHANGES, SUBDIVISIONS, generate_customers


REAL_SERVICE_AREA_TERMS = {
    "boerne",
    "bulverde",
    "canyon lake",
    "smithson",
    "new braunfels",
    "gonzales",
    "vintage oaks",
    "meyer ranch",
    "rockin j",
    "herff",
    "singing hills",
    "estraya",
}


def test_synthetic_labels_do_not_use_real_service_area_terms():
    labels = " ".join(EXCHANGES + SUBDIVISIONS).lower()
    leaked_terms = [term for term in REAL_SERVICE_AREA_TERMS if term in labels]
    assert leaked_terms == []


def test_generated_customer_labels_are_public_safe():
    df = generate_customers(500, seed=42)
    labels = " ".join(df["exchange"].unique().tolist() + df["subdivision"].unique().tolist()).lower()
    leaked_terms = [term for term in REAL_SERVICE_AREA_TERMS if term in labels]
    assert leaked_terms == []

