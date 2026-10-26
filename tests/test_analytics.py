import pandas as pd
import pytest
from churn.analytics import analyze, recommendations
from churn.data import generate_synthetic, validate_and_clean


def fixture_data():
    raw = generate_synthetic(20, quality_issues=False).iloc[:4].copy()
    raw["churn"] = [1, 0, 1, 0]
    raw["contract_type"] = ["Month-to-month", "Month-to-month", "One year", "Two year"]
    raw["tenure_months"] = [12, 13, 36, None]
    raw["monthly_charges"] = [49, 50, 90, None]
    return raw


def test_sql_matches_hand_calculated_metrics():
    summary, groups = analyze(fixture_data())
    assert summary["customers"] == 4
    assert summary["churned"] == 2
    assert summary["churn_rate"] == .5
    assert summary["avg_monthly_charges"] == pytest.approx(63)
    contracts = groups["Contract type"].set_index("cohort")
    assert contracts.loc["Month-to-month", "churn_rate"] == .5
    assert contracts.loc["One year", "churn_rate"] == 1
    tenure = groups["Tenure"].set_index("cohort")
    assert tenure.loc["00-12 months", "customers"] == 1
    assert tenure.loc["13-36 months", "customers"] == 2
    assert tenure.loc["Unknown", "churn_rate"] == 0
    charges = groups["Monthly charges"].set_index("cohort")
    assert set(charges.index) == {"Under 50", "50 to under 90", "90 or more", "Unknown"}
    for table in groups.values():
        assert table.customers.sum() == 4
        assert table.churned.sum() == 2


def test_sql_roundtrip_and_pandas_agree(tmp_path):
    df, _ = validate_and_clean(generate_synthetic(100))
    summary, groups = analyze(df, tmp_path / "test.sqlite")
    assert summary["churn_rate"] == pytest.approx(df.churn.mean())
    expected = df.groupby("customer_segment").churn.mean().sort_index()
    actual = groups["Customer segment"].set_index("cohort").churn_rate.sort_index()
    assert actual.tolist() == pytest.approx(expected.tolist())
    # Rebuilding a file-backed database should be idempotent.
    again, _ = analyze(df, tmp_path / "test.sqlite")
    assert again == summary


def test_recommendations_require_evidence_and_use_complement():
    summary, groups = analyze(fixture_data())
    assert recommendations(summary, groups).empty
    result = recommendations(summary, groups, min_customers=1)
    row = result[(result.dimension == "Contract type") & (result.cohort == "One year")].iloc[0]
    assert row.other_customers_rate == pytest.approx(1 / 3)
    assert row.gap_pp == pytest.approx(200 / 3)
    assert "Unknown" not in result.cohort.tolist()
