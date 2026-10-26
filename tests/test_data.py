import numpy as np
import pandas as pd
import pytest
from churn.data import generate_synthetic, validate_and_clean, load_data


def test_generation_is_reproducible():
    pd.testing.assert_frame_equal(generate_synthetic(), generate_synthetic())


def test_sample_quality_and_missingness():
    df, report = load_data()
    assert len(df) == 1800
    assert report["input_rows"] == 1802
    assert report["exact_duplicates_removed"] == 2
    assert df.customer_id.is_unique
    assert report["missing_by_column"]["monthly_charges"] == 36
    assert set(df.churn) == {0, 1}


@pytest.mark.parametrize("column,value", [
    ("churn", np.nan), ("churn", 2), ("tenure_months", -1),
    ("tenure_months", 1.5), ("monthly_charges", float("inf")),
    ("monthly_charges", "broken"), ("contract_type", "invalid"),
    ("customer_id", ""), ("support_calls", -1)])
def test_invalid_values_fail(column, value):
    df = generate_synthetic(100, quality_issues=False)
    df[column] = df[column].astype(object)
    df.loc[0, column] = value
    with pytest.raises(ValueError):
        validate_and_clean(df)


def test_missing_columns_and_empty_data_fail():
    df = generate_synthetic(100)
    with pytest.raises(ValueError, match="required columns"):
        validate_and_clean(df.drop(columns="churn"))
    with pytest.raises(ValueError, match="empty"):
        validate_and_clean(df.iloc[:0])


def test_conflicting_duplicate_id_fails():
    df = generate_synthetic(100, quality_issues=False)
    df.loc[1, "customer_id"] = df.loc[0, "customer_id"]
    with pytest.raises(ValueError, match="unique"):
        validate_and_clean(df)


def test_whitespace_missing_features_preserved_and_extra_columns_ignored():
    df = generate_synthetic(100, quality_issues=False)
    df["monthly_charges"] = df.monthly_charges.astype(object)
    df.loc[0, "monthly_charges"] = " "
    df.loc[0, "payment_method"] = " "
    df["cancellation_reason"] = "post-outcome leakage"
    clean, report = validate_and_clean(df)
    assert pd.isna(clean.loc[0, "monthly_charges"])
    assert pd.isna(clean.loc[0, "payment_method"])
    assert report["ignored_columns"] == ["cancellation_reason"]


def test_csv_preserves_none_service_category():
    df, report = load_data()
    assert (df.internet_service == "None").sum() == 275
    assert report["missing_by_column"]["internet_service"] == 0
