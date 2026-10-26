"""Generate and validate a documented, reproducible synthetic customer snapshot."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "synthetic_telecom.csv"
NUMERIC = ["tenure_months", "monthly_charges", "support_calls"]
CATEGORIES = {
    "contract_type": ["Month-to-month", "One year", "Two year"],
    "internet_service": ["DSL", "Fiber optic", "None"],
    "payment_method": ["Automatic", "Manual"],
    "customer_segment": ["Individual", "Family", "Small business"],
}
FEATURES = NUMERIC + list(CATEGORIES)
REQUIRED = ["customer_id"] + FEATURES + ["churn"]


def generate_synthetic(n=1800, seed=42, quality_issues=True):
    """Features at a hypothetical snapshot; churn is a simulated next-30-day outcome.

    Coefficients are invented simulation assumptions, not real telecom evidence.
    Missingness is injected after sampling the outcome, independently of churn.
    """
    if n < 20:
        raise ValueError("Generate at least 20 customers.")
    rng = np.random.default_rng(seed)
    tenure = rng.integers(0, 73, n)
    contract = rng.choice(CATEGORIES["contract_type"], n, p=[.56, .25, .19])
    internet = rng.choice(CATEGORIES["internet_service"], n, p=[.35, .49, .16])
    payment = rng.choice(CATEGORIES["payment_method"], n, p=[.6, .4])
    segment = rng.choice(CATEGORIES["customer_segment"], n, p=[.47, .38, .15])
    charges = np.clip(35 + 40 * (internet == "Fiber optic") +
                      15 * (segment == "Family") + rng.normal(0, 13, n), 18, 135).round(2)
    calls = rng.poisson(1.3, n)
    logit = (-2.0 + 1.1 * (contract == "Month-to-month") - .022 * tenure +
             .013 * (charges - 60) + .32 * calls + .3 * (payment == "Manual"))
    df = pd.DataFrame({
        "customer_id": [f"SYN-{i:05d}" for i in range(n)],
        "tenure_months": tenure, "monthly_charges": charges, "support_calls": calls,
        "contract_type": contract, "internet_service": internet,
        "payment_method": payment, "customer_segment": segment,
        "churn": rng.binomial(1, 1 / (1 + np.exp(-logit))),
    })
    if quality_issues:
        for col in ["monthly_charges", "tenure_months", "payment_method"]:
            df.loc[rng.choice(n, max(1, n // 50), replace=False), col] = np.nan
        df = pd.concat([df, df.iloc[:2]], ignore_index=True)
    return df


def validate_and_clean(raw):
    """Remove exact duplicates; fail on ambiguous IDs, invalid values or labels.

    Leave feature nulls intact: model imputers must learn on training data only.
    Unknown or missing target labels are never imputed.
    """
    missing = sorted(set(REQUIRED) - set(raw.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if raw.empty:
        raise ValueError("Dataset is empty.")
    df = raw[REQUIRED].copy()
    for col in ["customer_id"] + list(CATEGORIES):
        df[col] = df[col].astype(object).map(
            lambda x: np.nan if pd.isna(x) or str(x).strip() == "" else str(x).strip())
    duplicates = int(df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)
    if df.customer_id.isna().any() or df.customer_id.duplicated().any():
        raise ValueError("Customer IDs must be present and unique; conflicting duplicates found.")
    for col in NUMERIC + ["churn"]:
        source = df[col].replace(r"^\s*$", np.nan, regex=True)
        numeric = pd.to_numeric(source, errors="coerce")
        if (source.notna() & numeric.isna()).any() or np.isinf(numeric).any():
            raise ValueError(f"Invalid numeric value in {col}.")
        df[col] = numeric
    if df.churn.isna().any() or not df.churn.isin([0, 1]).all():
        raise ValueError("Churn must contain only known labels 0 and 1.")
    for col in NUMERIC:
        if df[col].notna().sum() == 0 or (df[col].dropna() < 0).any():
            raise ValueError(f"{col} must have observed, nonnegative values.")
    for col in ["tenure_months", "support_calls"]:
        if (df[col].dropna() % 1 != 0).any():
            raise ValueError(f"{col} must contain whole numbers.")
    for col, allowed in CATEGORIES.items():
        if not df[col].dropna().isin(allowed).all():
            raise ValueError(f"Unexpected category in {col}.")
    df["churn"] = df.churn.astype(int)
    report = {"input_rows": len(raw), "clean_rows": len(df),
              "exact_duplicates_removed": duplicates,
              "missing_by_column": df.isna().sum().to_dict(),
              "ignored_columns": sorted(set(raw.columns) - set(REQUIRED))}
    return df, report


def load_data(path=DATA_PATH):
    return validate_and_clean(pd.read_csv(path, keep_default_na=False, na_values=[""]))


def main():
    DATA_PATH.parent.mkdir(exist_ok=True)
    generate_synthetic().to_csv(DATA_PATH, index=False, lineterminator="\n")
    print(f"Wrote SYNTHETIC data: {DATA_PATH}")


if __name__ == "__main__":
    main()
