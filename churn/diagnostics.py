"""Audit the bundled simulation without rewriting data or tuning a model.

Run python -m churn.diagnostics. Generator probabilities are diagnostic only and
never supplied to a predictive pipeline or used to choose a threshold.
"""
import hashlib
import json
import numpy as np
from sklearn.metrics import roc_auc_score
from churn.data import ROOT, DATA_PATH, NUMERIC, CATEGORIES, load_data, generate_synthetic
from churn.modeling import train_models


def main():
    df, quality = load_data()
    result = train_models(df)
    original = generate_synthetic(quality_issues=False)
    # Only audit the shipped sample: require exact reproduction before recovering latent inputs.
    from pandas.testing import assert_frame_equal
    from churn.data import validate_and_clean
    regenerated, _ = validate_and_clean(generate_synthetic())
    assert_frame_equal(df, regenerated, check_dtype=False)
    logit = (-2 + 1.1 * (original.contract_type == "Month-to-month")
             - .022 * original.tenure_months + .013 * (original.monthly_charges - 60)
             + .32 * original.support_calls + .3 * (original.payment_method == "Manual"))
    latent_probability = 1 / (1 + np.exp(-logit))
    audit = {
        "dataset_sha256": hashlib.sha256(DATA_PATH.read_bytes()).hexdigest(),
        "class_counts": {str(k): int(v) for k, v in df.churn.value_counts().items()},
        "churn_share": float(df.churn.mean()), "quality": quality,
        "numeric_summary": df[NUMERIC].describe().to_dict(),
        "category_counts": {col: df[col].fillna("Missing").value_counts().to_dict() for col in CATEGORIES},
        "generator_probability_quantiles": np.quantile(latent_probability, [0, .25, .5, .75, .95, 1]).tolist(),
        "generator_probability_holdout_auc": roc_auc_score(result["y_test"], latent_probability.loc[result["X_test"].index]),
        "caveat": "Known simulation probabilities are not a trained model or a finite-sample performance ceiling; unavailable for real data.",
        "models": {name: {"flagged_at_0_50": int((probability >= .5).sum()),
                          "probability_quantiles": np.quantile(probability, [0, .25, .5, .75, .95, 1]).tolist(),
                          "confusion_matrix": result["matrices"][name].tolist()}
                   for name, probability in result["probabilities"].items()},
    }
    output = ROOT / "artifacts" / "data_model_audit.json"
    output.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
