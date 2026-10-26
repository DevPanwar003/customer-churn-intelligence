"""Write reproducible local evaluation artifacts: python -m churn.report."""
import json
import platform
from importlib.metadata import version
from churn.data import ROOT, DATA_PATH, load_data
from churn.analytics import analyze, recommendations
from churn.modeling import train_models, feature_importance, training_thresholds, precision_recall_data
import hashlib


def main():
    df, quality = load_data()
    output = ROOT / "artifacts"
    output.mkdir(exist_ok=True)
    summary, groups = analyze(df, output / "churn.sqlite")
    result = train_models(df)
    result["metrics"].to_csv(output / "model_metrics.csv")
    recommendations(summary, groups).to_csv(output / "retention_hypotheses.csv", index=False)
    for name in ["Logistic Regression", "Random Forest"]:
        feature_importance(result, name).to_csv(output / (name.lower().replace(" ", "_") + "_importance.csv"), index=False)
    thresholds = training_thresholds(result["X_train"], result["y_train"], result["models"])
    thresholds["table"].to_csv(output / "training_threshold_comparison.csv", index=False)
    curves, ap = precision_recall_data(result)
    curves.to_csv(output / "holdout_precision_recall.csv", index=False)
    ap.to_csv(output / "holdout_average_precision.csv", index=False)
    metadata = {"threshold_comparison_source": "5-fold OOF on outer training only; no threshold selected",
        "threshold_validation_folds": thresholds["folds"],
        "source": "SYNTHETIC; invented simulation, not actual customers",
        "dataset_sha256": hashlib.sha256(DATA_PATH.read_bytes()).hexdigest(),
        "seed": 42, "test_fraction": .25, "threshold": .5,
        "train_rows": len(result["X_train"]), "test_rows": len(result["X_test"]),
        "quality": quality, "summary": summary,
        "confusion_matrices": {k: v.tolist() for k, v in result["matrices"].items()},
        "python": platform.python_version(),
        "packages": {p: version(p) for p in ["pandas", "numpy", "scikit-learn", "streamlit", "plotly", "pytest"]}}
    (output / "evaluation.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(result["metrics"].round(4).to_string())
    print(f"Saved artifacts to {output}")


if __name__ == "__main__":
    main()
