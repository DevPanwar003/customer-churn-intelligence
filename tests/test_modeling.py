import numpy as np
import pytest
from churn.data import FEATURES, NUMERIC, generate_synthetic, validate_and_clean
from churn.modeling import train_models, feature_importance


@pytest.fixture(scope="module")
def result():
    df, _ = validate_and_clean(generate_synthetic(300))
    return train_models(df)


def test_split_is_stratified_disjoint_and_reproducible(result):
    assert set(result["X_train"].index).isdisjoint(result["X_test"].index)
    assert len(result["X_train"]) + len(result["X_test"]) == 300
    assert abs(result["y_train"].mean() - result["y_test"].mean()) < .03
    again = train_models(generate_synthetic(300))
    assert result["X_test"].index.tolist() == again["X_test"].index.tolist()
    for name in result["models"]:
        np.testing.assert_allclose(result["probabilities"][name], again["probabilities"][name])


def test_feature_allowlist_and_training_only_statistics(result):
    assert list(result["X_train"].columns) == FEATURES
    assert "churn" not in FEATURES and "customer_id" not in FEATURES
    for pipeline in result["models"].values():
        imputer = pipeline.named_steps["preprocess"].named_transformers_["numeric"].named_steps["impute"]
        np.testing.assert_allclose(imputer.statistics_, result["X_train"][NUMERIC].median())


def test_transform_does_not_learn_from_holdout_or_use_extra_fields(result):
    pipeline = result["models"]["Logistic Regression"]
    pre = pipeline.named_steps["preprocess"]
    imputer = pre.named_transformers_["numeric"].named_steps["impute"]
    before = imputer.statistics_.copy()
    test = result["X_test"].copy()
    test["monthly_charges"] = 999999
    test["contract_type"] = "New unseen contract"
    test["churn"] = 1
    test["customer_id"] = "leak"
    prediction = pipeline.predict_proba(test)
    assert np.isfinite(prediction).all()
    np.testing.assert_array_equal(before, imputer.statistics_)
    encoder = pre.named_transformers_["categorical"].named_steps["encode"]
    assert "New unseen contract" not in encoder.categories_[0]
    original = result["X_test"].copy()
    extra = original.assign(churn=1, customer_id="leak", cancellation_reason="leak")
    np.testing.assert_allclose(pipeline.predict_proba(original), pipeline.predict_proba(extra))


def test_metrics_probability_and_matrix_contracts(result):
    assert set(result["metrics"].columns) == {"accuracy", "precision", "recall", "f1", "roc_auc"}
    assert ((result["metrics"] >= 0) & (result["metrics"] <= 1)).all().all()
    for name, probability in result["probabilities"].items():
        assert len(probability) == len(result["y_test"])
        assert np.isfinite(probability).all()
        assert ((probability >= 0) & (probability <= 1)).all()
        matrix = result["matrices"][name]
        assert matrix.shape == (2, 2)
        assert matrix.sum() == len(result["y_test"])
    baseline = result["probabilities"]["Majority baseline"]
    majority = result["y_train"].mode()[0]
    assert (baseline == majority).all()
    assert result["metrics"].loc["Majority baseline", "roc_auc"] == .5


def test_importance_uses_original_features(result):
    importance = feature_importance(result, "Logistic Regression")
    assert set(importance.feature) == set(FEATURES)
    assert np.isfinite(importance[["mean_auc_decrease", "std"]]).all().all()


def test_single_class_fails_cleanly():
    df = generate_synthetic(100, quality_issues=False)
    df["churn"] = 0
    with pytest.raises(ValueError, match="each class"):
        train_models(df)


def test_hand_calculated_threshold_metrics():
    from churn.modeling import evaluate, threshold_comparison
    y = np.array([0, 0, 1, 1])
    p = np.array([.1, .4, .4, .9])
    metrics, matrix = evaluate(y, p, .4)
    np.testing.assert_array_equal(matrix, [[1, 1], [0, 2]])
    assert metrics["precision"] == pytest.approx(2 / 3)
    assert metrics["recall"] == 1
    assert metrics["f1"] == pytest.approx(.8)
    assert metrics["accuracy"] == .75
    assert metrics["roc_auc"] == .875
    table = threshold_comparison(y, p, [.4, .5])
    assert table.flagged_customers.tolist() == [3, 1]
    assert table.recall.tolist() == [1, .5]
    with pytest.raises(ValueError):
        threshold_comparison(y, p, [1.1])


def test_oof_refits_pipeline_without_test_rows(result, monkeypatch):
    from sklearn.pipeline import Pipeline
    from churn.modeling import training_thresholds
    original_fit = Pipeline.fit
    fit_records = []

    def audited_fit(self, X, y=None, **kwargs):
        fitted = original_fit(self, X, y, **kwargs)
        if "preprocess" in self.named_steps:
            imputer = self.named_steps["preprocess"].named_transformers_["numeric"].named_steps["impute"]
            np.testing.assert_allclose(imputer.statistics_, X[NUMERIC].median())
            fit_records.append(set(X.index))
        return fitted

    monkeypatch.setattr(Pipeline, "fit", audited_fit)
    templates = {"Logistic Regression": result["models"]["Logistic Regression"]}
    original_stats = templates["Logistic Regression"].named_steps["preprocess"].named_transformers_["numeric"].named_steps["impute"].statistics_.copy()
    comparison = training_thresholds(result["X_train"], result["y_train"], templates)
    assert len(fit_records) == 5
    validation_indices = []
    for fold, actual_fit in zip(comparison["folds"], fit_records):
        assert actual_fit == set(fold["fit_indices"])
        assert actual_fit.isdisjoint(fold["validation_indices"])
        assert (actual_fit | set(fold["validation_indices"])) == set(result["X_train"].index)
        assert (actual_fit | set(fold["validation_indices"])).isdisjoint(result["X_test"].index)
        validation_indices.extend(fold["validation_indices"])
    assert sorted(validation_indices) == sorted(result["X_train"].index)
    assert np.isfinite(comparison["probabilities"]["Logistic Regression"]).all()
    assert comparison["table"].recall.is_monotonic_decreasing
    np.testing.assert_array_equal(original_stats, templates["Logistic Regression"].named_steps["preprocess"].named_transformers_["numeric"].named_steps["impute"].statistics_)
    again = training_thresholds(result["X_train"], result["y_train"], templates)
    np.testing.assert_allclose(comparison["probabilities"]["Logistic Regression"], again["probabilities"]["Logistic Regression"])


def test_pr_and_average_precision_reference():
    from churn.modeling import precision_recall_data
    y = np.array([0, 0, 0, 1])
    curves, scores = precision_recall_data({"y_test": y,
        "probabilities": {"constant": np.zeros(4), "perfect": np.array([.1, .2, .3, .9])}})
    scores = scores.set_index("model")
    assert scores.loc["constant", "average_precision"] == .25
    assert scores.loc["perfect", "average_precision"] == 1
    assert (scores.no_skill_reference == .25).all()
    for _, curve in curves.groupby("model"):
        assert curve.recall.iloc[0] == 1
        assert curve.recall.iloc[-1] == 0
        assert curve.precision.iloc[-1] == 1
        assert ((curve[["recall", "precision"]] >= 0) & (curve[["recall", "precision"]] <= 1)).all().all()
