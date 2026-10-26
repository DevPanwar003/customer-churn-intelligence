"""Train-only preprocessing, fixed holdout evaluation, and permutation importance."""
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve, average_precision_score)
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from churn.data import FEATURES, NUMERIC, CATEGORIES, validate_and_clean

SEED = 42


def make_pipeline(estimator):
    numeric = Pipeline([("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                        ("scale", StandardScaler())])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="Unknown", keep_empty_features=True)),
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    preprocess = ColumnTransformer([
        ("numeric", numeric, NUMERIC), ("categorical", categorical, list(CATEGORIES))],
        remainder="drop")
    return Pipeline([("preprocess", preprocess), ("model", estimator)])


def evaluate(y, probability, threshold=.5):
    probability = np.asarray(probability)
    if not 0 <= threshold <= 1 or not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError("Probabilities and threshold must be in [0, 1].")
    prediction = (probability >= threshold).astype(int)
    metrics = {
        "accuracy": accuracy_score(y, prediction),
        "precision": precision_score(y, prediction, zero_division=0),
        "recall": recall_score(y, prediction, zero_division=0),
        "f1": f1_score(y, prediction, zero_division=0),
        "roc_auc": roc_auc_score(y, probability),
    }
    return metrics, confusion_matrix(y, prediction, labels=[0, 1])


def train_models(df):
    df, _ = validate_and_clean(df)
    if len(df) < 40 or df.churn.value_counts().reindex([0, 1], fill_value=0).min() < 8:
        raise ValueError("Training requires at least 40 rows and eight examples of each class.")
    X = df[FEATURES].copy()  # explicit allowlist excludes ID, target and any added fields
    y = df.churn.copy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.25, random_state=SEED, stratify=y)
    estimators = {
        "Majority baseline": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=SEED),
        "Random Forest": RandomForestClassifier(n_estimators=160, min_samples_leaf=6,
                                                 max_depth=10, random_state=SEED, n_jobs=1),
    }
    models, probabilities, matrices, rows = {}, {}, {}, []
    for name, estimator in estimators.items():
        pipeline = make_pipeline(estimator)
        pipeline.fit(X_train, y_train)
        probabilities[name] = pipeline.predict_proba(X_test)[:, 1]
        metrics, matrices[name] = evaluate(y_test, probabilities[name])
        rows.append({"model": name, **metrics})
        models[name] = pipeline
    return {"models": models, "probabilities": probabilities,
            "metrics": pd.DataFrame(rows).set_index("model"), "matrices": matrices,
            "X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test}


def feature_importance(result, name):
    importance = permutation_importance(result["models"][name], result["X_test"],
        result["y_test"], scoring="roc_auc", n_repeats=5, random_state=SEED, n_jobs=1)
    return pd.DataFrame({"feature": FEATURES, "mean_auc_decrease": importance.importances_mean,
                         "std": importance.importances_std}).sort_values("mean_auc_decrease")


def roc_data(result):
    rows = []
    for name, probability in result["probabilities"].items():
        fpr, tpr, _ = roc_curve(result["y_test"], probability)
        rows.extend({"model": name, "false_positive_rate": x, "true_positive_rate": y}
                    for x, y in zip(fpr, tpr))
    return pd.DataFrame(rows)


THRESHOLDS = (.10, .20, .30, .40, .50, .60)


def threshold_comparison(y, probability, thresholds=THRESHOLDS):
    """Descriptive comparisons only; never select or apply an optimum."""
    rows = []
    for threshold in thresholds:
        metrics, _ = evaluate(y, probability, threshold)
        rows.append({"threshold": threshold,
                     **{key: metrics[key] for key in ["precision", "recall", "f1"]},
                     "flagged_customers": int((np.asarray(probability) >= threshold).sum())})
    return pd.DataFrame(rows)


def training_thresholds(X_train, y_train, models):
    """Five-fold OOF probabilities using ONLY outer-training rows.

    Clone/refit the entire pipeline in every fold, including imputation and scaling.
    This function deliberately accepts no test features or labels. No threshold is chosen.
    """
    if y_train.value_counts().reindex([0, 1], fill_value=0).min() < 5:
        raise ValueError("Five-fold validation requires five training examples per class.")
    folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(X_train, y_train))
    rows, probabilities = [], {}
    for name, template in models.items():
        oof = np.full(len(y_train), np.nan)
        for fit_positions, validation_positions in folds:
            pipeline = clone(template)
            pipeline.fit(X_train.iloc[fit_positions], y_train.iloc[fit_positions])
            oof[validation_positions] = pipeline.predict_proba(X_train.iloc[validation_positions])[:, 1]
        probabilities[name] = oof
        table = threshold_comparison(y_train, oof)
        table.insert(0, "model", name)
        rows.append(table)
    return {"table": pd.concat(rows, ignore_index=True), "probabilities": probabilities,
            "folds": [{"fit_indices": X_train.index[fit].tolist(),
                       "validation_indices": X_train.index[valid].tolist()} for fit, valid in folds]}


def precision_recall_data(result):
    """Holdout ranking diagnostics, not threshold selection.

    Constant-score AP equals evaluation prevalence, the no-skill PR reference.
    Average precision is a step-weighted summary, not trapezoidal PR area.
    """
    curves, summaries = [], []
    prevalence = float(result["y_test"].mean())
    for name, probability in result["probabilities"].items():
        precision, recall, _ = precision_recall_curve(result["y_test"], probability)
        curves.extend({"model": name, "recall": r, "precision": p}
                      for p, r in zip(precision, recall))
        summaries.append({"model": name, "average_precision": average_precision_score(result["y_test"], probability),
                          "no_skill_reference": prevalence})
    return pd.DataFrame(curves), pd.DataFrame(summaries)
