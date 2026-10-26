# Data and model review

## Review findings: data and model behavior

The dataset and generation code were preserved. CSV SHA-256 before and after review:
`548b2b5276f789d545f671ed59836a5c3b5a56c72d02c060731b010cb4f843fe`.
The target is a Bernoulli draw for a **hypothetical next-30-day churn event**, not an actual
observed event. There are no timestamps, real exposure windows, censoring, or temporal drift.

The clean sample contains 338 churners and 1,462 non-churners (18.78% churn). Training has
253/1,350 churners; test has 85/450 (18.89%). Stratification preserves approximate class balance.

| Input | Observed distribution after cleaning |
|---|---|
| Tenure | Mean 36.25, median 36, range 0–72 months; 36 missing |
| Monthly charges | Mean 59.98, median 59.405, range 18–134.67 fictional units; 36 missing |
| Support calls | Mean 1.277, median 1, range 0–6 |
| Contract | 1,021 month-to-month; 410 one-year; 369 two-year |
| Internet | 863 fiber; 662 DSL; 275 explicitly no internet (`None`) |
| Payment | 1,094 automatic; 670 manual; 36 missing |
| Segment | 832 individual; 690 family; 278 small business |

**Parsing fix:** pandas' default CSV missing-value rules interpreted the valid string `None`
as null. Loading now uses `keep_default_na=False, na_values=[""]`, preserving the no-internet
category while treating blank cells as missing. A regression test locks this behavior. The CSV
was not rewritten. The original rounded model metrics remain unchanged after this semantic fix.

The generator draws tenure uniformly, calls from a Poisson distribution, and categorical inputs
independently with fixed proportions. Charges depend on internet service and segment plus clipped
Gaussian noise. This independence is a simplification: real tenure and contract choices may be
related. Internet and segment influence churn indirectly through charges, not as independent
terms in the target equation. There is no realistic longitudinal customer lifecycle.

### Why ROC-AUC is around 0.67

The signal deliberately overlaps: churn is sampled probabilistically rather than determined by
a hard rule. The generator's latent probability median is 0.165 and 95th percentile is 0.443.
Even the exact known generator probabilities, using original values before injected missingness,
have ROC-AUC **0.6768 on this particular holdout**. This is a diagnostic of the simulation, not a
trained competitor, a strict finite-sample ceiling, or a value available for real customers.
No recovered latent variables enter training, prediction or threshold comparisons.

Logistic Regression is reasonably matched to the additive log-odds generator. The forest's extra
flexibility offers no guaranteed advantage for this process; its leaf-size/depth restrictions and
finite training sample can also limit predictions. Missing inputs remove some information.
These observations explain plausible causes; we did not run ablations that quantify their separate
contributions or establish a statistically significant difference between models. Class imbalance
alone does not determine ROC-AUC, and changing a decision cutoff does not improve ROC-AUC.

### Why recall at 0.50 is low

Most estimated probabilities are below 0.50. On the test holdout, Logistic Regression flags just
16 customers (7 true positives, 9 false positives) and misses 78 churners. The forest flags 9
(3 true positives, 6 false positives) and misses 82. Predicting the majority class gets 81.11%
accuracy while finding no churners. This is a threshold/imbalance trade-off, not evidence that
all customers with a score below 0.50 are safe. No class reweighting or tuning was added to inflate
reported results.

## Threshold comparisons without test-set selection

The existing outer 75/25 split and final models remain unchanged. A separate comparison uses
five stratified folds **inside the 1,350 outer-training rows**, seed 42. For each fold a fresh
clone of the entire pipeline learns imputation, scaling, encoding and the model on four folds,
then predicts the remaining fold. Each training customer receives exactly one out-of-fold score.
The helper takes training features, labels and model templates; it takes no test data.

The dashboard shows precision, recall, F1 and number flagged at fixed cutoffs 0.10 through 0.60.
These are pooled out-of-fold metrics, not averages of fold metrics and not holdout scores.
For illustration, Logistic Regression at 0.20 yields precision 0.316, recall 0.660, F1 0.427;
at 0.50 these are 0.426, 0.079 and 0.133. Lowering the cutoff changes outreach workload and errors.
**No cutoff is selected, recommended, saved as optimal, or applied to the test results.**
Test metrics and confusion matrices retain the original prespecified 0.50 cutoff.

Future threshold decisions need training-only validation and explicit false-positive costs,
missed-churn costs and contact capacity. OOF scores do not prove probability calibration.
Repeatedly choosing configurations from validation scores can overfit validation too; a fresh
final holdout is needed after extensive iteration. The existing exposed test set must not become
a tuning set.

## Precision–Recall evaluation

The holdout Precision–Recall plot includes a horizontal no-skill reference at test prevalence
(0.1889). Average precision (AP) is reported separately from ROC-AUC; it weights precision by
increments of recall and is **not** trapezoidal PR area. A constant-score classifier, including
the majority baseline here, has AP equal to prevalence. Its endpoint (recall 0, precision 1)
is a plotting convention, not successful churn detection.

| Model | Holdout AP | No-skill reference |
|---|---:|---:|
| Majority baseline | 0.1889 | 0.1889 |
| Logistic Regression | 0.3258 | 0.1889 |
| Random Forest | 0.2958 | 0.1889 |

Both models improve on this ranking reference in the synthetic sample. That does not establish
real-world usefulness, calibrated risk, causal retention benefit, or a stable performance advantage.
Do not inspect test PR points to choose a cutoff. AP depends on prevalence, so comparisons across
datasets with different class balance need care.

## Architecture and reproducible review

```text
Documented CSV -> validation + exact deduplication -> cleaned customer table
    |-> SQLite grouping view -> cohort metrics -> descriptive retention hypotheses
    |-> outer stratified split
           |-> training rows -> five-fold cloned pipelines -> OOF threshold comparisons
           |-> training rows -> final pipelines -> test rows -> fixed-cutoff metrics,
                                                       ROC/PR curves, AP, importance
Streamlit app -> cached results -> charts/tables (cohort filters never change model split)
Report CLI -> local CSV/JSON/SQLite artifacts
Diagnostics CLI -> shipped-simulation audit; never supplies model features
```

To reproduce the review, run in the project directory:

```powershell
& '.\.venv\Scripts\python.exe' -m churn.diagnostics
& '.\.venv\Scripts\python.exe' -m churn.report
& '.\.venv\Scripts\python.exe' -m pytest -q --junitxml=artifacts/test-results.xml
```

Additional outputs: `artifacts/data_model_audit.json`, `training_threshold_comparison.csv`,
`holdout_precision_recall.csv`, and `holdout_average_precision.csv`. The evaluation JSON records
fold membership for auditability. The diagnostic command deliberately verifies that the input
matches the shipped simulation before reconstructing latent inputs; it is not for arbitrary data.

Added tests check hand-calculated threshold metrics (including ties at the cutoff), PR/AP baseline
behavior, deterministic OOF predictions, exactly-once validation coverage, train/test separation,
and per-fold imputation statistics. They verify that cloning does not mutate the fitted final model.
The new UI sections are covered by Streamlit AppTest. No dependencies were added.

