# Customer Churn Intelligence

An end-to-end Python portfolio project for exploring customer churn, comparing predictive models,
and developing evidence-linked retention hypotheses in a navy-and-teal Streamlit dashboard.

**Synthetic-data demonstration:** every customer and outcome is simulated. This is not an IBM
Telco dataset, real customer analysis, or evidence of business impact. Predictions are estimates,
not certainty; associations do not prove causation.

## Features

- Five pages: Overview, Data quality & explorer, SQL analysis, Model evaluation, Retention playbook.
- Interactive contract, segment and tenure filters; Plotly charts; CSV preview and filtered download.
- Schema validation, exact-duplicate removal, missing-value reporting and conflicting-ID checks.
- SQLite cohort analysis by contract, tenure, charges and segment, with visible SQL queries.
- Logistic Regression and Random Forest with train-only preprocessing and a majority baseline.
- Accuracy, precision, recall, F1, ROC-AUC, confusion matrices, ROC/PR curves and average precision.
- Training-only out-of-fold threshold comparisons; holdout permutation feature importance.
- Retention pilot hypotheses linked to observed cohort counts and rates, without causal claims.

## Technology stack

| Component | Technology |
|---|---|
| Language | Python 3.12 |
| Data | pandas, NumPy |
| Modeling | scikit-learn |
| SQL | SQLite via Python's standard-library sqlite3 |
| Dashboard | Streamlit, Plotly |
| Tests | pytest, Streamlit AppTest |

No paid APIs, credentials or separate database server are needed. Direct dependency versions are
pinned in [requirements.txt](requirements.txt); transitive dependencies are not fully locked.

## Project structure

```text
app.py                         Dashboard pages, filters and charts
churn/
  data.py                      Synthetic generation, loading, validation
  analytics.py                 SQLite queries and retention hypotheses
  modeling.py                  Pipelines, splits, metrics, OOF thresholds, PR curves
  report.py                    Reproducible CSV/JSON/SQLite reports
  diagnostics.py               Shipped-simulation audit (not model inputs)
data/
  synthetic_telecom.csv         Included sample, ready to run
  README.md                    Provenance and data dictionary
sql/analysis.sql               SQL cohort grouping view
tests/                         Data, SQL, modeling and dashboard regression tests
docs/model-review.md           Detailed distribution and performance analysis
artifacts/                     Reproducible synthetic evaluation snapshots
.streamlit/config.toml         Theme, localhost binding, telemetry disabled
requirements.txt               Direct dependency pins
pytest.ini                     Test discovery and project-local temporary files
.gitignore                     Environments, secrets, databases, caches and logs
```

The cleaned table feeds SQL exploration and a stratified 75/25 model split. Training rows feed
both final pipelines and a separate five-fold out-of-fold threshold comparison. Holdout rows feed
fixed-cutoff evaluation and interpretation. Sidebar filters never change the model split.

## Setup and launch on Windows

Open PowerShell in this project folder. With the existing environment on this machine:

```powershell
Set-Location 'C:\Users\devpa\Documents\Customer-Churn-Intelligence'
& '.\.venv\Scripts\python.exe' -m pip install -r requirements.txt
& '.\.venv\Scripts\python.exe' -m pytest -q
& '.\.venv\Scripts\python.exe' -m streamlit run app.py
```

Open [the local dashboard](http://127.0.0.1:8501). Stop with Ctrl+C. If port 8501 is already in
use, stop the earlier dashboard or append `--server.port 8502` and open that port instead.
No shell activation or execution-policy change is needed.

For a fresh Windows checkout, install Python 3.12 with the Python launcher, open PowerShell in
the checkout, then create an isolated environment **only if `.venv` does not exist**:

```powershell
py -3.12 -m venv .venv
& '.\.venv\Scripts\python.exe' -m pip install -r requirements.txt
& '.\.venv\Scripts\python.exe' -m pip check
& '.\.venv\Scripts\python.exe' -m pytest -q
& '.\.venv\Scripts\python.exe' -m streamlit run app.py
```

If `py` is unavailable, use the full path to your Python 3.12 executable instead. This machine
has neither `py` nor `python` on PATH; its existing `.venv` is usable through the commands above.
It was originally created from the bundled Python with `--system-site-packages` to reuse pandas
and NumPy. The final audit additionally uses `.venv-audit`, an isolated environment with no shared
site packages, to check the requirements independently. Both environments are excluded from Git.
Recreate virtual environments after moving a project; do not commit or copy them to another PC.

Optional reproducible reports (no external calls):

```powershell
& '.\.venv\Scripts\python.exe' -m churn.report
& '.\.venv\Scripts\python.exe' -m churn.diagnostics
```

The included CSV needs no regeneration. `python -m churn.data` recreates the seeded sample and
**overwrites** that CSV; use the environment's Python executable if Python is not on PATH.

## Dataset

[Dataset documentation](data/README.md) describes every feature and the exact simulation formula.
Seed 42 produces 1,800 unique customers plus two exact duplicates. Cleaning removes duplicates;
36 values are missing in each of tenure, charges and payment method. CSV loading preserves the
valid no-internet category `None` rather than interpreting it as missing.

There are 338 churners (18.78%). The target is a hypothetical next-30-day departure, generated as
a Bernoulli draw. No actual dates, outcomes, people or companies are represented. Numeric features
are imputed only inside training pipelines; missing labels and conflicting IDs fail validation.

## Evaluation

Same stratified split for all models: 1,350 training / 450 test rows, seed 42. Customer IDs and
targets are excluded by an explicit feature allowlist. Median imputation, standardization and
one-hot encoding are fitted within each training pipeline; unseen categories are handled safely.

Measured holdout results at the prespecified **0.50 cutoff**:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Average precision |
|---|---:|---:|---:|---:|---:|---:|
| Majority baseline | 0.8111 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.1889 |
| Logistic Regression | 0.8067 | 0.4375 | 0.0824 | 0.1386 | 0.6713 | 0.3258 |
| Random Forest | 0.8044 | 0.3333 | 0.0353 | 0.0638 | 0.6615 | 0.2958 |

Positive class is churn=1. Confusion-matrix rows are actual and columns predicted, ordered [0,1].
Undefined precision is reported as zero. Accuracy alone is misleading: the baseline finds no
churners despite 81.11% accuracy. Both learned models have poor recall at 0.50. The generator's
known probabilities score 0.6768 ROC-AUC on this holdout, consistent with noisy overlapping
outcomes; this is not a strict finite-sample ceiling or a real-world performance benchmark.

PR curves use holdout prevalence (0.1889) as the no-skill reference. Average precision is not
trapezoidal PR area. The constant-score baseline's precision=1, recall=0 endpoint is a plotting
convention, not useful prediction. Model probability estimates are not calibrated.

Threshold comparisons at 0.10–0.60 use five-fold **out-of-fold predictions on training rows only**.
Each fold refits the full pipeline. No threshold is selected, recommended or applied to test
results. Lower thresholds increase outreach and often recall at the expense of precision.
Permutation importance uses holdout ROC-AUC decreases over five shuffles; error bars are shuffle
standard deviations, not confidence intervals. Importance is neither causal effect nor direction.

See [the detailed review](docs/model-review.md), [metric snapshot](artifacts/model_metrics.csv),
and [training threshold comparison](artifacts/training_threshold_comparison.csv).

## Screenshots

Screenshot files are not included yet; there are no broken image placeholders. To add genuine
screenshots, run the app and use Windows Snipping Tool (Win+Shift+S). Save images under
`docs/screenshots/` and add relative Markdown links only after those files exist.

Capture the Overview, data-quality audit, fixed-cutoff evaluation, PR comparison and training-only
threshold comparison. Keep the synthetic-data notice, axes and validation/test labels visible.
Use a wide window for side-by-side plots and inspect a narrow window for responsive behavior.
Screenshots illustrate the product; they are not evidence of real-world performance.

## Tests and verification

```powershell
& '.\.venv\Scripts\python.exe' -m pytest -q --junitxml=artifacts/test-results.xml
```

Tests cover data rejection/cleaning, SQL arithmetic and bins, deterministic disjoint stratified
splits, train-only imputation, excluded fields, probabilities, baseline behavior, metric arithmetic,
OOF fold isolation, PR/AP reference behavior, all five dashboard pages and empty filters.
Streamlit AppTest checks page rendering and interactions; it is not a browser pixel-layout test.
The test XML and temporary files are excluded from Git. See [the final audit](docs/portfolio-audit.md)
for the actual final run and remaining publication tasks.

## Repository hygiene

Only source, documentation, the documented synthetic sample and explicitly allowed evaluation
CSV/JSON snapshots belong in Git. `.gitignore` excludes environments (including `.venv-audit`),
bytecode, pytest caches, local secrets, `.env` files, logs, databases and SQLite sidecar files.
The app uses in-memory SQLite; the report CLI recreates the ignored `artifacts/churn.sqlite`.
Generated reports are simulation evidence, not real customer data. Review staged files before
committing. There is no API key requirement and no credentials should be added.

## Limitations

- Synthetic, simplified independent inputs and invented outcomes; no external or temporal validation.
- One holdout split; no confidence intervals or statistically established model winner.
- Low recall at the default cutoff; no cost-optimized threshold or validated probability calibration.
- Repeated inspection can bias development; do not tune models against the exposed holdout.
- No fairness/subgroup audit, customer outreach workflow, deployment or production monitoring.
- Cohort gaps are descriptive, potentially confounded and overlapping; pilot ideas do not prove uplift.
- Direct dependencies are pinned, but transitive dependencies are not fully locked.

## Future improvements

1. Add a properly licensed public dataset with explicit provenance and prediction/outcome windows.
2. Use training-only cross-validation for tuning, calibration and cost-aware threshold policies.
3. Reserve a fresh final holdout; add temporal validation, uncertainty estimates and subgroup checks.
4. Add genuine screenshots and an automated CI test workflow after choosing repository hosting.
5. Evaluate retention interventions with randomized pilots before claiming incremental impact.

These are future work, not implemented capabilities. The current project is a local portfolio demo.
