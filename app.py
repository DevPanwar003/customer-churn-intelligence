r"""Run with: .\.venv\Scripts\python.exe -m streamlit run app.py"""
import pandas as pd
import plotly.express as px
import streamlit as st
from churn.data import DATA_PATH, load_data
from churn.analytics import analyze, recommendations, GROUPS, SUMMARY_SQL, group_query
from churn.modeling import train_models, feature_importance, roc_data, training_thresholds, precision_recall_data

st.set_page_config(page_title="Customer Churn Intelligence", page_icon="◈", layout="wide")
st.markdown("""<style>
.block-container {max-width:1440px;padding-top:4rem;}
h1 {letter-spacing:-.035em;} h2,h3 {letter-spacing:-.02em;}
[data-testid="stMetric"] {background:#10243B;border:1px solid #23405B;
 border-radius:12px;padding:18px;}
[data-testid="stSidebar"] {border-right:1px solid #23405B;}
.eyebrow {color:#2DD4BF;letter-spacing:.18em;font-size:.8rem;font-weight:700;}
.hero-note {color:#94A3B8;max-width:900px;}
@media(max-width:640px) {.block-container {padding:4rem 1rem 1rem;} h1 {font-size:2rem;}}
</style>""", unsafe_allow_html=True)


@st.cache_data
def get_data(file_bytes):
    # Preserve the literal None service category; parser semantics reviewed in V1.1.
    # File bytes are part of the cache key, so regenerated data invalidates caches.
    return load_data()


@st.cache_resource
def get_models(df):
    return train_models(df)


@st.cache_data
def get_importance(df, model_name):
    return feature_importance(get_models(df), model_name)


@st.cache_data
def get_thresholds(df):
    result = get_models(df)
    return training_thresholds(result["X_train"], result["y_train"], result["models"])


def chart(fig):
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font_color="#E2E8F0", margin=dict(l=12, r=12, t=40, b=20))
    axis_labels = {"cohort": "Customer cohort", "churn_rate": "Observed churn rate",
        "monthly_charges": "Monthly charge (fictional units)", "count": "Customers",
        "false_positive_rate": "False positive rate", "true_positive_rate": "Recall",
        "mean_auc_decrease": "Mean decrease in ROC-AUC", "feature": "Input feature"}
    fig.for_each_xaxis(lambda axis: axis.update(title_text=axis_labels.get(axis.title.text, axis.title.text)))
    fig.for_each_yaxis(lambda axis: axis.update(title_text=axis_labels.get(axis.title.text, axis.title.text)))
    st.plotly_chart(fig, width="stretch")


st.sidebar.markdown("### ◈ CHURN INTELLIGENCE")
st.sidebar.caption("PORTFOLIO PROJECT 03 · VERSION 1")
page = st.sidebar.radio("Workspace", ["Overview", "Data quality & explorer", "SQL analysis",
                                     "Model evaluation", "Retention playbook"])
st.sidebar.info("SYNTHETIC DATA\n\nSimulated customers. No real telecom findings.")
try:
    df, quality = get_data(DATA_PATH.read_bytes())
except (FileNotFoundError, ValueError) as error:
    st.error(f"Cannot load sample dataset: {error}")
    st.code(r".\.venv\Scripts\python.exe -m churn.data")
    st.stop()

st.markdown('<div class="eyebrow">CUSTOMER ANALYTICS / SYNTHETIC DEMO</div>', unsafe_allow_html=True)
st.title("Customer Churn Intelligence")
st.markdown('<p class="hero-note">Explore customer patterns, compare predictive models, and turn observed associations into testable retention ideas.</p>', unsafe_allow_html=True)

contracts = sorted(df.contract_type.dropna().unique())
segments = sorted(df.customer_segment.dropna().unique())
with st.sidebar.expander("Explore cohorts", expanded=True):
    selected_contracts = st.multiselect("Contract type", contracts, default=contracts)
    selected_segments = st.multiselect("Customer segment", segments, default=segments)
    tenure = st.slider("Tenure range (months)", 0, int(df.tenure_months.max()),
                       (0, int(df.tenure_months.max())))
    include_missing = st.checkbox("Include unknown tenure", value=True)
mask = df.contract_type.isin(selected_contracts) & df.customer_segment.isin(selected_segments)
tenure_mask = df.tenure_months.between(*tenure) | (df.tenure_months.isna() & include_missing)
filtered = df.loc[mask & tenure_mask].copy()
st.sidebar.caption(f"{len(filtered):,} of {len(df):,} customers selected. Filters affect EDA, SQL and recommendations. Model evaluation always uses the full cleaned dataset.")

if page == "Model evaluation":
    st.header("Model evaluation")
    st.caption("Full cleaned dataset · stratified 75/25 split · seed 42 · fixed threshold 0.50")
    with st.spinner("Training and evaluating models on the fixed split..."):
        result = get_models(df)
    a, b, c = st.columns(3)
    a.metric("Training customers", len(result["X_train"]))
    b.metric("Holdout customers", len(result["X_test"]))
    c.metric("Holdout churn share", f'{result["y_test"].mean():.1%}')
    st.dataframe(result["metrics"].style.format("{:.3f}"), width="stretch")
    st.info("Class imbalance can make a model that predicts no churn appear accurate. Compare recall (churners found), precision (flagged customers who churn), F1 and ROC-AUC against the majority baseline. Undefined precision is reported as 0.")
    st.caption("No class reweighting or resampling in V1. The fixed threshold is illustrative; choose a future operating threshold using training-only validation and outreach costs. This holdout is for evaluation, not repeated tuning.")
    name = st.selectbox("Inspect model", ["Logistic Regression", "Random Forest", "Majority baseline"])
    left, right = st.columns(2)
    with left:
        st.subheader("Confusion matrix")
        chart(px.imshow(result["matrices"][name], x=["Stay", "Churn"], y=["Stay", "Churn"],
            text_auto=True, color_continuous_scale="Teal", labels={"x": "Predicted", "y": "Actual", "color": "Customers"}))
    with right:
        st.subheader("ROC comparison")
        fig = px.line(roc_data(result), x="false_positive_rate", y="true_positive_rate", color="model",
                      color_discrete_sequence=["#94A3B8", "#2DD4BF", "#60A5FA"])
        fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash", color="#64748B"))
        chart(fig)
    st.subheader("Precision–Recall comparison · test holdout")
    pr_curve, ap_scores = precision_recall_data(result)
    fig = px.line(pr_curve, x="recall", y="precision", color="model",
                  labels={"recall": "Recall", "precision": "Precision"},
                  color_discrete_sequence=["#94A3B8", "#2DD4BF", "#60A5FA"])
    fig.add_hline(y=float(result["y_test"].mean()), line_dash="dash",
                  annotation_text="No-skill reference: holdout churn share")
    fig.update_xaxes(range=[0, 1])
    fig.update_yaxes(range=[0, 1.05])
    chart(fig)
    st.dataframe(ap_scores.style.format({"average_precision": "{:.3f}", "no_skill_reference": "{:.3f}"}), hide_index=True)
    st.caption("Average precision (AP) summarizes precision across recall levels; it is not trapezoidal PR area. A constant score has AP equal to churn prevalence. The majority baseline flags nobody at 0.50; its PR endpoint at precision 1, recall 0 is a plotting convention, not useful performance. This curve is descriptive, not a test-set threshold selector.")
    st.subheader("Threshold trade-offs · training-only validation")
    with st.spinner("Computing five-fold out-of-fold estimates on training rows..."):
        comparisons = get_thresholds(df)["table"]
    selected = comparisons[comparisons.model == name]
    st.dataframe(selected.style.format({"threshold": "{:.2f}", "precision": "{:.3f}", "recall": "{:.3f}", "f1": "{:.3f}"}), hide_index=True)
    chart(px.line(selected.melt(id_vars="threshold", value_vars=["precision", "recall", "f1"],
                                var_name="Metric", value_name="Score"),
                  x="threshold", y="Score", color="Metric", markers=True,
                  labels={"threshold": "Decision threshold"}))
    st.info("These comparisons use five-fold out-of-fold predictions from the outer training rows only. Each fold refits all preprocessing. No threshold is selected or applied; test metrics and confusion matrices remain at 0.50. Lower cutoffs usually find more churners but increase outreach and false positives. Choose a policy on training validation with explicit costs before a future final test.")
    if name != "Majority baseline":
        st.subheader("Which inputs does this model rely on?")
        importance = get_importance(df, name)
        chart(px.bar(importance, x="mean_auc_decrease", y="feature", orientation="h",
                     error_x="std", color_discrete_sequence=["#2DD4BF"]))
        st.caption("Holdout permutation importance: mean ROC-AUC decrease after shuffling one original input, over five repeats. Bars show ±1 standard deviation across shuffles, not confidence intervals. Negative values are possible. Correlated inputs can share importance; importance gives neither causal effect nor direction.")
    with st.expander("Example holdout estimates"):
        estimates = pd.DataFrame({"customer_id": df.loc[result["X_test"].index, "customer_id"],
                                  "actual_churn": result["y_test"],
                                  "estimated_churn_probability": result["probabilities"][name]})
        st.dataframe(estimates.head(30), hide_index=True, width="stretch")
        st.warning("Predicted churn is an estimate, not certainty. Probabilities are not calibrated or validated on real customers. Do not use this demo for customer decisions.")
    st.stop()

if filtered.empty:
    st.warning("No customers match these filters. Select at least one contract and segment, or widen the tenure range.")
    st.stop()
summary, groups = analyze(filtered)

if page == "Overview":
    a, b, c, d = st.columns(4)
    a.metric("Selected customers", f'{int(summary["customers"]):,}')
    b.metric("Observed churn", f'{summary["churn_rate"]:.1%}')
    c.metric("Churned customers", f'{int(summary["churned"]):,}')
    d.metric("Avg. monthly charge", f'{summary["avg_monthly_charges"]:.2f}')
    st.caption("All figures describe synthetic observations in the selected cohort. Charges are fictional currency units; averages exclude missing charges.")
    left, right = st.columns(2)
    with left:
        st.subheader("Churn by contract")
        chart(px.bar(groups["Contract type"], x="cohort", y="churn_rate", hover_data=["customers", "churned"],
                     color_discrete_sequence=["#2DD4BF"]).update_layout(yaxis_tickformat=".0%"))
    with right:
        st.subheader("Tenure and churn")
        chart(px.bar(groups["Tenure"].sort_values("cohort"), x="cohort", y="churn_rate", hover_data=["customers", "churned"],
                     color_discrete_sequence=["#60A5FA"]).update_layout(yaxis_tickformat=".0%"))
    st.subheader("Monthly charge distribution")
    chart(px.histogram(filtered.assign(outcome=filtered.churn.map({0:"Stayed", 1:"Churned"})),
                       x="monthly_charges", color="outcome", nbins=30, barmode="overlay",
                       color_discrete_map={"Stayed":"#2DD4BF", "Churned":"#FB923C"}, opacity=.7))
    st.caption("Missing charges are excluded from this histogram. These associations partly reflect the invented data generator.")

elif page == "Data quality & explorer":
    st.header("Data quality & explorer")
    st.caption("Quality audit below covers the entire source file; preview and descriptive statistics follow sidebar filters.")
    a, b, c = st.columns(3)
    a.metric("Source rows", quality["input_rows"])
    b.metric("Exact duplicates removed", quality["exact_duplicates_removed"])
    c.metric("Remaining duplicate IDs", int(df.customer_id.duplicated().sum()))
    st.dataframe(pd.DataFrame.from_dict(quality["missing_by_column"], orient="index", columns=["Missing after deduplication"]), width="stretch")
    st.info("Feature missing values remain visible in EDA. Model pipelines use training-set medians for numeric values and an explicit Unknown category for categorical values. Missing labels, invalid categories and conflicting IDs fail validation.")
    st.subheader("Dataset preview")
    st.dataframe(filtered.head(100), hide_index=True, width="stretch")
    st.download_button("Download selected synthetic customers", filtered.to_csv(index=False), "synthetic_customer_selection.csv", "text/csv")
    st.subheader("Descriptive statistics")
    st.dataframe(filtered.describe(include="all").astype(str), width="stretch")
    st.caption("Source: churn/data.py, NumPy random generator, seed 42. See data/README.md for the full data dictionary and simulation assumptions.")

elif page == "SQL analysis":
    st.header("SQL cohort analysis")
    dimension = st.selectbox("Group customers by", list(GROUPS))
    table = groups[dimension]
    chart(px.bar(table, x="cohort", y="churn_rate", hover_data=["customers", "churned"],
                 color_discrete_sequence=["#2DD4BF"]).update_layout(yaxis_tickformat=".0%"))
    st.dataframe(table.style.format({"churn_rate": "{:.1%}"}), hide_index=True, width="stretch")
    st.caption("SQLite operates on selected, deduplicated rows. Rates are churned / customers, not percentages averaged across cohorts. Unknown numeric values get their own group. Small cohorts are descriptive and unstable.")
    with st.expander("See executed SQL"):
        st.code(SUMMARY_SQL, language="sql")
        st.code(group_query(dimension), language="sql")
        st.code((DATA_PATH.parents[1] / "sql" / "analysis.sql").read_text(), language="sql")

elif page == "Retention playbook":
    st.header("From association to a testable idea")
    st.warning("Synthetic evidence only. Associations do not establish causation or prove that an intervention will reduce churn.")
    findings = recommendations(summary, groups)
    if findings.empty:
        st.info("No eligible higher-churn cohort found. Each cohort and its comparison group need at least 30 customers; widen the filters.")
    else:
        st.caption("Ranked by percentage-point gap versus all other selected customers in that dimension. At least 30 customers required on each side; overlapping cohorts are not independent.")
        for row in findings.head(4).itertuples():
            with st.container(border=True):
                st.subheader(f"{row.dimension}: {row.cohort}")
                st.write(f"Observed churn: **{row.churn_rate:.1%}** ({row.churned}/{row.customers}) versus **{row.other_customers_rate:.1%}** among other selected customers; gap **{row.gap_pp:.1f} percentage points**.")
                st.write(f"Hypothesis: {row.pilot_hypothesis}.")
        st.dataframe(findings, hide_index=True, width="stretch")
    st.markdown("**Validate before acting:** confirm the pattern on representative real data, check confounding and cohort sizes, then run a randomized pilot. Predefine a 30-day churn outcome, outreach cost, customer experience guardrails and a control group. Report uncertainty and incremental impact; this demo estimates neither savings nor treatment effects.")
