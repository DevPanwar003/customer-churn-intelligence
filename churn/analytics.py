"""SQLite metrics and evidence-linked, non-causal retention hypotheses."""
from contextlib import closing
import sqlite3
import pandas as pd
from churn.data import ROOT

GROUPS = {"Contract type": "contract_group", "Tenure": "tenure_group",
          "Monthly charges": "charge_group", "Customer segment": "segment_group"}
SUMMARY_SQL = """SELECT COUNT(*) AS customers, SUM(churn) AS churned,
    AVG(1.0 * churn) AS churn_rate, AVG(monthly_charges) AS avg_monthly_charges
    FROM customer_groups"""


def group_query(label):
    column = GROUPS[label]  # fixed allowlist; never interpolate user SQL
    return f"""SELECT {column} AS cohort, COUNT(*) AS customers,
        SUM(churn) AS churned, AVG(1.0 * churn) AS churn_rate
        FROM customer_groups GROUP BY {column} ORDER BY churn_rate DESC, cohort"""


def analyze(df, db_path=":memory:"):
    """A local in-memory SQLite database by default; optional file for CLI export."""
    with closing(sqlite3.connect(str(db_path))) as conn:
        conn.execute("DROP VIEW IF EXISTS customer_groups")
        df.to_sql("customers", conn, index=False, if_exists="replace")
        conn.executescript((ROOT / "sql" / "analysis.sql").read_text())
        summary = pd.read_sql_query(SUMMARY_SQL, conn).iloc[0].to_dict()
        groups = {name: pd.read_sql_query(group_query(name), conn) for name in GROUPS}
    return summary, groups


def recommendations(summary, groups, min_customers=30):
    """Compare each eligible cohort with its complement; output pilot hypotheses."""
    ideas = {
        "Contract type": "Test clearer renewal communication or an optional contract offer",
        "Tenure": "Test onboarding check-ins and early service support",
        "Monthly charges": "Test a bill review and plan-fit conversation",
        "Customer segment": "Test segment-specific service interviews and support",
    }
    findings = []
    for dimension, table in groups.items():
        for row in table.itertuples():
            rest_n = summary["customers"] - row.customers
            if row.cohort == "Unknown" or row.customers < min_customers or rest_n < min_customers:
                continue
            rest_rate = (summary["churned"] - row.churned) / rest_n
            gap = row.churn_rate - rest_rate
            if gap > 0:
                findings.append({"dimension": dimension, "cohort": row.cohort,
                    "customers": row.customers, "churned": row.churned,
                    "churn_rate": row.churn_rate, "other_customers_rate": rest_rate,
                    "gap_pp": gap * 100, "pilot_hypothesis": ideas[dimension]})
    return pd.DataFrame(findings).sort_values("gap_pp", ascending=False).reset_index(drop=True) if findings else pd.DataFrame()
