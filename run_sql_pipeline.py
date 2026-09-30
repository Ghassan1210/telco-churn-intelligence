"""Build and validate the normalized Telco churn SQLite analytical model."""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path


EXPECTED_ROWS = 7_043
TABLES = ("dim_customers", "dim_contracts", "dim_services", "fact_subscriptions")

COHORT_CHURN_SQL = """
WITH cohort_summary AS (
    SELECT
        tenure_cohort,
        COUNT(*) AS total_customers,
        SUM(CASE WHEN churn_flag = 'Yes' THEN 1 ELSE 0 END) AS churned_customers,
        SUM(CASE WHEN churn_flag = 'Yes' THEN monthly_charges ELSE 0 END) AS lost_mrr
    FROM fact_subscriptions
    GROUP BY tenure_cohort
)
SELECT tenure_cohort, total_customers, churned_customers,
       ROUND(100.0 * churned_customers / total_customers, 2) AS churn_rate_pct,
       ROUND(lost_mrr, 2) AS lost_mrr
FROM cohort_summary
ORDER BY tenure_cohort;
"""

RUNNING_LEAKAGE_SQL = """
WITH RECURSIVE tenure_months(tenure) AS (
    SELECT 0
    UNION ALL
    SELECT tenure + 1 FROM tenure_months WHERE tenure < 11
), monthly_leakage AS (
    SELECT tenure, SUM(monthly_charges) AS lost_mrr
    FROM fact_subscriptions
    WHERE churn_flag = 'Yes' AND tenure BETWEEN 0 AND 11
    GROUP BY tenure
)
SELECT months.tenure,
       ROUND(COALESCE(lost_mrr, 0), 2) AS lost_mrr,
       ROUND(SUM(COALESCE(lost_mrr, 0)) OVER (
           ORDER BY months.tenure ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
       ), 2) AS running_lost_mrr
FROM tenure_months AS months
LEFT JOIN monthly_leakage ON monthly_leakage.tenure = months.tenure
ORDER BY months.tenure;
"""

HIGH_RISK_SQL = """
SELECT customer_id, gender, senior_citizen, tenure, MonthlyCharges,
       TotalCharges, Contract, InternetService, Churn
FROM vw_high_risk_retention_list
ORDER BY MonthlyCharges DESC
LIMIT 5;
"""


def print_query_results(connection: sqlite3.Connection, title: str, query: str) -> None:
    print(f"\n{title}")
    cursor = connection.execute(query)
    headers = [column[0] for column in cursor.description]
    print(" | ".join(headers))
    for row in cursor.fetchall():
        print(" | ".join(str(value) for value in row))


def main() -> int:
    project_root = Path(__file__).resolve().parent
    database_path = project_root / "telco.db"
    sql_path = project_root / "sql" / "master_pipeline.sql"
    if not database_path.is_file():
        print(f"ERROR: Database not found: {database_path}", file=sys.stderr)
        return 1
    if not sql_path.is_file():
        print(f"ERROR: SQL master file not found: {sql_path}", file=sys.stderr)
        return 1

    try:
        with sqlite3.connect(database_path) as connection:
            connection.execute("PRAGMA foreign_keys = ON")
            connection.executescript(sql_path.read_text(encoding="utf-8"))

            row_counts = {
                table: connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                for table in TABLES
            }
            print("Table row counts verification")
            for table, count in row_counts.items():
                print(f"{table}: {count:,}")
            if any(count != EXPECTED_ROWS for count in row_counts.values()):
                raise RuntimeError(f"Row-count validation failed: {row_counts}")

            view_exists = connection.execute(
                "SELECT COUNT(*) FROM sqlite_master WHERE type = 'view' "
                "AND name = 'vw_high_risk_retention_list'"
            ).fetchone()[0]
            if view_exists != 1:
                raise RuntimeError("Analytical view was not created.")

            print_query_results(connection, "Tenure Cohort Churn & Lost MRR summary", COHORT_CHURN_SQL)
            print_query_results(
                connection,
                "Window function running-total revenue leakage (first 12 tenure months)",
                RUNNING_LEAKAGE_SQL,
            )
            print_query_results(connection, "Top 5 high-risk accounts", HIGH_RISK_SQL)
        return 0
    except (OSError, sqlite3.Error, RuntimeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())