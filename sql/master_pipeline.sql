PRAGMA foreign_keys = ON;

DROP VIEW IF EXISTS vw_high_risk_retention_list;
DROP TABLE IF EXISTS fact_subscriptions;
DROP TABLE IF EXISTS dim_services;
DROP TABLE IF EXISTS dim_contracts;
DROP TABLE IF EXISTS dim_customers;

CREATE TABLE dim_customers (
    customer_id TEXT PRIMARY KEY,
    gender TEXT NOT NULL,
    senior_citizen INTEGER NOT NULL,
    partner TEXT NOT NULL,
    dependents TEXT NOT NULL
);

CREATE TABLE dim_contracts (
    customer_id TEXT PRIMARY KEY REFERENCES dim_customers(customer_id),
    contract_type TEXT NOT NULL,
    paperless_billing TEXT NOT NULL,
    payment_method TEXT NOT NULL
);

CREATE TABLE dim_services (
    customer_id TEXT PRIMARY KEY REFERENCES dim_customers(customer_id),
    phone_service TEXT NOT NULL,
    multiple_lines TEXT NOT NULL,
    internet_service TEXT NOT NULL,
    online_security TEXT NOT NULL,
    online_backup TEXT NOT NULL,
    device_protection TEXT NOT NULL,
    tech_support TEXT NOT NULL,
    streaming_tv TEXT NOT NULL,
    streaming_movies TEXT NOT NULL,
    security_bundle TEXT NOT NULL
);

CREATE TABLE fact_subscriptions (
    customer_id TEXT PRIMARY KEY REFERENCES dim_customers(customer_id),
    tenure INTEGER NOT NULL,
    tenure_cohort TEXT NOT NULL,
    monthly_charges REAL NOT NULL,
    total_charges REAL NOT NULL,
    churn_flag TEXT NOT NULL
);

INSERT INTO dim_customers (customer_id, gender, senior_citizen, partner, dependents)
SELECT customerID, gender, SeniorCitizen, Partner, Dependents
FROM stg_telco;

INSERT INTO dim_contracts (customer_id, contract_type, paperless_billing, payment_method)
SELECT customerID, Contract, PaperlessBilling, PaymentMethod
FROM stg_telco;

INSERT INTO dim_services (
    customer_id, phone_service, multiple_lines, internet_service,
    online_security, online_backup, device_protection, tech_support,
    streaming_tv, streaming_movies, security_bundle
)
SELECT
    customerID, PhoneService, MultipleLines, InternetService,
    OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport,
    StreamingTV, StreamingMovies,
    CASE
        WHEN OnlineSecurity = 'Yes'
         AND OnlineBackup = 'Yes'
         AND DeviceProtection = 'Yes'
         AND TechSupport = 'Yes' THEN 'Full Protection'
        WHEN OnlineSecurity = 'No'
         AND OnlineBackup = 'No'
         AND DeviceProtection = 'No'
         AND TechSupport = 'No' THEN 'No Protection'
        ELSE 'Partial Protection'
    END
FROM stg_telco;

INSERT INTO fact_subscriptions (
    customer_id, tenure, tenure_cohort, monthly_charges, total_charges, churn_flag
)
SELECT
    customerID,
    tenure,
    CASE
        WHEN tenure BETWEEN 0 AND 12 THEN '01. 0-12 Mos'
        WHEN tenure BETWEEN 13 AND 24 THEN '02. 13-24 Mos'
        WHEN tenure BETWEEN 25 AND 48 THEN '03. 25-48 Mos'
        ELSE '04. 49+ Mos'
    END,
    MonthlyCharges,
    TotalCharges,
    Churn
FROM stg_telco;

CREATE INDEX idx_fact_subscriptions_churn_flag
    ON fact_subscriptions (churn_flag);
CREATE INDEX idx_dim_contracts_contract_type
    ON dim_contracts (contract_type);
CREATE INDEX idx_dim_services_internet_service
    ON dim_services (internet_service);

CREATE VIEW vw_high_risk_retention_list AS
SELECT
    f.customer_id,
    c.gender,
    c.senior_citizen,
    f.tenure,
    f.monthly_charges AS MonthlyCharges,
    f.total_charges AS TotalCharges,
    ct.contract_type AS Contract,
    s.internet_service AS InternetService,
    f.churn_flag AS Churn
FROM fact_subscriptions AS f
JOIN dim_customers AS c ON c.customer_id = f.customer_id
JOIN dim_contracts AS ct ON ct.customer_id = f.customer_id
JOIN dim_services AS s ON s.customer_id = f.customer_id
WHERE f.churn_flag = 'No'
  AND ct.contract_type = 'Month-to-month'
  AND f.tenure <= 6
  AND f.monthly_charges >= 70.00
ORDER BY MonthlyCharges DESC;

-- Tenure Cohort Churn & Lost MRR summary
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

-- Window function running-total revenue leakage for the first 12 tenure months
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

-- Top 5 high-risk accounts
SELECT customer_id, gender, senior_citizen, tenure, MonthlyCharges,
       TotalCharges, Contract, InternetService, Churn
FROM vw_high_risk_retention_list
ORDER BY MonthlyCharges DESC
LIMIT 5;