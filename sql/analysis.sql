-- SQLite feature groups use observed values, never model-imputed values.
CREATE VIEW customer_groups AS
SELECT *,
    COALESCE(contract_type, 'Unknown') AS contract_group,
    COALESCE(customer_segment, 'Unknown') AS segment_group,
    CASE WHEN tenure_months IS NULL THEN 'Unknown'
         WHEN tenure_months <= 12 THEN '00-12 months'
         WHEN tenure_months <= 36 THEN '13-36 months'
         ELSE '37+ months' END AS tenure_group,
    CASE WHEN monthly_charges IS NULL THEN 'Unknown'
         WHEN monthly_charges < 50 THEN 'Under 50'
         WHEN monthly_charges < 90 THEN '50 to under 90'
         ELSE '90 or more' END AS charge_group
FROM customers;
