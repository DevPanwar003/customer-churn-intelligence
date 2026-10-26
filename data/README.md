# Synthetic telecom sample

**Source: generated locally by `churn/data.py`; NOT a public dataset and NOT actual customers.**
The verified project folder was empty, so no documented local telecom dataset was available.
No external dataset was downloaded. This project does not claim IBM/Telco provenance.

Run `python -m churn.data` to reproduce `synthetic_telecom.csv` with NumPy seed 42.
The file has 1,800 unique simulated customers plus two exact duplicate rows (1,802 raw rows).
There are 36 missing values in each of tenure, monthly charges, and payment method among
unique customers. Missingness is random and injected after the outcome is sampled.

| Column | Meaning at hypothetical snapshot | Allowed values / units |
|---|---|---|
| customer_id | Artificial row identifier; excluded from models | SYN-xxxxx |
| tenure_months | Completed service months | Integer 0-72; missing allowed |
| monthly_charges | Current recurring bill | Fictional currency units 18-135; missing allowed |
| support_calls | Service calls in the preceding 30 days | Nonnegative integer |
| contract_type | Current commitment | Month-to-month, One year, Two year |
| internet_service | Current service | DSL, Fiber optic, None |
| payment_method | Current payment approach | Automatic, Manual; missing allowed |
| customer_segment | Pre-outcome household/account category | Individual, Family, Small business |
| churn | Simulated departure within next 30 days | 0 stayed, 1 churned |

These are invented snapshot semantics; no dates or longitudinal observations exist.
Charges depend on service, segment and Gaussian noise. Churn is Bernoulli-sampled from:

`sigmoid(-2 + 1.1*(month-to-month) - .022*tenure + .013*(charges-60) + .32*support_calls + .3*(manual_payment))`

This intentional structure gives the models learnable associations with irreducible noise.
Contract proportions, customer characteristics and coefficients are illustrative assumptions,
not industry estimates. Feature importance and recommendations partly recover these assumptions.
The data cannot establish actual telecom churn rates, model generalization, fairness, savings,
causality, or intervention effectiveness. Do not present its results as business impact.
