from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)

def main():
    rng = np.random.default_rng(24)
    n = 10_000
    tenure = rng.integers(1, 73, n)
    monthly_charges = rng.normal(68, 22, n).clip(15, 150)
    support_calls = rng.poisson(1.4, n).clip(0, 9)
    late_payments = rng.poisson(0.7, n).clip(0, 6)
    contract = rng.choice(["Month-to-month", "One year", "Two years"], n,
                          p=[.55, .27, .18])
    internet = rng.choice(["Fiber", "DSL", "None"], n, p=[.52, .38, .10])
    payment = rng.choice(["Card", "Direct debit", "Electronic check"], n)
    paperless = rng.choice(["Yes", "No"], n, p=[.72, .28])

    # Probabilité synthétique construite pour créer des relations interprétables.
    logit = (-2.0 + 0.045 * support_calls + 0.32 * late_payments
             - 0.035 * tenure + 0.012 * (monthly_charges - 60)
             + 1.05 * (contract == "Month-to-month")
             - 0.65 * (contract == "Two years")
             + 0.35 * (payment == "Electronic check"))
    probability = 1 / (1 + np.exp(-logit))
    churn = rng.binomial(1, probability)

    df = pd.DataFrame({
        "customer_id": [f"C{i:05d}" for i in range(n)],
        "tenure_months": tenure,
        "monthly_charges": monthly_charges.round(2),
        "support_calls_3m": support_calls,
        "late_payments_6m": late_payments,
        "contract_type": contract,
        "internet_service": internet,
        "payment_method": payment,
        "paperless_billing": paperless,
        "churn": churn
    })
    df.to_csv(DATA / "customers.csv", index=False)
    print(f"{len(df):,} profils clients écrits dans data/customers.csv")

if __name__ == "__main__":
    main()
