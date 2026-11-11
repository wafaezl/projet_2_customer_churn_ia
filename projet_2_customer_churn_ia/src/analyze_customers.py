from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "data" / "customers.csv")

summary = df.groupby("contract_type", as_index=False).agg(
    customers=("customer_id", "count"),
    churn_rate=("churn", "mean"),
    avg_monthly_charges=("monthly_charges", "mean"),
    avg_tenure=("tenure_months", "mean")
)
summary["churn_rate_pct"] = (summary["churn_rate"] * 100).round(1)
summary.to_csv(ROOT / "data" / "churn_by_contract.csv", index=False)

print("Taux global de churn :", round(df["churn"].mean() * 100, 1), "%")
print("\nAnalyse par type de contrat :")
print(summary.to_string(index=False))
print("\nStatistiques descriptives :")
print(df[["tenure_months", "monthly_charges", "support_calls_3m",
          "late_payments_6m"]].describe().round(2))
