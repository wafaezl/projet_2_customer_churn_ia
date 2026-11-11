from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
st.set_page_config(page_title="Customer Churn Analytics", layout="wide")
st.title("Customer Churn — Analyse client & IA prédictive")
st.caption("Projet portfolio Data Analyst & Machine Learning | Données synthétiques")

required = [DATA / "customers.csv", DATA / "churn_pipeline.joblib",
            DATA / "feature_importance.csv", DATA / "scored_customers.csv"]
if not all(p.exists() for p in required):
    st.warning("Fichiers absents. Lancez les commandes du README.")
    st.stop()

df = pd.read_csv(DATA / "customers.csv")
scored = pd.read_csv(DATA / "scored_customers.csv")
importance = pd.read_csv(DATA / "feature_importance.csv")
pipeline = joblib.load(DATA / "churn_pipeline.joblib")
metrics_path = DATA / "metrics.json"
metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}

churn_rate = df["churn"].mean() * 100
c1, c2, c3 = st.columns(3)
c1.metric("Clients analysés", f"{len(df):,}")
c2.metric("Taux de churn", f"{churn_rate:.1f}%")
c3.metric("F1-score test", f"{metrics.get('f1', 0):.3f}" if metrics else "—")

st.subheader("Comprendre le churn")
by_contract = df.groupby("contract_type")["churn"].mean().mul(100).sort_values(ascending=False)
st.bar_chart(by_contract)
st.subheader("Importance des variables")
st.bar_chart(importance.set_index("feature")["importance_mean"])
if metrics:
    st.subheader("Évaluation du modèle")
    st.json({k: v for k, v in metrics.items() if k != "confusion_matrix"})
    st.write("Matrice de confusion :", metrics["confusion_matrix"])

st.subheader("Explorer les clients scorés")
risk_filter = st.multiselect("Niveau de risque",
    ["Faible", "Modéré", "Élevé"], default=["Modéré", "Élevé"])
filtered = scored[scored["risk_segment"].isin(risk_filter)]
st.dataframe(filtered.sort_values("churn_probability", ascending=False).head(100),
             use_container_width=True)

st.subheader("Scorer un profil client")
with st.form("score_form"):
    tenure = st.slider("Ancienneté (mois)", 1, 72, 18)
    charges = st.number_input("Facturation mensuelle", 15.0, 150.0, 70.0)
    calls = st.slider("Appels au support (3 derniers mois)", 0, 9, 2)
    late = st.slider("Retards de paiement (6 derniers mois)", 0, 6, 1)
    contract = st.selectbox("Contrat", ["Month-to-month", "One year", "Two years"])
    internet = st.selectbox("Service Internet", ["Fiber", "DSL", "None"])
    payment = st.selectbox("Mode de paiement", ["Card", "Direct debit", "Electronic check"])
    paperless = st.selectbox("Facturation dématérialisée", ["Yes", "No"])
    submitted = st.form_submit_button("Estimer le risque")

if submitted:
    sample = pd.DataFrame([{
        "tenure_months": tenure, "monthly_charges": charges,
        "support_calls_3m": calls, "late_payments_6m": late,
        "contract_type": contract, "internet_service": internet,
        "payment_method": payment, "paperless_billing": paperless
    }])
    probability = pipeline.predict_proba(sample)[0, 1]
    st.metric("Probabilité estimée de churn", f"{probability:.1%}")
    if probability >= 0.65:
        st.warning("Risque élevé — profil à examiner en priorité.")
    elif probability >= 0.35:
        st.info("Risque modéré — surveiller l'évolution du comportement.")
    else:
        st.success("Risque faible selon le modèle.")
