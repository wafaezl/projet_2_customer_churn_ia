from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
TARGET = "churn"
DROP = ["customer_id", TARGET]

def main():
    df = pd.read_csv(DATA / "customers.csv")
    X = df.drop(columns=DROP)
    y = df[TARGET]

    numeric = X.select_dtypes(include="number").columns.tolist()
    categorical = X.select_dtypes(exclude="number").columns.tolist()
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical)
    ])
    model = RandomForestClassifier(
        n_estimators=350, min_samples_leaf=3, class_weight="balanced",
        random_state=42, n_jobs=-1
    )
    pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    pipeline.fit(X_train, y_train)
    pred = pipeline.predict(X_test)
    proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist()
    }
    (DATA / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    joblib.dump(pipeline, DATA / "churn_pipeline.joblib")

    importance = permutation_importance(
        pipeline, X_test, y_test, n_repeats=8, random_state=42,
        scoring="f1", n_jobs=-1
    )
    importance_df = pd.DataFrame({
        "feature": X.columns,
        "importance_mean": importance.importances_mean,
        "importance_std": importance.importances_std
    }).sort_values("importance_mean", ascending=False)
    importance_df.to_csv(DATA / "feature_importance.csv", index=False)

    scored = X_test.copy()
    scored["actual_churn"] = y_test.values
    scored["churn_probability"] = proba
    scored["risk_segment"] = pd.cut(
        proba, bins=[-0.001, 0.35, 0.65, 1.001],
        labels=["Faible", "Modéré", "Élevé"]
    )
    scored.to_csv(DATA / "scored_customers.csv", index=False)

    print("Métriques du modèle :")
    print(json.dumps(metrics, indent=2))
    print("\nRapport de classification :")
    print(classification_report(y_test, pred, zero_division=0))

if __name__ == "__main__":
    main()
