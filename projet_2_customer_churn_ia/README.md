# Analyse du comportement client et prédiction du churn

Projet portfolio Data Analyst & IA : analyse des clients, segmentation descriptive,
classification du risque de désabonnement et explication des facteurs associés.

## Fonctionnalités
- Génération de 10 000 profils clients synthétiques.
- Analyse exploratoire des comportements et du taux de churn.
- Pipeline de classification avec prétraitement numérique/catégoriel.
- Évaluation par précision, rappel, F1-score, ROC-AUC et matrice de confusion.
- Importance des variables par permutation.
- Application Streamlit pour explorer les segments et scorer des clients.

## Installation
```bash
python -m venv .venv
# Windows : .venv\Scripts\activate
# macOS/Linux : source .venv/bin/activate
pip install -r requirements.txt
```

## Exécution
```bash
python src/generate_data.py
python src/analyze_customers.py
python src/train_churn.py
streamlit run app.py
```

Les données sont synthétiques et le modèle est pédagogique. Ne pas interpréter les
scores comme une performance réelle sur une base client d'entreprise.
