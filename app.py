import streamlit as st
import mlflow.sklearn
import pandas as pd
from sklearn.datasets import fetch_california_housing

RUN_ID = "4bfa0cbd112246879b8e37668467ae2c"
# Charger le modèle depuis MLflow
model_uri = (
    f"{RUN_ID}/model"  # Remplacez <RUN_ID> par l'ID de votre meilleure exécution
)
model = mlflow.sklearn.load_model(f"runs:/{model_uri}")

# Charger les données pour obtenir les noms des caractéristiques
housing = fetch_california_housing()
feature_names = housing.feature_names

# Interface Streamlit
st.title("Prédiction du Prix Médian des Maisons en Californie")
st.markdown(
    "Utilisez ce formulaire pour estimer le prix médian d'une maison en Californie."
)

# Formulaire pour les caractéristiques de la maison
input_data = {}
for feature in feature_names:
    input_data[feature] = st.number_input(
        feature, value=housing.data[:, feature_names.index(feature)].mean()
    )

# Bouton pour lancer la prédiction
if st.button("Prédire"):
    # Créer un DataFrame avec les caractéristiques saisies
    input_df = pd.DataFrame([input_data])

    # Prédiction
    prediction = model.predict(input_df)[0]
    st.success(f"Prix médian estimé : ${prediction:.2f}")

# Afficher les informations sur le modèle
with st.expander("Détails du Modèle"):
    st.write(
        "Ce modèle utilise un algorithme de Random Forest pour prédire le prix médian des maisons en Californie."
    )
    st.write(
        f"Le modèle a été entraîné sur le jeu de données California Housing et a obtenu un score R2 de {mlflow.get_run(RUN_ID).data.metrics['R2 score']:.3f} lors de l'évaluation."
    )
