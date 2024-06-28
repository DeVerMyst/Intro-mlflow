import mlflow
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# Charger le modèle MLflow (depuis l'exécution)
run_id = "742ac704423440e4a667b0c97c1df36b"  # Remplacez <run_id> par l'ID de votre exécution
model_name = "modele_regression_lineaire_diabetes"
model_uri = f"runs:/{run_id}/{model_name}"
model = mlflow.pyfunc.load_model(model_uri)


# Définir le schéma des données d'entrée
class InputData(BaseModel):
    data: list


# Définir le point de terminaison de prédiction
@app.post("/predict")
async def predict(input_data: InputData):
    prediction = model.predict(input_data.data)
    return {"prediction": prediction.tolist()}
