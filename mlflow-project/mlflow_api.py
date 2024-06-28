import mlflow
from fastapi import FastAPI
from pydantic import BaseModel
import time


app = FastAPI()

# Charger le modèle MLflow (depuis l'exécution)
run_id = (
    "742ac704423440e4a667b0c97c1df36b"  # Remplacez <run_id> par l'ID de votre exécution
)
model_name = "modele_regression_lineaire_diabetes"
model_uri = f"runs:/{run_id}/{model_name}"
model = mlflow.pyfunc.load_model(model_uri)


# Définir le schéma des données d'entrée
class InputData(BaseModel):
    data: list


# Définir le point de terminaison de prédiction
@app.post("/predict")
async def predict(input_data: InputData):
    start_time = time.time()
    prediction = model.predict(input_data.data)

    end_time = time.time()
    response_time = end_time - start_time
    mlflow.log_metric("temps_reponse", response_time)

    return {"prediction": prediction.tolist()}
