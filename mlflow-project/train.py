import mlflow
import mlflow.sklearn
import argparse
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import mlflow
import mlflow.sklearn
from sklearn import __version__ as sklearn_version



# Créer un analyseur d'arguments
parser = argparse.ArgumentParser()
parser.add_argument("-t", "--test_size", type=float, default=0.2, help="Proportion des données à utiliser pour l'ensemble de test")
parser.add_argument("-r", "--random_state", type=int, default=42, help="Graine aléatoire pour la reproductibilité")
args = parser.parse_args()

# Charger les données
diabetes = datasets.load_diabetes()
X = diabetes.data
y = diabetes.target

# Diviser les données (en utilisant les paramètres passés en argument)
(X_train,
 X_test,
 y_train,
 y_test) = train_test_split(X,
                            y,
                            test_size=args.test_size,
                            random_state=args.random_state)

# Créer et entraîner le modèle
model = LinearRegression()
model.fit(X_train, y_train)

# Évaluer le modèle
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2s = r2_score(y_test, y_pred)

# Enregistrer les résultats avec MLflow
mlflow.log_metric("mse", mse)
mlflow.log_metric("r2_score", r2s)
mlflow.log_param("test_size", args.test_size)
mlflow.log_param("random_state", args.random_state)
mlflow.log_param("sklearn_version", sklearn_version)
mlflow.log_param("mlflow_version", mlflow.__version__)
mlflow.sklearn.log_model(model, "modele_regression_lineaire_diabetes")
