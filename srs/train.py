import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import mlflow
import mlflow.sklearn
import os

def train():
    X_train = pd.read_csv("data/processed/X_train.csv")
    y_train = pd.read_csv("data/processed/y_train.csv")

    mlflow.set_experiment("Housing Model")

    with mlflow.start_run(run_name="baseline_model"):
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train.values.ravel())

        mlflow.log_param("n_estimators", 100)

        mlflow.sklearn.log_model(model, "model")

        os.makedirs("models", exist_ok=True)
        mlflow.sklearn.save_model(model, "models/baseline_model")

        print("Baseline model logged!")

if __name__ == "__main__":
    train()