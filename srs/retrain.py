import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

def retrain():
    X_train = pd.read_csv("data/drifted/X_train_drifted.csv")
    y_train = pd.read_csv("data/drifted/y_train_drifted.csv")

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train.values.ravel())

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/model_retrained.pkl")

    print("Retrained model saved!")

if __name__ == "__main__":
    retrain()