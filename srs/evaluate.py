import pandas as pd
import joblib
import numpy as np
import os
import mlflow
from sklearn.metrics import mean_squared_error, r2_score


# 🔹 Flexible loader for normal + drifted datasets
def load_data(data_path):
    if "drifted" in data_path:
        X_test = pd.read_csv(os.path.join(data_path, "X_test_drifted.csv"))
        y_test = pd.read_csv(os.path.join(data_path, "y_test_drifted.csv")).values.ravel()
    else:
        X_test = pd.read_csv(os.path.join(data_path, "X_test.csv"))
        y_test = pd.read_csv(os.path.join(data_path, "y_test.csv")).values.ravel()

    return X_test, y_test


# 🔹 Evaluation function with MLflow logging
def evaluate(model_path, data_path, run_name):
    X_test, y_test = load_data(data_path)

    model = joblib.load(model_path)

    preds = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    with mlflow.start_run(run_name=run_name):
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("R2", r2)

    print(f"{run_name} → RMSE: {rmse:.4f}, R2: {r2:.4f}")


# 🔹 Run all stages
if __name__ == "__main__":
    # Baseline (original data)
    evaluate("models/model.pkl", "data/processed", "baseline_eval")

    # Drifted performance (same old model)
    evaluate("models/model.pkl", "data/drifted", "drifted_eval")

    # Retrained model performance
    evaluate("models/model_retrained.pkl", "data/drifted", "retrained_eval")