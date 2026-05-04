from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI()

# Load retrained model (best model)
model = joblib.load("models/model_retrained.pkl")


@app.get("/")
def home():
    return {"message": "MLOps Drift Detection API Running"}


@app.post("/predict")
def predict(data: dict):
    try:
        features = np.array(list(data.values())).reshape(1, -1)
        prediction = model.predict(features)[0]

        return {
            "prediction": float(prediction)
        }

    except Exception as e:
        return {"error": str(e)}