import pandas as pd
import numpy as np
import os

def introduce_drift():
    X_train = pd.read_csv("data/processed/X_train.csv")
    y_train = pd.read_csv("data/processed/y_train.csv")

    X_train_drifted = X_train.copy()

    # Apply SAME drift
    X_train_drifted["MedInc"] *= np.random.uniform(1.5, 2.0)
    X_train_drifted["HouseAge"] += np.random.normal(10, 5, size=len(X_train_drifted))
    X_train_drifted["Population"] *= np.random.uniform(0.5, 1.5)

    os.makedirs("data/drifted", exist_ok=True)

    X_train_drifted.to_csv("data/drifted/X_train_drifted.csv", index=False)
    y_train.to_csv("data/drifted/y_train_drifted.csv", index=False)

if __name__ == "__main__":
    introduce_drift()