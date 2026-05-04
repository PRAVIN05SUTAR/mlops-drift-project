import pandas as pd
from sklearn.datasets import fetch_california_housing
import os

def load_data():
    data = fetch_california_housing(as_frame=True)
    df = data.frame

    os.makedirs("data/raw", exist_ok=True)
    df.to_csv("data/raw/housing.csv", index=False)

    print("Dataset saved to data/raw/housing.csv")

if __name__ == "__main__":
    load_data()