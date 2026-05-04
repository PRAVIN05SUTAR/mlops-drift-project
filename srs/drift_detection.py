import pandas as pd
import numpy as np

def detect_drift():
    original = pd.read_csv("data/processed/X_test.csv")
    drifted = pd.read_csv("data/drifted/X_test_drifted.csv")

    print("---- Drift Detection Report ----\n")

    for col in original.columns:
        orig_mean = original[col].mean()
        drift_mean = drifted[col].mean()

        diff = abs(orig_mean - drift_mean)

        print(f"{col}:")
        print(f"  Original Mean: {orig_mean:.2f}")
        print(f"  Drifted Mean : {drift_mean:.2f}")
        print(f"  Difference   : {diff:.2f}")

        # Simple threshold
        if diff > 0.5:
            print("  Drift Detected!\n")
        else:
            print("  No Significant Drift\n")

if __name__ == "__main__":
    detect_drift()