# MLOps Drift Detection Project

## Overview
End-to-end MLOps pipeline with drift detection, retraining, experiment tracking (MLflow), API serving (FastAPI), Dockerization, and CI/CD (GitHub Actions).

## Architecture
Data → Preprocess → Train → Drift Simulation → Evaluate (MLflow) → Serve (FastAPI in Docker) → Monitor

## How to run (local)
```bash
pip install -r requirements.txt
python srs/train.py
python srs/evaluate.py
uvicorn api.main:app --reload
