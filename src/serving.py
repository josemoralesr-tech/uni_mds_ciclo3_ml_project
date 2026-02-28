from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import json

MODEL_PATH = "models/random_forest_credit_default.joblib"
FEATURES_PATH = "models/feature_names.json"

app = FastAPI(title="Credit Default - Random Forest API")

model = joblib.load(MODEL_PATH)
feature_names = json.load(open(FEATURES_PATH, "r", encoding="utf-8"))

class PredictRequest(BaseModel):
    features: dict  # {"LIMIT_BAL": 20000, "SEX": 2, ...}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(req: PredictRequest):
    missing = [f for f in feature_names if f not in req.features]
    if missing:
        raise HTTPException(status_code=400, detail=f"Faltan features: {missing[:8]}... total={len(missing)}")

    X = pd.DataFrame([{k: req.features[k] for k in feature_names}])
    pred = int(model.predict(X)[0])
    proba = float(model.predict_proba(X)[:, 1][0])
    return {"prediction": pred, "prob_default": proba}