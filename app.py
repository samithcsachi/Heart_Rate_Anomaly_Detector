from fastapi import FastAPI
import joblib
import pandas as pd
from pydantic import BaseModel

# ===============================
# Define request schemas
# ===============================
class HeartRateInput(BaseModel):
    feature_values: list

class AnomalyInput(BaseModel):
    feature_values: list

# ===============================
# Load models
# ===============================
heart_model_artifacts = joblib.load("artifacts/model_trainer/Heart_Rate_Predictor_model.joblib")
heart_model = heart_model_artifacts['model']
heart_features = heart_model_artifacts['feature_columns']

anomaly_model_artifacts = joblib.load("artifacts/model_trainer/Anomaly_Detector_model.joblib")
anomaly_model = anomaly_model_artifacts['model']
anomaly_features = anomaly_model_artifacts['feature_columns']

# ===============================
# Create FastAPI app
# ===============================
app = FastAPI(title="Health Monitoring API")

@app.get("/")
def home():
    return {"message": "Health Monitoring API is running!"}

# ===============================
# Heart Rate Prediction (Regression)
# ===============================
@app.post("/predict_heart_rate")
def predict_heart_rate(input_data: HeartRateInput):
    try:
        X = pd.DataFrame([input_data.feature_values], columns=heart_features)
        prediction = heart_model.predict(X)[0]
        return {"heart_rate_prediction": float(prediction)}
    except Exception as e:
        return {"error": str(e)}

# ===============================
# Anomaly Detection (Classification)
# ===============================
@app.post("/detect_anomaly")
def detect_anomaly(input_data: AnomalyInput):
    try:
        X = pd.DataFrame([input_data.feature_values], columns=anomaly_features)
        prediction = anomaly_model.predict(X)[0]
        return {"anomaly_detected": bool(prediction)}
    except Exception as e:
        return {"error": str(e)}
