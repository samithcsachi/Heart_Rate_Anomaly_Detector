from fastapi import FastAPI
import joblib
import pandas as pd
from datetime import datetime
from typing import Literal, Annotated
from pydantic import BaseModel, Field

# ===============================
# Define request schemas
# ===============================

class HeartRateInput(BaseModel):
    age: Annotated[int, Field(..., gt=0, lt=120, description="The age of the user")]
    gender: Annotated[Literal['M', 'F'], Field(..., description="Gender of the user")]
    weight_kg: Annotated[float, Field(..., gt=0, description='Weight of the user')]
    height_cm: Annotated[float, Field(..., gt=0, lt=250, description='Height of the user')]
    bmi: Annotated[float, Field(..., gt=0, lt=100, description='BMI of the user')]
    fitness_level: Annotated[Literal['lightly_active', 'fairly_active', 'sedentary', 'very_active'], Field(..., description="Fitness level")]
    performance_level: Annotated[Literal['low', 'moderate', 'high'], Field(..., description="Performance level")]
    resting_hr: Annotated[int, Field(..., gt=0, lt=120, description="Resting HR")]
    max_hr: Annotated[int, Field(..., gt=0, lt=220, description="Max HR")]
    activity_type: Annotated[Literal['sleeping', 'walking', 'resting', 'light', 'commuting', 'exercise'], Field(..., description="Activity type")]
    activity_intensity: Annotated[float, Field(..., gt=0.0, description="Activity intensity")]
    steps_5min: Annotated[int, Field(..., gt=0, description="Steps in 5 min")]
    calories_5min: Annotated[float, Field(..., gt=0, description="Calories in 5 min")]
    hrv_rmssd: Annotated[float, Field(..., gt=0, description="Heart rate variability RMSSD")]
    stress_score: Annotated[int, Field(..., gt=0, lt=100, description="Stress score")]
    signal_quality: Annotated[float, Field(..., gt=0, description="Signal quality")]
    skin_temperature: Annotated[float, Field(..., gt=0, description="Skin temperature")]
    device_battery: Annotated[int, Field(..., gt=0, description="Device battery")]
    elevation_gain: Annotated[int, Field(..., ge=0, description="Elevation gain")]
    sleep_stage: Annotated[Literal['light_sleep', 'deep_sleep', 'rem_sleep'], Field(..., description="Sleep stage")]
    date: Annotated[datetime, Field(..., description="Timestamp")]


class AnomalyInput(BaseModel):
    heart_rate: Annotated[float, Field(..., gt=0.0, description="Heart rate")]
    resting_hr_baseline: Annotated[int, Field(..., gt=0, lt=120, description="Resting HR baseline")]
    activity_type: Annotated[Literal['sleeping', 'walking', 'resting', 'light', 'commuting', 'exercise'], Field(..., description="Activity type")]
    activity_intensity: Annotated[float, Field(..., gt=0, description="Activity intensity")]
    steps_5min: Annotated[int, Field(..., gt=0, description="Steps in 5 min")]
    calories_5min: Annotated[float, Field(..., gt=0, description="Calories in 5 min")]
    hrv_rmssd: Annotated[float, Field(..., gt=0, description="Heart rate variability RMSSD")]
    stress_score: Annotated[int, Field(..., gt=0, lt=100, description="Stress score")]
    confidence_score: Annotated[float, Field(..., gt=0.0, description="Confidence score")]
    signal_quality: Annotated[float, Field(..., gt=0, description="Signal quality")]
    skin_temperature: Annotated[float, Field(..., gt=0, description="Skin temperature")]
    device_battery: Annotated[int, Field(..., gt=0, description="Device battery")]
    elevation_gain: Annotated[int, Field(..., ge=0, description="Elevation gain")]
    sleep_stage: Annotated[Literal['light_sleep', 'deep_sleep', 'rem_sleep'], Field(..., description="Sleep stage")]
    date: Annotated[datetime, Field(..., description="Timestamp")]

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
# Utility: preprocess features
# ===============================
def preprocess_heart_features(data_dict: dict) -> pd.DataFrame:
    # Encode datetime
    data_dict['date_encoded'] = data_dict['date'].timestamp()

    # One-hot categorical encodings
    data_dict['gender_M'] = 1 if data_dict['gender'] == 'M' else 0
    data_dict['gender_F'] = 1 if data_dict['gender'] == 'F' else 0

    for act in ['sleeping', 'walking', 'resting', 'light', 'commuting', 'exercise']:
        data_dict[f"activity_type_{act}"] = 1 if data_dict['activity_type'] == act else 0

    for stage in ['light_sleep', 'deep_sleep', 'rem_sleep']:
        data_dict[f"sleep_stage_{stage}"] = 1 if data_dict['sleep_stage'] == stage else 0

    # Restrict to model features only
    return pd.DataFrame([{f: data_dict.get(f, 0) for f in heart_features}])


def preprocess_anomaly_features(data_dict: dict) -> pd.DataFrame:
    data_dict['date_encoded'] = data_dict['date'].timestamp()

    for act in ['sleeping', 'walking', 'resting', 'light', 'commuting', 'exercise']:
        data_dict[f"activity_type_{act}"] = 1 if data_dict['activity_type'] == act else 0

    for stage in ['light_sleep', 'deep_sleep', 'rem_sleep']:
        data_dict[f"sleep_stage_{stage}"] = 1 if data_dict['sleep_stage'] == stage else 0

    return pd.DataFrame([{f: data_dict.get(f, 0) for f in anomaly_features}])

# ===============================
# Endpoints
# ===============================
@app.post("/predict_heart_rate")
def predict_heart_rate(input_data: HeartRateInput):
    try:
        data_dict = input_data.model_dump()
        X = preprocess_heart_features(data_dict)
        prediction = heart_model.predict(X)[0]
        return {"heart_rate_prediction": float(prediction)}
    except Exception as e:
        return {"error": str(e)}


@app.post("/detect_anomaly")
def detect_anomaly(input_data: AnomalyInput):
    try:
        data_dict = input_data.model_dump()
        X = preprocess_anomaly_features(data_dict)
        prediction = anomaly_model.predict(X)[0]
        return {"anomaly_detected": bool(prediction)}
    except Exception as e:
        return {"error": str(e)}
