from fastapi import FastAPI
import joblib
import pandas as pd
from datetime import datetime
from pydantic import BaseModel

# ===============================
# Define request schemas
# ===============================
# schemas.py

class HeartRateInput(BaseModel):
    user_id: str
    age: int
    gender: str
    weight_kg: float
    height_cm: float
    bmi: float
    fitness_level: str
    performance_level: str
    resting_hr: int
    max_hr: int
    activity_type: str
    activity_intensity: float
    steps_5min: int
    calories_5min: float
    hrv_rmssd: float
    stress_score: int
    signal_quality: float
    skin_temperature: float
    device_battery: int
    elevation_gain: int
    sleep_stage: str
    date: datetime


class AnomalyInput(BaseModel):
    heart_rate: float
    resting_hr_baseline: int
    activity_type: str
    activity_intensity: float
    steps_5min: int
    calories_5min: float
    hrv_rmssd: float
    stress_score: int
    confidence_score: float
    signal_quality: float
    skin_temperature: float
    device_battery: int
    elevation_gain: int
    sleep_stage: str
    date: datetime


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
        data_dict = input_data.dict()

       
        data_dict['date_encoded'] = data_dict['date'].timestamp()

      
        data_dict['gender_M'] = 1 if data_dict['gender'].lower() == 'male' else 0
        data_dict['gender_F'] = 1 if data_dict['gender'].lower() == 'female' else 0

        data_dict['activity_type_running'] = 1 if data_dict['activity_type'] == 'running' else 0
        data_dict['activity_type_walking'] = 1 if data_dict['activity_type'] == 'walking' else 0
  

        data_dict['sleep_stage_light'] = 1 if data_dict['sleep_stage'] == 'light' else 0
        data_dict['sleep_stage_deep'] = 1 if data_dict['sleep_stage'] == 'deep' else 0
     

    
        X = pd.DataFrame([{f: data_dict.get(f, 0) for f in heart_features}])

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
        data_dict = input_data.dict()

       
        data_dict['date_encoded'] = data_dict['date'].timestamp()

     
        activity_types = ['running', 'walking', 'exercise'] 
        for act in activity_types:
            col_name = f"activity_type_{act}"
            data_dict[col_name] = 1 if data_dict['activity_type'] == act else 0

       
        sleep_stages = ['light', 'deep', 'rem']  
        for stage in sleep_stages:
            col_name = f"sleep_stage_{stage}"
            data_dict[col_name] = 1 if data_dict['sleep_stage'] == stage else 0

       
        X = pd.DataFrame([{f: data_dict.get(f, 0) for f in anomaly_features}])

        prediction = anomaly_model.predict(X)[0]
        return {"anomaly_detected": bool(prediction)}

    except Exception as e:
        return {"error": str(e)}

