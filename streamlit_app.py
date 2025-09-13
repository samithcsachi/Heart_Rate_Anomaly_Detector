import streamlit as st
from huggingface_hub import hf_hub_download
import joblib
import pandas as pd
from datetime import datetime



# ==========================
# Preprocessing function
# ==========================
def preprocess_heart_input(payload, feature_columns):
    df = pd.DataFrame([payload])

    # Encode gender
    df['gender_M'] = (df['gender'] == 'M').astype(int)
    df['gender_F'] = (df['gender'] == 'F').astype(int)

    # One-hot encode fitness_level
    for lvl in ['lightly_active', 'sedentary', 'very_active', 'fairly_active']:
        col_name = f'fitness_level_{lvl}'
        df[col_name] = (df['fitness_level'] == lvl).astype(int)

    # One-hot encode performance_level
    for lvl in ['low', 'moderate', 'high']:
        col_name = f'performance_level_{lvl}'
        df[col_name] = (df['performance_level'] == lvl).astype(int)

    # One-hot encode activity_type
    for act in ['sleeping', 'walking', 'resting', 'light', 'commuting', 'exercise']:
        col_name = f'activity_type_{act}'
        df[col_name] = (df['activity_type'] == act).astype(int)

    # One-hot encode sleep_stage
    for stage in ['light_sleep', 'deep_sleep', 'rem_sleep']:
        col_name = f'sleep_stage_{stage}'
        df[col_name] = (df['sleep_stage'] == stage).astype(int)

    # Encode date
    df['date_encoded'] = pd.to_datetime(df['date']).astype(int) // 10**9

    # Make sure all model columns exist, fill missing with 0
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0

    # Keep only the columns the model expects
    df = df[feature_columns]

    return df



def preprocess_anomaly_input(payload, feature_columns):
    df = pd.DataFrame([payload])

    # One-hot encode activity_type
    for act in ['sleeping', 'walking', 'resting', 'light', 'commuting', 'exercise']:
        col_name = f'activity_type_{act}'
        df[col_name] = (df['activity_type'] == act).astype(int)

    # One-hot encode sleep_stage
    for stage in ['light_sleep', 'deep_sleep', 'rem_sleep']:
        col_name = f'sleep_stage_{stage}'
        df[col_name] = (df['sleep_stage'] == stage).astype(int)

    # Encode date
    df['date_encoded'] = pd.to_datetime(df['date']).astype(int) // 10**9

    # Ensure all model features exist, fill missing with 0
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0

    # Keep only the columns the model expects
    df = df[feature_columns]

    return df



# ======================
# Load models from Hugging Face
# ======================
HF_REPO = "samithcs/heart-rate-models"
HEART_MODEL_FILENAME = "Heart_Rate_Predictor_model.joblib"
ANOMALY_MODEL_FILENAME = "Anomaly_Detector_model.joblib"

@st.cache_resource
def load_model(filename: str):
    local_path = hf_hub_download(repo_id=HF_REPO, filename=filename)
    artifacts = joblib.load(local_path)
    return artifacts['model'], artifacts['feature_columns']

heart_model, heart_features = load_model(HEART_MODEL_FILENAME)
anomaly_model, anomaly_features = load_model(ANOMALY_MODEL_FILENAME)

# ======================
#  Streamlit UI
# ======================
st.set_page_config(page_title="Health Monitoring App", layout="centered")
st.title("Health Monitoring Dashboard")

st.sidebar.header("Choose Prediction Task")
task = st.sidebar.radio("Select Task:", ["Heart Rate Prediction", "Anomaly Detection"])

# ======================
# Heart Rate Prediction
# ======================
if task == "Heart Rate Prediction":
    st.subheader("Predict Heart Rate")

    age = st.number_input("Age", min_value=1, max_value=120, value=25)
    gender = st.radio("Gender", ["M", "F"])
    weight_kg = st.number_input("Weight (kg)", min_value=1.0, value=70.0)
    height_cm = st.number_input("Height (cm)", min_value=50.0, max_value=250.0, value=175.0)
    bmi = st.number_input("BMI", min_value=1.0, max_value=100.0, value=22.5)
    fitness_level = st.selectbox("Fitness Level", ["lightly_active", "fairly_active", "sedentary", "very_active"])
    performance_level = st.selectbox("Performance Level", ["low", "moderate", "high"])
    resting_hr = st.number_input("Resting Heart Rate", min_value=30, max_value=120, value=65)
    max_hr = st.number_input("Max Heart Rate", min_value=100, max_value=220, value=190)
    activity_type = st.selectbox("Activity Type", ["sleeping", "walking", "resting", "light", "commuting", "exercise"])
    activity_intensity = st.slider("Activity Intensity", 0.0, 1.0, 0.5)
    steps_5min = st.number_input("Steps in 5 min", min_value=0, value=200)
    calories_5min = st.number_input("Calories in 5 min", min_value=0.0, value=20.0)
    hrv_rmssd = st.number_input("HRV RMSSD", min_value=0.0, value=35.0)
    stress_score = st.slider("Stress Score", 0, 100, 40)
    signal_quality = st.number_input("Signal Quality", min_value=0.0, value=0.9)
    skin_temperature = st.number_input("Skin Temperature (°C)", min_value=30.0, value=36.5)
    device_battery = st.slider("Device Battery (%)", 0, 100, 85)
    elevation_gain = st.number_input("Elevation Gain", min_value=0, value=5)
    sleep_stage = st.selectbox("Sleep Stage", ["light_sleep", "deep_sleep", "rem_sleep"])
    date = st.date_input("Date", datetime.today())

    if st.button("Predict Heart Rate"):
        payload = {
            "age": age, "gender": gender, "weight_kg": weight_kg, "height_cm": height_cm, "bmi": bmi,
            "fitness_level": fitness_level, "performance_level": performance_level,
            "resting_hr": resting_hr, "max_hr": max_hr, "activity_type": activity_type,
            "activity_intensity": activity_intensity, "steps_5min": steps_5min, "calories_5min": calories_5min,
            "hrv_rmssd": hrv_rmssd, "stress_score": stress_score, "signal_quality": signal_quality,
            "skin_temperature": skin_temperature, "device_battery": device_battery,
            "elevation_gain": elevation_gain, "sleep_stage": sleep_stage, "date": str(date) + "T00:00:00"
        }

        # Convert payload to DataFrame and select features
        input_df = preprocess_heart_input(payload, heart_features)
        
        
        prediction = heart_model.predict(input_df[heart_features])[0]
        st.success(f"Predicted Heart Rate: {prediction:.2f} bpm")

# ======================
# Anomaly Detection
# ======================
if task == "Anomaly Detection":
    st.subheader("Detect Anomaly")

    heart_rate = st.number_input("Heart Rate", min_value=1.0, value=120.0)
    resting_hr_baseline = st.number_input("Resting HR Baseline", min_value=30, max_value=120, value=65)
    activity_type = st.selectbox("Activity Type", ["sleeping", "walking", "resting", "light", "commuting", "exercise"])
    activity_intensity = st.slider("Activity Intensity", 0.0, 1.0, 0.7)
    steps_5min = st.number_input("Steps in 5 min", min_value=0, value=500)
    calories_5min = st.number_input("Calories in 5 min", min_value=0.0, value=50.0)
    hrv_rmssd = st.number_input("HRV RMSSD", min_value=0.0, value=35.0)
    stress_score = st.slider("Stress Score", 0, 100, 40)
    confidence_score = st.number_input("Confidence Score", min_value=0.0, max_value=1.0, value=0.95)
    signal_quality = st.number_input("Signal Quality", min_value=0.0, value=0.95)
    skin_temperature = st.number_input("Skin Temperature (°C)", min_value=30.0, value=36.5)
    device_battery = st.slider("Device Battery (%)", 0, 100, 90)
    elevation_gain = st.number_input("Elevation Gain", min_value=0, value=10)
    sleep_stage = st.selectbox("Sleep Stage", ["light_sleep", "deep_sleep", "rem_sleep"])
    date = st.date_input("Date", datetime.today())

    if st.button("Detect Anomaly"):
        payload = {
            "heart_rate": heart_rate, "resting_hr_baseline": resting_hr_baseline,
            "activity_type": activity_type, "activity_intensity": activity_intensity,
            "steps_5min": steps_5min, "calories_5min": calories_5min, "hrv_rmssd": hrv_rmssd,
            "stress_score": stress_score, "confidence_score": confidence_score,
            "signal_quality": signal_quality, "skin_temperature": skin_temperature,
            "device_battery": device_battery, "elevation_gain": elevation_gain,
            "sleep_stage": sleep_stage, "date": str(date) + "T00:00:00"
        }

        input_df = preprocess_anomaly_input(payload, anomaly_features)
        
        
        anomaly_detected = bool(anomaly_model.predict(input_df[anomaly_features])[0])
        if anomaly_detected:
            st.error("Anomaly Detected!")
        else:
            st.success("Normal Condition")
