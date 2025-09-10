import streamlit as st
import requests
from datetime import datetime


API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Health Monitoring App", layout="centered")
st.title("Health Monitoring Dashboard")

st.sidebar.header("Choose Prediction Task")
task = st.sidebar.radio("Select Task:", ["Heart Rate Prediction", "Anomaly Detection"])

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
            
            "age": age,
            "gender": gender,
            "weight_kg": weight_kg,
            "height_cm": height_cm,
            "bmi": bmi,
            "fitness_level": fitness_level,
            "performance_level": performance_level,
            "resting_hr": resting_hr,
            "max_hr": max_hr,
            "activity_type": activity_type,
            "activity_intensity": activity_intensity,
            "steps_5min": steps_5min,
            "calories_5min": calories_5min,
            "hrv_rmssd": hrv_rmssd,
            "stress_score": stress_score,
            "signal_quality": signal_quality,
            "skin_temperature": skin_temperature,
            "device_battery": device_battery,
            "elevation_gain": elevation_gain,
            "sleep_stage": sleep_stage,
            "date": str(date) + "T00:00:00"
        }

        response = requests.post(f"{API_URL}/predict_heart_rate", json=payload)

        if response.status_code == 200:
            result = response.json()
            st.success(f" Predicted Heart Rate: {result['heart_rate_prediction']:.2f} bpm")
        else:
            st.error("Error: Could not get prediction.")


if task == "Anomaly Detection":
    st.subheader(" Detect Anomaly")

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
            "heart_rate": heart_rate,
            "resting_hr_baseline": resting_hr_baseline,
            "activity_type": activity_type,
            "activity_intensity": activity_intensity,
            "steps_5min": steps_5min,
            "calories_5min": calories_5min,
            "hrv_rmssd": hrv_rmssd,
            "stress_score": stress_score,
            "confidence_score": confidence_score,
            "signal_quality": signal_quality,
            "skin_temperature": skin_temperature,
            "device_battery": device_battery,
            "elevation_gain": elevation_gain,
            "sleep_stage": sleep_stage,
            "date": str(date) + "T00:00:00"
        }

        response = requests.post(f"{API_URL}/detect_anomaly", json=payload)

        if response.status_code == 200:
            result = response.json()
            if result["anomaly_detected"]:
                st.error("Anomaly Detected!")
            else:
                st.success("Normal Condition")
        else:
            st.error("Error: Could not get prediction.")