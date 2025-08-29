import os
import urllib.request as request
import pandas as pd
import numpy as np
from faker import Faker
from datetime import datetime, timedelta
import random
import math
from Heart_Rate_Anomaly_Detector import logger
from Heart_Rate_Anomaly_Detector.utils.common import get_size  
from pathlib import Path
from Heart_Rate_Anomaly_Detector.entity.config_entity import DataGenerationConfig

class DataGeneration:
   def __init__(self, config: DataGenerationConfig):
       self.config = config

   def generate_files(self) -> dict:
       users_path = Path(self.config.local_data_file_users)
       readings_path = Path(self.config.local_data_file_reading)
       
       users_path.parent.mkdir(parents=True, exist_ok=True)
       readings_path.parent.mkdir(parents=True, exist_ok=True)

       if not users_path.exists() or not readings_path.exists():
           try:
               fake = Faker()
               num_users = 50
               days_per_user = 21

               hr_ranges = {
                   'athlete': {'resting': (40, 60), 'active': (120, 160), 'max_factor': 0.95},
                   'high': {'resting': (50, 65), 'active': (130, 170), 'max_factor': 0.92},
                   'moderate': {'resting': (60, 75), 'active': (140, 180), 'max_factor': 0.90},
                   'low': {'resting': (65, 85), 'active': (150, 190), 'max_factor': 0.85}
               }

               activity_patterns = {
                   'sedentary': {'daily_steps': (2000, 5000), 'active_hours': 1},
                   'lightly_active': {'daily_steps': (5000, 7500), 'active_hours': 2},
                   'fairly_active': {'daily_steps': (7500, 10000), 'active_hours': 3},
                   'very_active': {'daily_steps': (10000, 15000), 'active_hours': 4}
               }

               def generate_users():
                   users = []
                   for i in range(num_users):
                       age = int(np.random.beta(2, 5) * 62 + 18)
                       gender = random.choices(['M', 'F'], weights=[0.48, 0.52])[0]
                       
                       height_cm = np.random.normal(175 if gender == 'M' else 162, 8)
                       bmi = np.random.gamma(2, 2) + 20
                       weight_kg = bmi * (height_cm/100) ** 2
                       
                       fitness_weights = [0.35, 0.35, 0.25, 0.05]
                       if age > 60:
                           fitness_weights = [0.45, 0.35, 0.15, 0.05]
                       fitness_level = random.choices(['sedentary', 'lightly_active', 'fairly_active', 'very_active'], 
                                                   weights=fitness_weights)[0]
                       
                       fitness_performance = {
                           'sedentary': 'low',
                           'lightly_active': 'low' if random.random() < 0.7 else 'moderate',
                           'fairly_active': 'moderate' if random.random() < 0.8 else 'high',
                           'very_active': 'high' if random.random() < 0.7 else 'athlete'
                       }
                       performance_level = fitness_performance[fitness_level]
                       
                       if gender == 'F':
                           max_hr = 206 - (0.88 * age)
                       else:
                           max_hr = 214 - (0.8 * age)
                       
                       base_range = hr_ranges[performance_level]['resting']
                       resting_hr = random.randint(base_range[0], base_range[1])
                       
                       if gender == 'F':
                           resting_hr += random.randint(2, 7)
                       if age > 60:
                           resting_hr += random.randint(3, 8)
                       
                       conditions = []
                       condition_probabilities = {
                           'hypertension': 0.15 if age < 45 else 0.35 if age < 65 else 0.60,
                           'diabetes': 0.05 if age < 45 else 0.12 if age < 65 else 0.20,
                           'heart_disease': 0.02 if age < 45 else 0.08 if age < 65 else 0.15,
                           'anxiety': 0.18,
                           'sleep_apnea': 0.05 if bmi < 25 else 0.15 if bmi < 30 else 0.25
                       }
                       
                       for condition, prob in condition_probabilities.items():
                           if random.random() < prob:
                               conditions.append(condition)
                       
                       medications = []
                       if 'hypertension' in conditions:
                           medications.extend(random.sample(['beta_blocker', 'ace_inhibitor', 'calcium_channel_blocker'], 
                                                       random.randint(1, 2)))
                       if 'anxiety' in conditions:
                           if random.random() < 0.4:
                               medications.append('beta_blocker')
                       if 'heart_disease' in conditions:
                           medications.append('beta_blocker')
                       
                       smoker = random.random() < (0.25 if age < 35 else 0.20 if age < 55 else 0.15)
                       caffeine_user = random.random() < 0.85
                       alcohol_user = random.random() < 0.70
                       
                       user = {
                           'user_id': f"USER_{i+1:04d}",
                           'name': fake.name(),
                           'email': fake.email(),
                           'age': age,
                           'gender': gender,
                           'weight_kg': round(weight_kg, 1),
                           'height_cm': round(height_cm, 1),
                           'bmi': round(bmi, 1),
                           'fitness_level': fitness_level,
                           'performance_level': performance_level,
                           'resting_hr': resting_hr,
                           'max_hr': round(max_hr),
                           'conditions': ','.join(conditions) if conditions else 'none',
                           'medications': ','.join(medications) if medications else 'none',
                           'smoker': smoker,
                           'caffeine_user': caffeine_user,
                           'alcohol_user': alcohol_user,
                           'sleep_quality': random.choices(['poor', 'fair', 'good', 'excellent'], 
                                                       weights=[0.15, 0.25, 0.45, 0.15])[0],
                           'registration_date': fake.date_between(start_date='-1y', end_date='today')
                       }
                       users.append(user)
                   
                   return pd.DataFrame(users)

               def get_circadian_multiplier(hour, minute):
                   time_decimal = hour + minute / 60
                   primary_cycle = 0.08 * np.sin(2 * np.pi * (time_decimal - 4) / 24)
                   secondary_cycle = 0.03 * np.sin(4 * np.pi * (time_decimal - 2) / 24)
                   return 1.0 + primary_cycle + secondary_cycle

               def simulate_activity(hour, minute, is_weekend, fitness_level, sleep_start, wake_time):
                   if (hour >= sleep_start or hour < wake_time):
                       return {
                           'type': 'sleeping',
                           'intensity': 0.0,
                           'steps': random.randint(0, 2),
                           'elevation': 0
                       }
                   
                   weekday_activities = {
                       6: {'exercise': 0.15, 'walking': 0.25, 'light': 0.45, 'resting': 0.15},
                       7: {'exercise': 0.12, 'walking': 0.30, 'light': 0.48, 'resting': 0.10},
                       8: {'exercise': 0.08, 'walking': 0.20, 'light': 0.35, 'commuting': 0.25, 'resting': 0.12},
                       9: {'light': 0.60, 'resting': 0.35, 'walking': 0.05},
                       12: {'walking': 0.35, 'light': 0.45, 'resting': 0.20},
                       17: {'exercise': 0.20, 'walking': 0.30, 'commuting': 0.25, 'light': 0.20, 'resting': 0.05},
                       18: {'exercise': 0.25, 'walking': 0.25, 'light': 0.35, 'resting': 0.15},
                       19: {'walking': 0.20, 'light': 0.50, 'resting': 0.30}
                   }
                   
                   weekend_activities = {
                       8: {'exercise': 0.20, 'walking': 0.30, 'light': 0.35, 'resting': 0.15},
                       10: {'exercise': 0.25, 'walking': 0.35, 'light': 0.30, 'resting': 0.10},
                       14: {'exercise': 0.15, 'walking': 0.40, 'light': 0.35, 'resting': 0.10},
                       16: {'exercise': 0.20, 'walking': 0.35, 'light': 0.35, 'resting': 0.10}
                   }
                   
                   activity_probs = (weekend_activities if is_weekend else weekday_activities).get(
                       hour, {'light': 0.50, 'resting': 0.45, 'walking': 0.05}
                   )
                   
                   if fitness_level in ['fairly_active', 'very_active']:
                       if 'exercise' in activity_probs:
                           activity_probs['exercise'] *= 1.5
                       activity_probs['walking'] *= 1.3
                   
                   activity_type = random.choices(list(activity_probs.keys()), 
                                               weights=list(activity_probs.values()))[0]
                   
                   activity_mapping = {
                       'sleeping': {'intensity': 0.0, 'steps': (0, 2), 'elevation': 0},
                       'resting': {'intensity': 0.1, 'steps': (0, 5), 'elevation': 0},
                       'light': {'intensity': 0.3, 'steps': (8, 25), 'elevation': 0},
                       'walking': {'intensity': 0.5, 'steps': (40, 80), 'elevation': random.randint(0, 3)},
                       'commuting': {'intensity': 0.4, 'steps': (20, 40), 'elevation': random.randint(0, 2)},
                       'exercise': {'intensity': random.uniform(0.7, 0.9), 'steps': (60, 120), 
                                   'elevation': random.randint(0, 8)}
                   }
                   
                   activity_params = activity_mapping[activity_type]
                   
                   return {
                       'type': activity_type,
                       'intensity': activity_params['intensity'],
                       'steps': random.randint(*activity_params['steps']),
                       'elevation': activity_params['elevation']
                   }

               def calculate_hr_impact(intensity, resting_hr, max_hr, performance_level):
                   if intensity == 0:
                       return 0
                   
                   hr_reserve = max_hr - resting_hr
                   target_hr_increase = intensity * hr_reserve
                   
                   efficiency_factors = {
                       'athlete': 0.85, 'high': 0.90, 'moderate': 1.0, 'low': 1.15
                   }
                   
                   return target_hr_increase * efficiency_factors[performance_level]

               def apply_modifiers(hr, user_profile, hour, activity_data):
                   if user_profile['caffeine_user'] and 7 <= hour <= 11:
                       hr += random.uniform(3, 8)
                   
                   if 9 <= hour <= 17 and activity_data['type'] != 'exercise':
                       stress_factor = random.uniform(1.0, 1.08)
                       hr *= stress_factor
                   
                   if activity_data['intensity'] > 0.6:
                       hr += random.uniform(2, 5)
                   
                   if hour > 14 and random.random() < 0.1:
                       hr += random.uniform(3, 7)
                   
                   return hr

               def apply_conditions(hr, conditions, medications, hour):
                   if 'hypertension' in conditions:
                       hr += random.uniform(2, 6)
                   
                   if 'diabetes' in conditions:
                       if random.random() < 0.15:
                           hr += random.uniform(-3, 8)
                   
                   if 'anxiety' in conditions:
                       if random.random() < 0.05:
                           hr += random.uniform(15, 30)
                   
                   if 'sleep_apnea' in conditions and (22 <= hour or hour <= 6):
                       hr += random.uniform(-5, 10)
                   
                   if 'beta_blocker' in medications:
                       hr *= random.uniform(0.85, 0.92)
                   
                   return hr

               def calculate_hrv(current_hr, resting_hr, activity_intensity):
                   base_hrv = random.uniform(25, 65)
                   hr_factor = max(0.3, 1 - (current_hr - resting_hr) / resting_hr)
                   activity_factor = max(0.4, 1 - activity_intensity)
                   return base_hrv * hr_factor * activity_factor

               def calculate_stress(current_hr, resting_hr, hrv):
                   hr_stress = min(50, (current_hr - resting_hr) / resting_hr * 100)
                   hrv_stress = max(0, 50 - hrv)
                   total_stress = (hr_stress + hrv_stress) * 0.6 + random.uniform(0, 20)
                   return min(100, max(0, round(total_stress)))

               def detect_anomalies(hr, resting_hr, max_hr, age, conditions, activity_data, hour):
                   is_anomaly = False
                   anomaly_type = None
                   severity = None
                   
                   base_anomaly_rate = 0.005
                   
                   if age > 65:
                       base_anomaly_rate *= 2
                   if 'heart_disease' in conditions:
                       base_anomaly_rate *= 3
                   if 'anxiety' in conditions:
                       base_anomaly_rate *= 1.5
                   
                   if random.random() < base_anomaly_rate:
                       is_anomaly = True
                       
                       if activity_data['intensity'] < 0.2:
                           if hr > resting_hr + 40:
                               anomaly_type = 'resting_tachycardia'
                               severity = 'moderate' if hr < resting_hr + 60 else 'high'
                           elif hr < 45:
                               anomaly_type = 'bradycardia'
                               severity = 'moderate' if hr > 35 else 'high'
                           elif random.random() < 0.3:
                               anomaly_type = 'irregular_rhythm'
                               severity = 'low'
                       else:
                           if hr > max_hr * 0.95:
                               anomaly_type = 'exercise_induced_tachycardia'
                               severity = 'high'
                           elif hr < resting_hr + 10:
                               anomaly_type = 'chronotropic_incompetence'
                               severity = 'moderate'
                   
                   if (22 <= hour or hour <= 6) and 'sleep_apnea' in conditions:
                       if random.random() < 0.02:
                           is_anomaly = True
                           anomaly_type = 'sleep_related_bradycardia'
                           severity = 'low'
                   
                   return {'is_anomaly': is_anomaly, 'type': anomaly_type, 'severity': severity}

               def calculate_calories(hr, user_profile, activity_data):
                   age = user_profile['age']
                   weight = user_profile['weight_kg']
                   gender = user_profile['gender']
                   
                   if gender == 'M':
                       calories_per_min = ((-95.7735 + (0.634 * hr) + (0.404 * weight) + 
                                       (0.394 * age) - (0.271 * age)) / 4.184) / 60
                   else:
                       calories_per_min = ((-20.4022 + (0.4472 * hr) - (0.1263 * weight) + 
                                       (0.074 * age) - (0.05741 * age)) / 4.184) / 60
                   
                   calories_5min = max(1.0, calories_per_min * 5)
                   return round(calories_5min, 2)

               def simulate_device(minute):
                   minutes_per_day = 1440
                   battery_drain_rate = random.uniform(0.8, 1.2) / 100
                   battery_level = 100 - (minute / minutes_per_day) * battery_drain_rate * 100
                   
                   base_signal_quality = random.uniform(0.85, 1.0)
                   confidence = base_signal_quality * random.uniform(0.9, 1.0)
                   
                   return {
                       'battery': max(5, round(battery_level)),
                       'signal_quality': round(base_signal_quality, 3),
                       'confidence': round(confidence, 3)
                   }

               def determine_sleep_stage(hour, wake_time, sleep_start, current_hr, resting_hr):
                   if not (hour >= sleep_start or hour < wake_time):
                       return None
                   
                   hr_ratio = current_hr / resting_hr
                   
                   if hr_ratio < 0.90:
                       return 'deep_sleep'
                   elif hr_ratio < 0.95:
                       return 'light_sleep'
                   elif hr_ratio > 1.05:
                       return 'rem_sleep'
                   else:
                       return 'light_sleep'

               def generate_day_pattern(user_profile, date):
                   data_points = []
                   
                   resting_hr = user_profile['resting_hr']
                   max_hr = user_profile['max_hr']
                   fitness_level = user_profile['fitness_level']
                   performance = user_profile['performance_level']
                   age = user_profile['age']
                   conditions = user_profile['conditions'].split(',') if user_profile['conditions'] != 'none' else []
                   medications = user_profile['medications'].split(',') if user_profile['medications'] != 'none' else []
                   
                   is_weekend = date.weekday() >= 5
                   
                   for minute in range(0, 1440, 5):
                       hour = minute // 60
                       min_in_hour = minute % 60
                       timestamp = datetime.combine(date, datetime.min.time()) + timedelta(minutes=minute)
                       
                       circadian_multiplier = get_circadian_multiplier(hour, min_in_hour)
                       
                       sleep_start = 22 + random.uniform(-1, 2)
                       wake_time = 6.5 + random.uniform(-1, 1.5)
                       
                       if is_weekend:
                           sleep_start += 1
                           wake_time += 1
                       
                       activity_data = simulate_activity(hour, min_in_hour, is_weekend, 
                                                       fitness_level, sleep_start, wake_time)
                       
                       base_hr = resting_hr * circadian_multiplier
                       
                       activity_hr_increase = calculate_hr_impact(
                           activity_data['intensity'], resting_hr, max_hr, performance
                       )
                       
                       current_hr = base_hr + activity_hr_increase
                       
                       current_hr = apply_modifiers(current_hr, user_profile, hour, activity_data)
                       current_hr = apply_conditions(current_hr, conditions, medications, hour)
                       current_hr += np.random.normal(0, 2)
                       
                       hrv_rmssd = calculate_hrv(current_hr, resting_hr, activity_data['intensity'])
                       current_hr = max(35, min(current_hr, max_hr * 0.98))
                       
                       anomaly_info = detect_anomalies(
                           current_hr, resting_hr, max_hr, age, conditions, activity_data, hour
                       )
                       
                       device_info = simulate_device(minute)
                       
                       data_point = {
                           'user_id': user_profile['user_id'],
                           'timestamp': timestamp,
                           'heart_rate': round(current_hr, 1),
                           'resting_hr_baseline': resting_hr,
                           'activity_type': activity_data['type'],
                           'activity_intensity': activity_data['intensity'],
                           'steps_5min': activity_data['steps'],
                           'calories_5min': calculate_calories(current_hr, user_profile, activity_data),
                           'hrv_rmssd': round(hrv_rmssd, 1),
                           'stress_score': calculate_stress(current_hr, resting_hr, hrv_rmssd),
                           'is_anomaly': anomaly_info['is_anomaly'],
                           'anomaly_type': anomaly_info['type'],
                           'anomaly_severity': anomaly_info['severity'],
                           'confidence_score': device_info['confidence'],
                           'signal_quality': device_info['signal_quality'],
                           'skin_temperature': round(36.1 + np.random.normal(0, 0.3), 1),
                           'device_battery': device_info['battery'],
                           'elevation_gain': activity_data.get('elevation', 0),
                           'sleep_stage': determine_sleep_stage(hour, wake_time, sleep_start, current_hr, resting_hr)
                       }
                       
                       data_points.append(data_point)
                   
                   return data_points

               def generate_dataset():
                   users_df = generate_users()
                   all_hr_data = []
                   
                   for idx, user in users_df.iterrows():
                       start_date = user['registration_date']
                       for day in range(days_per_user):
                           current_date = start_date + timedelta(days=day)
                           daily_data = generate_day_pattern(user, current_date)
                           all_hr_data.extend(daily_data)
                   
                   hr_df = pd.DataFrame(all_hr_data)
                   return users_df, hr_df

               users_df, hr_df = generate_dataset()
               
               users_df.to_csv(users_path, index=False)
               hr_df.to_csv(readings_path, index=False)

               logger.info(f"Generated dataset with {len(users_df)} users and {len(hr_df)} heart rate readings.")
               
               return {'users': str(users_path), 'readings': str(readings_path)}
           except Exception as e:
               logger.error(f"Failed to generate files: {e}")
               raise
       else:
           logger.info(f"Files already exist: {users_path}, {readings_path}")
           return {'users': str(users_path), 'readings': str(readings_path)}