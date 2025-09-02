import pandas as pd
import os
import numpy as np
from pathlib import Path
from Heart_Rate_Anomaly_Detector import logger
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
from pathlib import Path
from Heart_Rate_Anomaly_Detector.entity.config_entity import DataTransformationConfig
import warnings
warnings.filterwarnings("ignore")


class DataTransformation:
    def __init__(self, config):
        self.config = config
        self.scaler = None
        self.imputers = {}
        self.label_encoders = {}
        
    def load_and_preprocess_data(self):
       
        logger.info("Loading and preprocessing data...")
        
        df1 = pd.read_csv(self.config.data_path_reading)
        df2 = pd.read_csv(self.config.data_path_users)
        
        df = pd.merge(df1, df2, on='user_id')
        logger.info(f"Loaded data shape: {df.shape}")
        
        df.rename(columns={'timestamp': 'date'}, inplace=True)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        
        
        df = df.infer_objects()
        
        return df
    
    def handle_missing_values(self, df):
        
        df = df.copy()
        logger.info("Handling missing values...")      
        missing_before = df.isnull().sum().sum()
        logger.info(f"Missing values before processing: {missing_before}")
        
       
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
     
        if 'date' in categorical_cols:
            categorical_cols.remove('date')
              
        if numeric_cols:
            df_with_date = df.set_index('date') if 'date' in df.columns else df
            df_with_date[numeric_cols] = df_with_date[numeric_cols].interpolate(method='time')
            df = df_with_date.reset_index() if 'date' in df_with_date.index.names else df_with_date
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        
        
        if categorical_cols:
            for col in categorical_cols:
                df[col] = df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown')
        
        missing_after = df.isnull().sum().sum()
        logger.info(f"Missing values after processing: {missing_after}")
        
        return df
    
    def create_advanced_features(self, df):
        
        df = df.copy()
        logger.info("Creating advanced features...")
        
        
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            
            
            if 'month' not in df.columns:
                df['month'] = df['date'].dt.month
            if 'day' not in df.columns:
                df['day'] = df['date'].dt.day
            if 'year' not in df.columns:
                df['year'] = df['date'].dt.year
            if 'day_of_week' not in df.columns:
                df['day_of_week'] = df['date'].dt.dayofweek
            if 'hour' not in df.columns and df['date'].dt.hour.nunique() > 1:
                df['hour'] = df['date'].dt.hour
            if 'is_weekend' not in df.columns:
                df['is_weekend'] = (df['date'].dt.dayofweek >= 5).astype(int)
            if 'day_of_year' not in df.columns:
                df['day_of_year'] = df['date'].dt.dayofyear
            
            
            df['quarter'] = df['date'].dt.quarter
            df['week_of_year'] = df['date'].dt.isocalendar().week
            
            
            df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
            df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
            df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
            df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
            
            if 'hour' in df.columns:
                df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
                df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
            
           
            df['is_winter'] = df['month'].isin([12, 1, 2]).astype(int)
            df['is_summer'] = df['month'].isin([6, 7, 8]).astype(int)
            df['is_spring'] = df['month'].isin([3, 4, 5]).astype(int)
            df['is_autumn'] = df['month'].isin([9, 10, 11]).astype(int)

            if 'hrv_rmssd' in df.columns and 'stress_score' in df.columns:
                df['hrv_stress_ratio'] = df['hrv_rmssd'] / (df['stress_score'] + 1e-8)
                df['hrv_stress_interaction'] = df['hrv_rmssd'] * df['stress_score']
            
            
            if 'sleep_stage' in df.columns:
                df['is_deep_sleep'] = (df['sleep_stage'] == 'deep').astype(int)
                df['is_rem_sleep'] = (df['sleep_stage'] == 'rem').astype(int)
                df['is_light_sleep'] = (df['sleep_stage'] == 'light').astype(int)
                df['is_awake'] = (df['sleep_stage'] == 'awake').astype(int)
            
        
            if 'fitness_level' in df.columns:
                fitness_map = {'beginner': 1, 'intermediate': 2, 'advanced': 3, 'elite': 4}
                df['fitness_numeric'] = df['fitness_level'].str.lower().map(fitness_map).fillna(1)
                
                
                if 'intensity_numeric' in df.columns:
                    df['fitness_intensity_ratio'] = df['fitness_numeric'] / (df['intensity_numeric'] + 1e-8)
            
        
            if 'performance_level' in df.columns:
                perf_map = {'poor': 1, 'below_average': 2, 'average': 3, 'above_average': 4, 'excellent': 5}
                df['performance_numeric'] = df['performance_level'].str.lower().map(perf_map).fillna(3)
            
            
            if 'bmi' in df.columns:
                df['bmi'] = pd.to_numeric(df['bmi'], errors='coerce')
                df['bmi_underweight'] = (df['bmi'] < 18.5).astype(int)
                df['bmi_normal'] = ((df['bmi'] >= 18.5) & (df['bmi'] < 25)).astype(int)
                df['bmi_overweight'] = ((df['bmi'] >= 25) & (df['bmi'] < 30)).astype(int)
                df['bmi_obese'] = (df['bmi'] >= 30).astype(int)
        
            if 'signal_quality' in df.columns:
                df['poor_signal'] = (df['signal_quality'] <= 0.5).astype(int)
                df['excellent_signal'] = (df['signal_quality'] >= 0.9).astype(int)
            
            if 'device_battery' in df.columns:
                df['low_battery'] = (df['device_battery'] <= 20).astype(int)
            
            
            if 'is_anomaly' in df.columns:
                df['anomaly_binary'] = df['is_anomaly'].astype(int)
                
            if 'anomaly_severity' in df.columns:
                df['anomaly_severity'] = pd.to_numeric(df['anomaly_severity'], errors='coerce')
                df['high_severity_anomaly'] = (df['anomaly_severity'] >= 0.8).astype(int)
        
        
                
        logger.info(f"Features created. New shape: {df.shape}")
        return df
  
    


    def create_lag_features(self, df, target_col=None, lags=[1, 2, 5]):

        df = df.copy()
        target_col = target_col or getattr(self.config, 'target_column', 'heart_rate')

        logger.info(f"Creating Lag Features for target: {target_col} ...")
        df = df.sort_values(['user_id', 'date'])
        
        for lag in lags:
            if target_col in df.columns:
                df[f'{target_col}_lag_{lag}'] = df.groupby('user_id')[target_col].shift(lag)
                df[f'{target_col}_diff_{lag}'] = df[target_col] - df[f'{target_col}_lag_{lag}']
                df[target_col] = df[target_col].astype(float)
                df[f'{target_col}_pct_change_{lag}'] = (df.groupby('user_id')[target_col].pct_change(lag).replace([np.inf, -np.inf], 0).fillna(0))



            if 'hrv_rmssd' in df.columns:
                df[f'hrv_rmssd_lag_{lag}'] = df.groupby('user_id')['hrv_rmssd'].shift(lag)
            if 'stress_score' in df.columns:
                df[f'stress_score_lag_{lag}'] = df.groupby('user_id')['stress_score'].shift(lag)
            if 'steps_5min' in df.columns:
                df[f'steps_5min_lag_{lag}'] = df.groupby('user_id')['steps_5min'].shift(lag)

        logger.info(f"Lag Features created. New shape: {df.shape}")
        return df

    def create_rolling_features(self, df, target_col=None, windows=[5, 10, 30]):
        df = df.copy()
        target_col = target_col or getattr(self.config, 'target_column', 'heart_rate')

        logger.info(f"Creating Rolling Features for target: {target_col} ...")
        df = df.sort_values(['user_id', 'date'])

        metrics = [target_col, 'hrv_rmssd', 'stress_score', 'steps_5min', 'calories_5min', 'skin_temperature']
        metrics = [m for m in metrics if m in df.columns]

        for metric in metrics:
            for window in windows:
                df[f'{metric}_rolling_mean_{window}'] = df.groupby('user_id')[metric].transform(
                    lambda x: x.rolling(window=window, min_periods=1).mean()
                )
                df[f'{metric}_rolling_std_{window}'] = df.groupby('user_id')[metric].transform(
                    lambda x: x.rolling(window=window, min_periods=1).std()
                )
                df[f'{metric}_rolling_min_{window}'] = df.groupby('user_id')[metric].transform(
                    lambda x: x.rolling(window=window, min_periods=1).min()
                )
                df[f'{metric}_rolling_max_{window}'] = df.groupby('user_id')[metric].transform(
                    lambda x: x.rolling(window=window, min_periods=1).max()
                )
                df[f'{metric}_rolling_median_{window}'] = df.groupby('user_id')[metric].transform(
                    lambda x: x.rolling(window=window, min_periods=1).median()
                )

                mean_vals = df[f'{metric}_rolling_mean_{window}']
                median_vals = df[f'{metric}_rolling_median_{window}']
                std_vals = df[f'{metric}_rolling_std_{window}']

                df[f'{metric}_dist_from_mean_{window}'] = df[metric] - mean_vals
                df[f'{metric}_dist_from_median_{window}'] = df[metric] - median_vals
                df[f'{metric}_zscore_{window}'] = (df[metric] - mean_vals) / (std_vals + 1e-8)

        logger.info(f"Rolling Features created. New shape: {df.shape}")
        return df


    def create_user_specific_features(self, df, target_col=None):
        df = df.copy()
        target_col = target_col or getattr(self.config, 'target_column', 'heart_rate')

        logger.info(f"Creating User Specific Features for target: {target_col} ...")

        metrics_for_baseline = [target_col, 'hrv_rmssd', 'stress_score', 'steps_5min', 'calories_5min', 'skin_temperature']
        metrics_for_baseline = [m for m in metrics_for_baseline if m in df.columns]
        
        for metric in metrics_for_baseline:
            user_stats = df.groupby('user_id')[metric].agg([
                'mean', 'std', 'min', 'max', 'median'
            ]).reset_index()
            user_stats.columns = ['user_id'] + [f'user_{metric}_{col}' for col in user_stats.columns[1:]]
            
            
            df = df.merge(user_stats, on='user_id', how='left')
            
            
            df[f'{metric}_above_baseline'] = df[metric] - df[f'user_{metric}_mean']
            df[f'{metric}_zscore_user'] = df[f'{metric}_above_baseline'] / (df[f'user_{metric}_std'] + 1e-8)
            
            
            df[f'{metric}_percentile_user'] = df.groupby('user_id')[metric].rank(pct=True)
        
      
        if 'max_hr' in df.columns:
            df['max_hr_to_use'] = df['max_hr'].fillna(220 - df['age'])
        else:
            df['max_hr_to_use'] = 220 - df['age']
        
        
        df['hr_zone_1'] = (df[target_col] <= 0.6 * df['max_hr_to_use']).astype(int)  # Recovery
        df['hr_zone_2'] = ((df[target_col] > 0.6 * df['max_hr_to_use']) & 
                          (df[target_col] <= 0.7 * df['max_hr_to_use'])).astype(int)  # Aerobic base
        df['hr_zone_3'] = ((df[target_col] > 0.7 * df['max_hr_to_use']) & 
                          (df[target_col] <= 0.8 * df['max_hr_to_use'])).astype(int)  # Aerobic
        df['hr_zone_4'] = ((df[target_col] > 0.8 * df['max_hr_to_use']) & 
                          (df[target_col] <= 0.9 * df['max_hr_to_use'])).astype(int)  # Threshold
        df['hr_zone_5'] = (df[target_col] > 0.9 * df['max_hr_to_use']).astype(int)  # Anaerobic
        
     
        if 'resting_hr' in df.columns:
            df['hr_reserve'] = df['max_hr_to_use'] - df['resting_hr']
            df['hr_reserve_pct'] = (df[target_col] - df['resting_hr']) / (df['hr_reserve'] + 1e-8)
        
        df['hr_pct_max'] = df[target_col] / df['max_hr_to_use']
        
        
        if 'resting_hr_baseline' in df.columns:
            df['hr_above_resting_baseline'] = df[target_col] - df['resting_hr_baseline']
            df['hr_ratio_to_baseline'] = df[target_col] / (df['resting_hr_baseline'] + 1e-8)

        logger.info(f"User Specific Features created. New shape: {df.shape}")
        return df

    def encode_medical_features(self, df):

        df = df.copy()

        logger.info("Creating Medical Features...")
        
        
        if 'conditions' in df.columns:
            
            df['conditions_clean'] = df['conditions'].fillna('none').str.lower()
            
            
            df['has_diabetes'] = df['conditions_clean'].str.contains('diabetes', case=False, na=False).astype(int)
            df['has_hypertension'] = df['conditions_clean'].str.contains('hypertension|high blood pressure', case=False, na=False).astype(int)
            df['has_heart_disease'] = df['conditions_clean'].str.contains('heart disease|cardiac|coronary', case=False, na=False).astype(int)
            df['has_asthma'] = df['conditions_clean'].str.contains('asthma', case=False, na=False).astype(int)
            df['has_arrhythmia'] = df['conditions_clean'].str.contains('arrhythmia|irregular', case=False, na=False).astype(int)
            df['has_thyroid'] = df['conditions_clean'].str.contains('thyroid|hyperthyroid|hypothyroid', case=False, na=False).astype(int)
            
           
            condition_cols = ['has_diabetes', 'has_hypertension', 'has_heart_disease', 
                             'has_asthma', 'has_arrhythmia', 'has_thyroid']
            df['num_medical_conditions'] = df[condition_cols].sum(axis=1)
        
       
        if 'medications' in df.columns:
            df['medications_clean'] = df['medications'].fillna('none').str.lower()
            
            
            df['takes_beta_blockers'] = df['medications_clean'].str.contains(
                'beta blocker|metoprolol|propranolol|atenolol|carvedilol', case=False, na=False
            ).astype(int)
            
           
            df['takes_ace_inhibitors'] = df['medications_clean'].str.contains(
                'ace inhibitor|lisinopril|enalapril|captopril', case=False, na=False
            ).astype(int)
            
            
            df['takes_ccb'] = df['medications_clean'].str.contains(
                'amlodipine|diltiazem|verapamil|nifedipine', case=False, na=False
            ).astype(int)
            
           
            df['takes_stimulants'] = df['medications_clean'].str.contains(
                'adderall|ritalin|stimulant', case=False, na=False
            ).astype(int)
            
            
            med_cols = ['takes_beta_blockers', 'takes_ace_inhibitors', 'takes_ccb', 'takes_stimulants']
            df['num_hr_affecting_medications'] = df[med_cols].sum(axis=1)
        
        
        lifestyle_factors = ['smoker', 'caffeine_user', 'alcohol_user']
        for factor in lifestyle_factors:
            if factor in df.columns:
                df[f'{factor}_binary'] = (df[factor] == True).astype(int)
        
        
        if 'sleep_quality' in df.columns:
            df['sleep_quality'] = pd.to_numeric(df['sleep_quality'], errors='coerce')
            df['poor_sleep'] = (df['sleep_quality'] <= 2).astype(int)
            df['excellent_sleep'] = (df['sleep_quality'] >= 4).astype(int)
        
        logger.info(f"Medical Features created. New shape: {df.shape}")
        
        return df
    
    def encode_activity_features(self, df):

        df = df.copy()

        logger.info("Creating Activity Features...")
        
        
        if 'activity_type' in df.columns:
            
            df['activity_type_clean'] = df['activity_type'].fillna('unknown').astype(str).str.lower()

            
            if 'activity_type' not in self.label_encoders:
                self.label_encoders['activity_type'] = LabelEncoder()
                df['activity_type_encoded'] = self.label_encoders['activity_type'].fit_transform(
                    df['activity_type_clean']
                )
            else:
                df['activity_type_encoded'] = self.label_encoders['activity_type'].transform(
                    df['activity_type_clean']
                )
            
            
            df['is_running'] = df['activity_type_clean'].str.contains('run|jog', case=False, na=False).astype(int)
            df['is_walking'] = df['activity_type_clean'].str.contains('walk', case=False, na=False).astype(int)
            df['is_cycling'] = df['activity_type_clean'].str.contains('cycl|bike', case=False, na=False).astype(int)
            df['is_strength'] = df['activity_type_clean'].str.contains('strength|weight|lift|gym', case=False, na=False).astype(int)
            df['is_cardio'] = df['activity_type_clean'].str.contains('cardio|aerobic', case=False, na=False).astype(int)
            df['is_resting'] = df['activity_type_clean'].str.contains('rest|sleep|sitting', case=False, na=False).astype(int)
            df['is_swimming'] = df['activity_type_clean'].str.contains('swim', case=False, na=False).astype(int)
            df['is_yoga'] = df['activity_type_clean'].str.contains('yoga|pilates', case=False, na=False).astype(int)
        
        
        if 'activity_intensity' in df.columns:
            
            df['activity_intensity_clean'] = df['activity_intensity'].fillna('unknown').astype(str).str.lower()
           
            
            intensity_map = {
                'rest': 0, 'resting': 0,
                'low': 1, 'light': 1, 'easy': 1,
                'moderate': 2, 'medium': 2,
                'high': 3, 'vigorous': 3, 'hard': 3,
                'very_high': 4, 'maximum': 4, 'max': 4, 'very high': 4
            }
            
            df['intensity_numeric'] = df['activity_intensity_clean'].map(intensity_map).fillna(0)
            
         
            df['is_rest'] = (df['intensity_numeric'] == 0).astype(int)
            df['is_low_intensity'] = (df['intensity_numeric'] == 1).astype(int)
            df['is_moderate_intensity'] = (df['intensity_numeric'] == 2).astype(int)
            df['is_high_intensity'] = (df['intensity_numeric'] >= 3).astype(int)
        
        
        if 'steps_5min' in df.columns:
            df['steps_5min_log'] = np.log1p(df['steps_5min'].fillna(0))
            df['is_high_steps'] = (df['steps_5min'] > df['steps_5min'].quantile(0.8)).astype(int)
            df['is_sedentary'] = (df['steps_5min'] <= 10).astype(int)
        
        if 'calories_5min' in df.columns:
            df['calories_5min_log'] = np.log1p(df['calories_5min'].fillna(0))
            df['is_high_calorie_burn'] = (df['calories_5min'] > df['calories_5min'].quantile(0.8)).astype(int)
        
        # Elevation impact
        if 'elevation_gain' in df.columns:
            df['elevation_gain_log'] = np.log1p(df['elevation_gain'].fillna(0) + 1)
            df['is_high_elevation'] = (df['elevation_gain'] > 100).astype(int)
        
        logger.info(f"Activity Features created. New shape: {df.shape}")
        
        return df
    
        
    def train_test_splitting(self):
        logger.info("Starting comprehensive data transformation pipeline...")

        df = self.load_and_preprocess_data()
        df = self.handle_missing_values(df)
        df = self.create_advanced_features(df)

        target_col = getattr(self.config, 'target_column', 'heart_rate')
        df = self.create_lag_features(df, target_col=target_col)
        df = self.create_rolling_features(df, target_col=target_col)
        df = self.create_user_specific_features(df, target_col=target_col)
        df = self.encode_medical_features(df)
        df = self.encode_activity_features(df)

        
        
        logger.info("Performing train-test split...")
        
       
        df = df.sort_values('date') if 'date' in df.columns else df
        
        
        split_ratio = getattr(self.config, 'train_split_ratio', 0.75)
        split_index = int(len(df) * split_ratio)
        
        train = df.iloc[:split_index].copy()
        test = df.iloc[split_index:].copy()
        
        
        if 'date' in train.columns:
            train = train.reset_index(drop=True)
            test = test.reset_index(drop=True)
        
       
        os.makedirs(self.config.root_dir, exist_ok=True)
        train.to_csv(os.path.join(self.config.root_dir, f"train_{target_col}.csv"), index=False)
        test.to_csv(os.path.join(self.config.root_dir, f"test_{target_col}.csv"), index=False)

        
        
        if self.scaler is not None:
            scaler_path = os.path.join(self.config.root_dir, "scaler.joblib")
            joblib.dump(self.scaler, scaler_path)
            logger.info(f"Scaler saved to: {scaler_path}")

        
        if self.label_encoders:
            encoders_path = os.path.join(self.config.root_dir, "label_encoders.joblib")
            joblib.dump(self.label_encoders, encoders_path)
            logger.info(f"Label encoders saved to: {encoders_path}")

        
        
   
           
        logger.info("Data transformation pipeline completed successfully!")
        
       
        return train, test