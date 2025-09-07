import pandas as pd
import os
import numpy as np
from datetime import datetime
import joblib
from Heart_Rate_Anomaly_Detector import logger
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from pathlib import Path
from Heart_Rate_Anomaly_Detector.entity.config_entity import ModelTrainerConfig
import warnings
warnings.filterwarnings("ignore")


class ModelTrainer:
    def __init__(self, config, schema_config, params, model_name):
        self.config = config
        self.schema_config = schema_config
        self.params = params
        self.model_name = model_name

        self.scaler = None
        self.feature_columns = None
        self.encoder = None

        self.target_column = schema_config.models[model_name].target_column
        self.features = schema_config.models[model_name].features

    def load_data(self):
        data_paths = {
            "HeartRatePredictor": (
                self.config.train_heart_rate_data_path,
                self.config.test_heart_rate_data_path
            ),
            "AnomalyDetector": (
                self.config.train_is_anomaly_data_path,
                self.config.test_is_anomaly_data_path
            )
        }

        if self.model_name not in data_paths:
            raise ValueError(f"Unknown model name: {self.model_name}")

        train_path, test_path = data_paths[self.model_name]
        train_data = pd.read_csv(train_path)
        test_data = pd.read_csv(test_path)

        logger.info(f"Data loaded for {self.model_name} - Train: {train_data.shape}, Test: {test_data.shape}")
        return train_data, test_data

    def _encode_categorical_features(self, train_x, test_x, train_y):
        cat_cols = train_x.select_dtypes(include=['object']).columns.tolist()

        exclude_cols = ['user_id', 'email', 'name']
        id_cols_to_drop = [col for col in exclude_cols if col in train_x.columns]
        if id_cols_to_drop:
            logger.info(f"Dropping ID columns: {id_cols_to_drop}")
            train_x = train_x.drop(columns=id_cols_to_drop)
            test_x = test_x.drop(columns=id_cols_to_drop)
            cat_cols = [col for col in cat_cols if col not in id_cols_to_drop]

        if not cat_cols:
            return train_x, test_x

        logger.info(f"Processing categorical columns: {cat_cols}")
        cardinality_info = {col: train_x[col].nunique() for col in cat_cols}
        for col, count in cardinality_info.items():
            logger.info(f"Column '{col}': {count} unique values")

        high_card_cols = [col for col, count in cardinality_info.items() if count > 20]
        low_card_cols = [col for col, count in cardinality_info.items() if count <= 20]

        if high_card_cols:
            train_x, test_x = self._apply_target_encoding(train_x, test_x, train_y, high_card_cols)

        if low_card_cols:
            train_x, test_x = self._apply_onehot_encoding(train_x, test_x, low_card_cols)

        return train_x, test_x

    def _apply_target_encoding(self, train_x, test_x, train_y, cols):
        logger.info(f"Applying target encoding to: {cols}")
        target_mean = train_y.mean()

        for col in cols:
            temp_df = pd.DataFrame({"category": train_x[col], "target": train_y.values})
            encoding_map = temp_df.groupby("category")["target"].mean().to_dict()

            train_x[f"{col}_encoded"] = train_x[col].map(encoding_map).fillna(target_mean)
            test_x[f"{col}_encoded"] = test_x[col].map(encoding_map).fillna(target_mean)

            train_x = train_x.drop(columns=[col])
            test_x = test_x.drop(columns=[col])

        return train_x, test_x

    def _apply_onehot_encoding(self, train_x, test_x, cols):
        logger.info(f"Applying one-hot encoding to: {cols}")
        expected_features = sum(train_x[col].nunique() - 1 for col in cols)

        if expected_features > 500:
            logger.warning(f"Too many expected features ({expected_features}), switching to frequency encoding")
            return self._apply_frequency_encoding(train_x, test_x, cols)

        try:
            self.encoder = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
            train_encoded = self.encoder.fit_transform(train_x[cols])
            test_encoded = self.encoder.transform(test_x[cols])

            feature_names = self.encoder.get_feature_names_out(cols)
            train_encoded_df = pd.DataFrame(train_encoded, columns=feature_names, index=train_x.index)
            test_encoded_df = pd.DataFrame(test_encoded, columns=feature_names, index=test_x.index)

            remaining_cols = [col for col in train_x.columns if col not in cols]
            train_x = pd.concat([train_x[remaining_cols], train_encoded_df], axis=1)
            test_x = pd.concat([test_x[remaining_cols], test_encoded_df], axis=1)

        except MemoryError:
            logger.error("Memory error in one-hot encoding, falling back to frequency encoding")
            return self._apply_frequency_encoding(train_x, test_x, cols)

        return train_x, test_x

    def _apply_frequency_encoding(self, train_x, test_x, cols):
        logger.info(f"Applying frequency encoding to: {cols}")
        for col in cols:
            freq_map = train_x[col].value_counts(normalize=True).to_dict()
            train_x[f"{col}_freq"] = train_x[col].map(freq_map)
            test_x[f"{col}_freq"] = test_x[col].map(freq_map).fillna(0)
            train_x = train_x.drop(columns=[col])
            test_x = test_x.drop(columns=[col])
        return train_x, test_x

    def prepare_features(self, train_data, test_data):
        available_features = [col for col in self.features if col in train_data.columns]

        train_x = train_data[available_features].copy()
        test_x = test_data[available_features].copy()

        train_y = train_data[self.target_column]
        test_y = test_data[self.target_column]

        train_x, test_x = self._encode_categorical_features(train_x, test_x, train_y)
        train_x, test_x = self._final_cleanup(train_x, test_x)

        if self.model_name == "AnomalyDetector":
            self.scaler = StandardScaler()
            train_x = pd.DataFrame(self.scaler.fit_transform(train_x), columns=train_x.columns)
            test_x = pd.DataFrame(self.scaler.transform(test_x), columns=test_x.columns)

        self.feature_columns = train_x.columns.tolist()
        logger.info(f"Final feature preparation complete - Train: {train_x.shape}, Test: {test_x.shape}")

        return train_x, test_x, train_y, test_y

    def _final_cleanup(self, train_x, test_x):
        remaining_object_cols = train_x.select_dtypes(include=["object"]).columns.tolist()
        if remaining_object_cols:
            logger.warning(f"Dropping remaining object columns: {remaining_object_cols}")
            train_x = train_x.drop(columns=remaining_object_cols)
            test_x = test_x.drop(columns=remaining_object_cols)

        train_x = train_x.apply(pd.to_numeric, errors="coerce").fillna(0)
        test_x = test_x.apply(pd.to_numeric, errors="coerce").fillna(0)

        test_x = test_x.reindex(columns=train_x.columns, fill_value=0)
        return train_x, test_x

    def get_model(self):
        if self.model_name == "HeartRatePredictor":
            config = self.params.HEART_RATE_PREDICTOR
            return RandomForestRegressor(
                n_estimators=config.n_estimators,
                max_depth=config.max_depth,
                random_state=config.random_state
            )
        elif self.model_name == "AnomalyDetector":
            config = self.params.ANOMALY_DETECTOR
            return IsolationForest(
                n_estimators=config.n_estimators,
                max_samples=config.max_samples if hasattr(config, "max_samples") else "auto",
                contamination=config.contamination,
                random_state=config.random_state
            )
        else:
            raise ValueError(f"Unsupported model name: {self.model_name}")

    def _evaluate_regression(self, model, train_x, test_x, train_y, test_y):
        train_preds = model.predict(train_x)
        test_preds = model.predict(test_x)

        metrics = {
            "train_rmse": np.sqrt(mean_squared_error(train_y, train_preds)),
            "test_rmse": np.sqrt(mean_squared_error(test_y, test_preds)),
            "train_mae": np.mean(np.abs(train_y - train_preds)),
            "test_mae": np.mean(np.abs(test_y - test_preds)),
            "test_r2": r2_score(test_y, test_preds)
        }

        print("=== Regression Results ===")
        print(f"Train RMSE: {metrics['train_rmse']:.4f}")
        print(f"Test RMSE: {metrics['test_rmse']:.4f}")
        print(f"Train MAE: {metrics['train_mae']:.4f}")
        print(f"Test MAE: {metrics['test_mae']:.4f}")
        print(f"Test R²: {metrics['test_r2']:.4f}")

        return metrics, test_preds

    def _evaluate_classification(self, train_y, test_y, train_preds, test_preds):
        train_y = np.where(train_y != 0, 1, 0)
        test_y = np.where(test_y != 0, 1, 0)

        metrics = {
            "train_accuracy": accuracy_score(train_y, train_preds),
            "test_accuracy": accuracy_score(test_y, test_preds),
            "test_precision": precision_score(test_y, test_preds, zero_division=0),
            "test_recall": recall_score(test_y, test_preds, zero_division=0),
            "test_f1": f1_score(test_y, test_preds, zero_division=0)
        }

        print("=== Classification Results ===")
        print(f"Train Accuracy: {metrics['train_accuracy']:.4f}")
        print(f"Test Accuracy: {metrics['test_accuracy']:.4f}")
        print(f"Test Precision: {metrics['test_precision']:.4f}")
        print(f"Test Recall: {metrics['test_recall']:.4f}")
        print(f"Test F1 Score: {metrics['test_f1']:.4f}\n")

        print("=== Class Distribution ===")
        print(f"Train - Normal: {(train_y == 0).sum()}, Anomaly: {(train_y == 1).sum()}")
        print(f"Test - Normal: {(test_y == 0).sum()}, Anomaly: {(test_y == 1).sum()}")
        print(f"Predictions - Normal: {(test_preds == 0).sum()}, Anomaly: {(test_preds == 1).sum()}")

        return metrics

    def save_model_artifacts(self, model, metrics=None, test_data=None):
        os.makedirs(self.config.root_dir, exist_ok=True)

        model_artifacts = {
            "model": model,
            "scaler": self.scaler,
            "encoder": self.encoder,
            "feature_columns": self.feature_columns,
            "target_column": self.target_column,
            "model_type": type(model).__name__,
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics
        }

        if self.model_name == "HeartRatePredictor":
            model_filename = self.config.heart_rate_predictor_model_name
        elif self.model_name == "AnomalyDetector":
            model_filename = self.config.anomaly_detector_model_name
        else: raise ValueError(f"Unsupported model name: {self.model_name}")
        

        model_path = os.path.join(self.config.root_dir, model_filename)
        joblib.dump(model_artifacts, model_path)
        logger.info(f"Model artifacts saved at: {model_path}")

        if test_data is not None:
            test_data_path = os.path.join(self.config.root_dir, f"{self.model_name}_test_data.joblib")
            joblib.dump(test_data, test_data_path)
            logger.info(f"Test data saved at: {test_data_path}")

    def train_model(self):
        try:
            train_data, test_data = self.load_data()
            train_x, test_x, train_y, test_y = self.prepare_features(train_data, test_data)

            model = self.get_model()
            logger.info(f"Training {self.model_name} with features shape: {train_x.shape}")

            if self.model_name == "HeartRatePredictor":
                model.fit(train_x, train_y)
                metrics, predictions = self._evaluate_regression(model, train_x, test_x, train_y, test_y)

            else:
                model.fit(train_x)
                train_preds = np.where(model.predict(train_x) == -1, 1, 0)
                test_preds = np.where(model.predict(test_x) == -1, 1, 0)
                metrics = self._evaluate_classification(train_y, test_y, train_preds, test_preds)
                predictions = test_preds

            test_data_combined = pd.concat([test_x, test_y], axis=1)
            self.save_model_artifacts(model, metrics=metrics, test_data=test_data_combined)

            logger.info(f"Model training completed successfully for {self.model_name}")
            return model, predictions

        except Exception as e:
            logger.error(f"Error in model training: {str(e)}")
            raise e
