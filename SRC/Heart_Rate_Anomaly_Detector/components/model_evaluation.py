import pandas as pd
import os
import numpy as np
from pathlib import Path
from Heart_Rate_Anomaly_Detector import logger
import joblib
from Heart_Rate_Anomaly_Detector.entity.config_entity import ModelEvaluationConfig
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Dict



class ModelEvaluation:
    def __init__(self, config):
        
        self.config = config
        self.predictions: Dict[str, np.ndarray] = {}
        self.actuals: Dict[str, np.ndarray] = {}

    def load_model_and_artifacts(self, model_key: str):
        model_path = self.config.model_path[model_key]
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        model_artifacts = joblib.load(model_path)
        logger.info(f"Loaded model artifacts for {model_key}: {model_artifacts.get('model_type', 'Unknown')}")
        return model_artifacts

    def load_test_data(self, model_key: str):
        test_path = self.config.test_data_path[model_key]
        if not os.path.exists(test_path):
            raise FileNotFoundError(f"Test data not found: {test_path}")
        
        if str(test_path).endswith(".csv"):
            df = pd.read_csv(test_path)
        elif str(test_path).endswith(".joblib"):
            df = joblib.load(test_path)
            if isinstance(df, dict):
                X_test = df.get('X_test')
                y_test = df.get('y_test')
                if X_test is None or y_test is None:
                    raise ValueError(f"Joblib test data must contain 'X_test' and 'y_test'")
                return X_test, y_test
        else:
            raise ValueError(f"Unsupported test file format: {test_path}")
        
        target_col = self.config.target_columns[model_key]
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in test data")
        X_test = df.drop(columns=[target_col])
        y_test = df[target_col]
        return X_test, y_test

    def validate_feature_compatibility(self, X_test, expected_features):
        X_test = X_test.reindex(columns=expected_features, fill_value=0)
        logger.info(f"Feature validation completed. Final shape: {X_test.shape}")
        return X_test

    def calculate_metrics(self, y_true, y_pred):
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, 1e-8, y_true))) * 100

        residuals = y_true - y_pred
        mean_residual = np.mean(residuals)
        std_residual = np.std(residuals)

        pred_range = y_pred.max() - y_pred.min()
        actual_range = y_true.max() - y_true.min()
        range_coverage = (pred_range / actual_range * 100) if actual_range > 0 else 0

        metrics = {
            "mae": float(mae),
            "rmse": float(rmse),
            "r2": float(r2),
            "mape": float(mape),
            "mean_residual": float(mean_residual),
            "std_residual": float(std_residual),
            "range_coverage": float(range_coverage),
            "n_samples": len(y_true),
            "prediction_stats": {
                "mean": float(np.mean(y_pred)),
                "std": float(np.std(y_pred)),
                "min": float(np.min(y_pred)),
                "max": float(np.max(y_pred)),
            },
            "actual_stats": {
                "mean": float(np.mean(y_true)),
                "std": float(np.std(y_true)),
                "min": float(np.min(y_true)),
                "max": float(np.max(y_true)),
            }
        }
        return metrics

    def save_results(self, model_key: str, metrics, model_artifacts):
        os.makedirs(self.config.root_dir, exist_ok=True)

        result_path = Path(self.config.report_path[model_key])
        os.makedirs(result_path.parent, exist_ok=True)

        full_results = {
            "model_key": model_key,
            "model_info": {
                "model_type": model_artifacts.get("model_type", "Unknown"),
                "target_column": model_artifacts.get("target_column", "Unknown"),
                "timestamp": model_artifacts.get("timestamp", "Unknown"),
                "feature_count": len(model_artifacts.get("feature_columns", []))
            },
            "metrics": metrics,
            "evaluation_timestamp": datetime.now().isoformat()
        }

        with open(result_path, "w") as f:
            json.dump(full_results, f, indent=4)

        logger.info(f"Saved evaluation results for {model_key} at {result_path}")

    def evaluate_model(self, model_key: str):
        logger.info(f"Evaluating model: {model_key}")
        model_artifacts = self.load_model_and_artifacts(model_key)
        model = model_artifacts["model"]

        X_test, y_test = self.load_test_data(model_key)
        expected_features = model_artifacts.get("feature_columns", [])
        if expected_features:
            X_test = self.validate_feature_compatibility(X_test, expected_features)

        logger.info("Generating predictions...")
        y_pred = model.predict(X_test)

        self.predictions[model_key] = y_pred
        self.actuals[model_key] = y_test

        metrics = self.calculate_metrics(y_test, y_pred)
        self.save_results(model_key, metrics, model_artifacts)
        logger.info(f"Evaluation completed for {model_key}")
        return metrics

    def evaluate_all(self):
        results = {}
        for model_key in self.config.model_path.keys():
            results[model_key] = self.evaluate_model(model_key)
        return results
