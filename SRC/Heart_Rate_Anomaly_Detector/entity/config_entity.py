from dataclasses import dataclass
from pathlib import Path
from typing import List
from typing import Dict



@dataclass(frozen=True)
class DataGenerationConfig:
    root_dir: Path
    local_data_file_reading: Path
    local_data_file_users: Path


@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    data_path_reading: Path
    data_path_users: Path
    target_column: str
    features: List[str]


@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: Path
    train_heart_rate_data_path: Path
    test_heart_rate_data_path: Path
    train_is_anomaly_data_path: Path
    test_is_anomaly_data_path: Path
    data_transformation_dir: Path  
    
   
    heart_rate_predictor_model_name: str
    anomaly_detector_model_name: str
    
    
    heart_rate_target_column: str
    anomaly_target_column: str


@dataclass
class ModelEvaluationConfig:
    root_dir: Path
    test_data_path: Dict[str, Path]   
    model_path: Dict[str, Path]       
    report_path: Dict[str, Path]      
    target_columns: Dict[str, str]