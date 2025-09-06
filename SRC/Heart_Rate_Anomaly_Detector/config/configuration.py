from Heart_Rate_Anomaly_Detector.constants import  *
from Heart_Rate_Anomaly_Detector.utils.common import read_yaml, create_directories
from Heart_Rate_Anomaly_Detector.entity.config_entity import (DataGenerationConfig, DataTransformationConfig, ModelTrainerConfig)



class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH,
        schema_filepath = SCHEMA_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        self.schema = read_yaml(schema_filepath)

        create_directories([self.config.artifacts_root])

    def get_data_generation_config(self) -> DataGenerationConfig:
        config = self.config.data_generation
        create_directories([config.root_dir])
        data_generation_config = DataGenerationConfig(
            root_dir=config.root_dir,
            local_data_file_reading=config.local_data_files.readings,
            local_data_file_users=config.local_data_files.users,
        )
        return data_generation_config
    

    
    def get_data_transformation_config(self, model: str) -> DataTransformationConfig:
        config = self.config.data_transformation

        
        model_schema = self.schema.models.get(model)
        if model_schema is None:
            raise ValueError(f"Unknown model: {model}")

        return DataTransformationConfig(
            root_dir=config.root_dir,
            data_path_reading=config.data_path.readings,
            data_path_users=config.data_path.users,
            target_column=model_schema.target_column,
            features=model_schema.features
    )


    def get_model_trainer_config(self) -> ModelTrainerConfig:
        config = self.config.model_trainer
        
        
        heart_rate_params = self.params.HEART_RATE_PREDICTOR
        anomaly_params = self.params.ANOMALY_DETECTOR
        
       
        heart_rate_model_schema = self.schema.models.HeartRatePredictor
        anomaly_model_schema = self.schema.models.AnomalyDetector

        create_directories([config.root_dir])

        model_trainer_config = ModelTrainerConfig(
            root_dir=Path(config.root_dir),
            train_heart_rate_data_path=Path(config.data_path.train_heart_rate),
            test_heart_rate_data_path=Path(config.data_path.test_heart_rate),
            train_is_anomaly_data_path=Path(config.data_path.train_is_anomaly),
            test_is_anomaly_data_path=Path(config.data_path.test_is_anomaly),
            data_transformation_dir=Path(self.config.data_transformation.root_dir),
            
         
            heart_rate_predictor_model_name=config.model_name.heart_rate_predictor,
            anomaly_detector_model_name=config.model_name.anomaly_detector,
            
           
            heart_rate_target_column=heart_rate_model_schema.target_column,
            anomaly_target_column=anomaly_model_schema.target_column
        )
        return model_trainer_config
    
    def get_heart_rate_features(self) -> list:
       
        return self.schema.models.HeartRatePredictor.features
    
    def get_anomaly_features(self) -> list:
       
        return self.schema.models.AnomalyDetector.features
    
    def get_column_dtypes(self) -> dict:
        
        return self.schema.columns