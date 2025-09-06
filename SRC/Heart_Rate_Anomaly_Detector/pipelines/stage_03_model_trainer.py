from Heart_Rate_Anomaly_Detector.config.configuration import ConfigurationManager
from Heart_Rate_Anomaly_Detector.components.model_trainer import ModelTrainer
from Heart_Rate_Anomaly_Detector import logger


STAGE_NAME = "Model Trainer Stage"


class ModelTrainerTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config_manager = ConfigurationManager()
        model_trainer_config = config_manager.get_model_trainer_config()
        
        hr_trainer = ModelTrainer(
            config=model_trainer_config, 
            schema_config=config_manager.schema,  
            params=config_manager.params,         
            model_name="HeartRatePredictor"
        )
        hr_model, hr_predictions = hr_trainer.train_model()   
        
        anomaly_trainer = ModelTrainer(
            config=model_trainer_config, 
            schema_config=config_manager.schema,  
            params=config_manager.params,         
            model_name="AnomalyDetector"
        )
        anomaly_model, anomaly_predictions = anomaly_trainer.train_model()

if __name__ == "__main__":
    try:
        logger.info(f"\n\n{'*'*20} {STAGE_NAME} {'*'*20}\n")
        obj = ModelTrainerTrainingPipeline()
        obj.main()
        logger.info(f"\n\n{'*'*20} {STAGE_NAME} completed {'*'*20}\n")
    except Exception as e:
        logger.exception(e)
        raise e
    
