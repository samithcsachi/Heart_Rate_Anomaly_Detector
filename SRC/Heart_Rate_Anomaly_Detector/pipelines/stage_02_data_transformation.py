from Heart_Rate_Anomaly_Detector.config.configuration import ConfigurationManager
from Heart_Rate_Anomaly_Detector.components.data_transformation import DataTransformation
from Heart_Rate_Anomaly_Detector import logger


STAGE_NAME = "Data Transformation Stage"

class DataTransformationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        heart_rate_config = ConfigurationManager().get_data_transformation_config(model="HeartRatePredictor")
        transformer = DataTransformation(heart_rate_config)
        train_hr, test_hr = transformer.train_test_splitting()

        
        anomaly_config = ConfigurationManager().get_data_transformation_config(model="AnomalyDetector")
        transformer = DataTransformation(anomaly_config)
        train_anom, test_anom = transformer.train_test_splitting()


if __name__ == "__main__":
    try:
        logger.info(f"\n\n{'*'*20} {STAGE_NAME} {'*'*20}\n")
        obj = DataTransformationTrainingPipeline()
        obj.main()
        logger.info(f"\n\n{'*'*20} {STAGE_NAME} completed {'*'*20}\n")
    except Exception as e:
        logger.exception(e)
        raise e
    



    