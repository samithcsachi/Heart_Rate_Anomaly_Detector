from Heart_Rate_Anomaly_Detector.config.configuration import ConfigurationManager
from Heart_Rate_Anomaly_Detector.components.data_generation import DataGeneration
from Heart_Rate_Anomaly_Detector import logger


STAGE_NAME = "Data Generation Stage"

class DataGenerationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_generation_config = config.get_data_generation_config()
        data_ingestion = DataGeneration(config=data_generation_config)
        data_ingestion.generate_files()


if __name__ == "__main__":
    try:
        logger.info(f"\n\n{'*'*20} {STAGE_NAME} {'*'*20}\n")
        obj = DataGenerationTrainingPipeline()
        obj.main()
        logger.info(f"\n\n{'*'*20} {STAGE_NAME} completed {'*'*20}\n")
    except Exception as e:
        logger.exception(e)
        raise e