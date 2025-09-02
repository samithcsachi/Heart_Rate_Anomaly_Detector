from Heart_Rate_Anomaly_Detector import logger
from Heart_Rate_Anomaly_Detector.pipelines.stage_01_data_generation import DataGenerationTrainingPipeline
from Heart_Rate_Anomaly_Detector.pipelines.stage_02_data_transformation import DataTransformationTrainingPipeline



STAGE_NAME = "Data Generation Stage"


try:
    logger.info(f"\n\n{'*'*20} {STAGE_NAME} {'*'*20}\n")
    data_generation = DataGenerationTrainingPipeline()
    data_generation.main()
    logger.info(f"\n\n{'*'*20} {STAGE_NAME} completed {'*'*20}\n")
except Exception as e:
    logger.exception(e)
    raise e




STAGE_NAME = "Data Transformation Stage"


try:
    logger.info(f"\n\n{'*'*20} {STAGE_NAME} {'*'*20}\n")
    data_transformation = DataTransformationTrainingPipeline()
    data_transformation.main()
    logger.info(f"\n\n{'*'*20} {STAGE_NAME} completed {'*'*20}\n")
except Exception as e:
    logger.exception(e)
    raise e