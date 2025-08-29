from Heart_Rate_Anomaly_Detector import logger
from Heart_Rate_Anomaly_Detector.pipelines.stage_01_data_generation import DataGenerationTrainingPipeline



STAGE_NAME = "Data Generation Stage"


try:
    logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")
    data_generation = DataGenerationTrainingPipeline()
    data_generation.main()
    logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e