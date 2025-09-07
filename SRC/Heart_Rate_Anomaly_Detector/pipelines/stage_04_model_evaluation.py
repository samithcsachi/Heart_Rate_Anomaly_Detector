from Heart_Rate_Anomaly_Detector.config.configuration import ConfigurationManager
from Heart_Rate_Anomaly_Detector.components.model_evaluation import ModelEvaluation
from Heart_Rate_Anomaly_Detector import logger


STAGE_NAME = "Model Trainer Stage"


class ModelEvaluationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        model_evaluation_config = config.get_model_evaluation_config()
        model_evaluator = ModelEvaluation(config=model_evaluation_config)
        evaluation_results = model_evaluator.evaluate_all()


if __name__ == "__main__":
    try:
        logger.info(f"\n\n{'*'*20} {STAGE_NAME} {'*'*20}\n")
        obj = ModelEvaluationTrainingPipeline()
        obj.main()
        logger.info(f"\n\n{'*'*20} {STAGE_NAME} completed {'*'*20}\n")
    except Exception as e:
        logger.exception(e)
        raise e