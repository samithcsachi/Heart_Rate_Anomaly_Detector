from Heart_Rate_Anomaly_Detector.constants import  *
from Heart_Rate_Anomaly_Detector.utils.common import read_yaml, create_directories
from Heart_Rate_Anomaly_Detector.entity.config_entity import DataGenerationConfig



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