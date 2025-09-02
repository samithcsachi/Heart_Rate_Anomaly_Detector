from dataclasses import dataclass
from pathlib import Path
from typing import List


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