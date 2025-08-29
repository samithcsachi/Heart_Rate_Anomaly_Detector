from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataGenerationConfig:
    root_dir: Path
    local_data_file_reading: Path
    local_data_file_users: Path