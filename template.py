import os 
from pathlib import Path
import logging 


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


project_name = 'Heart_Rate_Anomaly_Detector'

list_of_files = [
    '.github/workflows/.gitkeep',
    f"SRC/{project_name}/__init__.py",
    f"SRC/{project_name}/components/__init__.py",
    f"SRC/{project_name}/utils/__init__.py",
    f"SRC/{project_name}/utils/common.py",
    f"SRC/{project_name}/config/__init__.py",
    f"SRC/{project_name}/config/configuration.py",
    f"SRC/{project_name}/pipelines/__init__.py",
    f"SRC/{project_name}/entity/__init__.py",
    f"SRC/{project_name}/entity/config_entity.py",
    f"SRC/{project_name}/constants/__init__.py",
    "config/config.yaml",
    "dvc.yaml",
    "params.yaml",
    "schema.yaml",
    "research/01_data_generation.ipynb",
    "research/02_data_validation.ipynb",
    "research/03_data_transformation.ipynb",
    "research/04_model_trainer.ipynb",  
    "research/05_model_evaluation.ipynb",
    "requirements.txt",
    "setup.py",
    "main.py",
    
]

for file_path in list_of_files:
    file_path = Path(file_path)
    file_dir, filename = os.path.split(file_path)
   
    if file_dir !="":
        os.makedirs(file_dir, exist_ok=True)
        logging.info(f"Created directory: {file_dir} for the file: {filename}")
    
    if (not os.path.exists(file_path)) or (os.path.getsize(file_path) == 0):
        with open(file_path, 'w') as f:
            pass
        logging.info(f"Created empty file: {file_path}")

    else:
        logging.info(f"File already exists: {file_path} and is not empty.")
        