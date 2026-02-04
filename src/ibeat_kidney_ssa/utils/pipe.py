import os, sys
import logging
from pathlib import Path


def setup_pipeline(build, pipeline):
    # Define output locations
    dir_output = os.path.join(build, pipeline)
    os.makedirs(dir_output, exist_ok=True)
    logging.basicConfig(
        filename=f"{dir_output}.log",
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def setup_stage(build, pipeline, module):
    # Outputs of the stage
    stage = Path(module).name[:-3]
    dir_output = os.path.join(build, pipeline, stage)
    os.makedirs(dir_output, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"{dir_output}.log"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return dir_output