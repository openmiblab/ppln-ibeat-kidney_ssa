import os, sys
import logging
from pathlib import Path


def setup_logging(build, pipeline):
    dir_logs = os.path.join(build, pipeline)
    os.makedirs(dir_logs, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"{dir_logs}.log"),
            logging.StreamHandler(sys.stdout),
        ],
    )

def setup_pipeline(build, pipeline):
    setup_logging(build, pipeline)

def setup_stage(build, pipeline, module):
    setup_logging(build, pipeline)

    # Outputs of the stage
    stage = Path(module).name[:-3]
    dir_output = os.path.join(build, pipeline, stage)
    os.makedirs(dir_output, exist_ok=True)

    return dir_output