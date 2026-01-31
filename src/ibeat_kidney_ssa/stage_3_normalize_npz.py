import os
import logging
import argparse

from tqdm import tqdm
import numpy as np
import dbdicom as db


def run(build):

    dir_input = os.path.join(build, 'kidney_ssa', 'stage_1_normalize')
    dir_output = os.path.join(build, 'kidney_ssa', 'stage_3_normalize_npz')
    logging.basicConfig(
        filename=f"{dir_output}.log",
        level=logging.INFO,          
        format='%(asctime)s - %(levelname)s - %(message)s',  
    )
    try:
        db.to_npz(dir_input, dir_output)
        logging.info(f"Successfully written: {dir_input}")
    except:
        logging.exception(f"Error writing: {dir_input}")

 
if __name__ == '__main__':

    BUILD = r"C:\Users\md1spsx\Documents\Data\iBEAt_Build"

    parser = argparse.ArgumentParser()
    parser.add_argument("--build", type=str, default=BUILD, help="Build folder")
    args = parser.parse_args()

    run(args.build)
