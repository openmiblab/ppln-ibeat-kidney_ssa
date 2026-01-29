import os
import logging
import argparse

from tqdm import tqdm
import numpy as np
import dbdicom as db

from utils import normalize

def save_normalized_as_npz(data, build):

    kidney_masks = db.series(data)
    for kidney_mask in tqdm(kidney_masks, desc='Saving kidneys as npz..'):
        patient = kidney_mask[1]
        study_desc, study_id = kidney_mask[2][0], kidney_mask[2][1]
        series_desc, series_nr = kidney_mask[3][0], kidney_mask[3][1]
        filedir = os.path.join(
            build, 
            f"Patient__{patient}", 
            f"Study__{study_id + 1}__{study_desc}",
        )
        os.makedirs(filedir, exist_ok=True)
        filename = f"Series__{series_nr + 1}__{series_desc}.npz"
        filepath = os.path.join(filedir, filename)
        if os.path.exists(filepath):
            continue
        vol = db.volume(kidney_mask, verbose=0)
        array = vol.values.astype(bool)
        np.savez_compressed(filepath, mask=array)

 


if __name__ == '__main__':

    DATA = r"C:\Users\md1spsx\Documents\Data\iBEAt_Build\kidneyvol"

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=DATA, help="Project folder")
    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        filename=os.path.join(args.data, 'log.txt'),      # log file name
        level=logging.INFO,           # log level
        format='%(asctime)s - %(levelname)s - %(message)s',  # log format
    )

    # save_normalized_as_npz(DATA, BUILD)
    normalize(args.data)
