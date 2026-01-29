import os
import logging
import argparse
from pathlib import Path
import time

from tqdm import tqdm
import numpy as np
import dbdicom as db
import vreg
import dask

from utils import normalize


def normalize_rotation(proj):

    data = os.path.join(proj, 'stage_7_normalized_npz')
    build = os.path.join(proj, 'stage_7_normalized_rotation_npz') 
    
    t0 = time.time()

    mask_ref = os.path.join(data, "Patient__3128_C03", "Study__1__Visit1", "Series__1__normalized_left_kidney_mask.npz")
    kidney_masks = [f for f in Path(data).rglob("*") if f.is_file()]
    tasks = [dask.delayed(_normalize_rotation)(mask, mask_ref, build) for mask in kidney_masks]
    dask.compute(*tasks)

    print(f"Computation time: {time.time() - t0:.3f} seconds")


def _normalize_rotation(kidney_mask, kidney_mask_ref, build):

    # Define filepath - replace datapath by build
    studypath = os.path.dirname(kidney_mask)
    patientpath = os.path.dirname(studypath)
    series = os.path.basename(kidney_mask)
    study = os.path.basename(studypath)
    patient = os.path.basename(patientpath)
    dir = os.path.join(build, patient, study)
    filepath = os.path.join(dir, series)
    if os.path.exists(filepath):
        return

    # Compute coeffs
    mask = np.load(kidney_mask)['mask']
    mask_ref = np.load(kidney_mask_ref)['mask']
    dice, mask_rot = normalize.invariant_dice_coefficient(mask_ref, mask, angle_step=1, return_mask=True, verbose=0)

    # Save with label
    os.makedirs(dir, exist_ok=True)
    np.savez_compressed(filepath, mask=mask_rot.astype(bool))    


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
    normalize_rotation(args.data)
