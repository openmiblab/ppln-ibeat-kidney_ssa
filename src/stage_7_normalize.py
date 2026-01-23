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
import utils.data



def normalize_kidneys(build_path):

    datapath = os.path.join(build_path, 'kidneyvol', 'stage_3_edit')
    resultspath = os.path.join(build_path, 'kidneyvol', 'stage_7_normalized')

    kidney_labels = db.series(datapath)
    current_results = db.series(resultspath)

    spacing_norm = 1.0
    volume_norm = 1e6

    for kidney_label in tqdm(kidney_labels, desc=f'Normalizing kidneys..'):
        study = [resultspath] + kidney_label[1:3]

        # Define outputs
        rk_series = study + [('normalized_right_kidney_mask', 0)]
        lk_series = study + [('normalized_left_kidney_mask', 0)]

        # Read data
        if (rk_series not in current_results) or (lk_series not in current_results):
            try:
                vol = db.volume(kidney_label, verbose=0)
            except Exception as e:
                logging.error(f"Cannot read data of patient {rk_series[1]} in study {rk_series[2][0]}: {e}")
                continue

        # Compute right kidney
        if rk_series not in current_results:
            try:
                rk_mask = vol.values == 2
                rk_mask_norm, _ = normalize.normalize_kidney_mask(rk_mask, vol.spacing, 'right')
                rk_vol_norm = vreg.volume(rk_mask_norm.astype(int))
                db.write_volume(rk_vol_norm, rk_series, ref=kidney_label, verbose=0)
            except Exception as e:
                logging.error(f"Error normalizing right kidney of patient {rk_series[1]} in study {rk_series[2][0]}: {e}")

        # Compute left kidney
        if lk_series not in current_results:
            try:
                lk_mask = vol.values == 1
                lk_mask_norm, _ = normalize.normalize_kidney_mask(lk_mask, vol.spacing, 'left')
                lk_vol_norm = vreg.volume(lk_mask_norm.astype(int))
                db.write_volume(lk_vol_norm, lk_series, ref=kidney_label, verbose=0)
            except Exception as e:
                logging.error(f"Error normalizing left kidney of patient {lk_series[1]} in study {lk_series[2][0]}: {e}")



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
