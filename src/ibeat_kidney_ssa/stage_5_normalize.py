import os
import logging
import argparse

import dbdicom as db
from dbdicom import npz
import vreg
import dask
import miblab_ssa as ssa

from ibeat_kidney_ssa.utils import pipe

PIPELINE = 'kidney_ssa'

def run(build):

    logging.info("Stage 5 --- Normalizing kidneys ---")
    dir_input = os.path.join(build, 'kidney_shape', 'stage_3_edit')
    dir_input_ref = os.path.join(build, PIPELINE, 'stage_3_average_controls')
    dir_output = pipe.setup_stage(build, PIPELINE, __file__)
    for database in ['Controls', 'Patients']:

        logging.info(f"Stage 5. Computing {database}..")
        db_masks = os.path.join(dir_input, database) 
        run_db(db_masks, dir_input_ref, dir_output)


def run_db(db_masks, db_input_ref, db_normalize):
    
    # Parallel computation
    logging.info("Stage 5. Scheduling tasks..")
    kidney_labels = db.series(db_masks)
    tasks = [   
        dask.delayed(normalize_kidneys)(kl, db_input_ref, db_normalize) 
        for kl in kidney_labels
    ]
    logging.info('Stage 5. Computing tasks..')
    dask.compute(*tasks)


def normalize_kidneys(kidney_label, db_input_ref, db_normalize):
        
    # Define outputs
    study = [db_normalize] + kidney_label[1:3]
    lk_series = study + [('normalized_kidney_left', 0)]
    rk_series = study + [('normalized_kidney_right', 0)]
    lk_exists = npz.exists(lk_series)
    rk_exists = npz.exists(rk_series)

    # Get reference kidney
    ref_kidney = [db_input_ref, 'iBEAt_000', ('Visit1', 0), ('average_normalized_control_kidney', 0)]

    # Open database for reading
    dbd = db.open(kidney_label[0])

    try:
        # Read data
        if (not rk_exists) or (not lk_exists):
            vol = dbd.volume(kidney_label, verbose=0).to_right_handed()
            ref_mask = npz.volume(ref_kidney).values.astype(bool)
            
        # Compute right kidney
        if not rk_exists:
            rk_mask = vol.values == 2
            if rk_mask.sum() == 0:
                logging.info(f"Stage 1. Empty right kidney in study: {study[1:]}")
            else:
                rk_mask_norm, _ = ssa.normalize_kidney_mask(rk_mask, vol.spacing, 'right', ref=ref_mask, verbose=0)
                rk_vol_norm = vreg.volume(rk_mask_norm.astype(int))
                npz.write_volume(rk_vol_norm, rk_series)

        # Compute left kidney
        if not lk_exists:
            lk_mask = vol.values == 1
            if lk_mask.sum() == 0:
                logging.info(f"Stage 1. Empty left kidney in study: {study[1:]}")
            else:
                lk_mask_norm, _ = ssa.normalize_kidney_mask(lk_mask, vol.spacing, 'left', ref=ref_mask, verbose=0)
                lk_vol_norm = vreg.volume(lk_mask_norm.astype(int))
                npz.write_volume(lk_vol_norm, lk_series)

        logging.info(f"Stage 5. Successfully normalized kidneys in study: {study[1:]}")
    except:
        logging.exception(f"Stage 5. Error normalizing kidneys in study: {study[1:]}")




if __name__ == '__main__':

    BUILD = r"C:\Users\md1spsx\Documents\Data\iBEAt_Build"

    parser = argparse.ArgumentParser()
    parser.add_argument("--build", type=str, default=BUILD, help="Build folder")
    args = parser.parse_args()

    run(args.build)
