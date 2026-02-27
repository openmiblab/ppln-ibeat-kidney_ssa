import os
import logging
import argparse

import dbdicom as db
from dbdicom import npz
import vreg
import dask
import miblab_ssa as ssa

from miblab import pipe

PIPELINE = 'kidney_ssa'

def run(build):

    logging.info("Stage 1 --- Normalize controls ---")
    dir_input = os.path.join(build, 'kidney_shape', 'stage_3_edit', 'Controls')
    dir_output = pipe.stage_output_dir(build, PIPELINE, __file__)

    logging.info("Stage 1. Scheduling tasks..")
    kidney_labels = db.series(dir_input)
    tasks = [   
        dask.delayed(normalize_kidneys)(kl, dir_output) 
        for kl in kidney_labels
    ]
    logging.info('Stage 1. Computing tasks..')
    dask.compute(*tasks)

    logging.info("Stage 1. Finished normalizing controls.")


def normalize_kidneys(kidney_label, db_normalize):
        
    # Define outputs
    study = [db_normalize] + kidney_label[1:3]
    lk_series = study + [('normalized_kidney_left', 0)]
    rk_series = study + [('normalized_kidney_right', 0)]
    lk_exists = npz.exists(lk_series)
    rk_exists = npz.exists(rk_series)

    # Open database for reading (This avoids writing to disk for parallel computation)
    dbd = db.open(kidney_label[0])

    try:
    
        # Read data
        if (not rk_exists) or (not lk_exists):
            vol = dbd.volume(kidney_label, verbose=0).to_right_handed()
            
        # Compute right kidney
        if not rk_exists:
            rk_mask = vol.values == 2
            if rk_mask.sum() == 0:
                logging.info(f"Stage 1. Empty right kidney in study: {study[1:]}")
            else:
                rk_mask_norm, _ = ssa.normalize_kidney_mask(rk_mask, vol.spacing, 'right', verbose=0)
                rk_vol_norm = vreg.volume(rk_mask_norm.astype(int))
                npz.write_volume(rk_vol_norm, rk_series)

        # Compute left kidney
        if not lk_exists:
            lk_mask = vol.values == 1
            if lk_mask.sum() == 0:
                logging.info(f"Stage 1. Empty left kidney in study: {study[1:]}")
            else:
                lk_mask_norm, _ = ssa.normalize_kidney_mask(lk_mask, vol.spacing, 'left', verbose=0)
                lk_vol_norm = vreg.volume(lk_mask_norm.astype(int))
                npz.write_volume(lk_vol_norm, lk_series)

        logging.info(f"Stage 1. Successfully normalized kidneys in study: {study[1:]}")
    except:
        logging.exception(f"Stage 1. Error normalizing kidneys in study: {study[1:]}")




if __name__ == '__main__':

    BUILD = r"C:\Users\md1spsx\Documents\Data\iBEAt_Build"
    pipe.run_script(run, BUILD, PIPELINE)
