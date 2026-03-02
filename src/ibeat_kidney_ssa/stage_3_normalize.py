import os
import sys
import logging

import dbdicom as db
from dbdicom import npz
import vreg
import miblab_ssa as ssa
from joblib import Parallel, delayed
from miblab import pipe
from tqdm import tqdm
import zarr

from ibeat_kidney_ssa.utils import data, display

PIPELINE = 'kidney_ssa'

def run(build, logfile):

    dir_output = pipe.stage_output_dir(build, PIPELINE, __file__)

    logging.info("Stage 3 --- Normalizing kidneys ---")
    dir_input = os.path.join(build, 'kidney_shape', 'stage_3_edit')
    dir_input_ref = os.path.join(build, PIPELINE, 'stage_2_average_controls')
    # pipe.adjust_workers(client, min_ram_per_worker=4, overhead_ram=8)
    
    # Inputs
    average_kidney_mask = os.path.join(dir_input_ref, 'average_kidney_mask.zarr')

    # Outputs
    db_normalize = os.path.join(dir_output, 'normalized_kidney_masks_database')
    zarr_path = os.path.join(dir_output, 'normalized_kidney_masks.zarr')
    dir_png = os.path.join(dir_output, 'normalized_kidney_masks_imgs')
    imgs_mp4 = os.path.join(dir_output, 'normalized_kidney_masks_movie.mp4')

    for database in ['Controls', 'Patients']:

        logging.info(f"Stage 3. Computing {database}..")
        db_masks = os.path.join(dir_input, database) 
        kidney_labels = db.series(db_masks)   
        Parallel(n_jobs=-1, verbose=10)(
            delayed(normalize_kidneys)(kl, average_kidney_mask, db_normalize, logfile) 
            for kl in tqdm(kidney_labels, desc=f'Normalizing {database} kidneys')
        )

    logging.info(f"Stage 3. Saving baseline normalized kidneys as zarr..")
    kidney_masks = [k for k in npz.series(db_normalize) if k[2][0] in ['Visit1', 'Baseline']]
    kidney_masks_sorted, kidney_labels_sorted = data.sort_kidney_masks(kidney_masks) 
    kidney_masks_sorted = [npz.file(mask) for mask in kidney_masks_sorted]
    ssa.save_masks_as_zarr(zarr_path, kidney_masks_sorted, kidney_labels_sorted)

    logging.info('Stage 3. Display baseline normalized kidneys')
    # pipe.adjust_workers(client, min_ram_per_worker=16)
    display.recon(zarr_path, dir_png, imgs_mp4)

    logging.info("Stage 3 --- Finished normalizing kidneys ---")


def normalize_kidneys(kidney_label, average_kidney_mask, db_normalize, logfile):

    # Setup logger inside worker
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(logfile),
            logging.StreamHandler(sys.stdout),
        ],
    )
        
    # Define outputs
    study = [db_normalize] + kidney_label[1:3]
    lk_series = study + [('normalized_kidney_left', 0)]
    rk_series = study + [('normalized_kidney_right', 0)]
    lk_exists = npz.exists(lk_series)
    rk_exists = npz.exists(rk_series)

    # Open database for reading
    dbd = db.open(kidney_label[0])

    try:
        # Read data
        if (not rk_exists) or (not lk_exists):
            vol = dbd.volume(kidney_label, verbose=0).to_right_handed()
            ref_mask = zarr.open_group(average_kidney_mask, mode='r')['masks'][0]
            
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
    except Exception as e:
        logging.exception(f"Stage 5. Error normalizing kidneys in study {study[1:]}")



if __name__ == '__main__':

    BUILD = r"C:\Users\md1spsx\Documents\Data\iBEAt_Build"
    # pipe.run_client_stage(run, BUILD, PIPELINE, __file__, min_ram_per_worker=4)
    pipe.run_stage(run, BUILD, PIPELINE, __file__)
