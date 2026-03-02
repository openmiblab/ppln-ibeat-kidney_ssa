import os, sys
import logging

import numpy as np
import pydmr
import numpyradiomics as npr
from dbdicom import npz
from joblib import Parallel, delayed
from tqdm import tqdm

from miblab import pipe

PIPELINE = 'kidney_ssa'


def run(build, logfile):

    logging.info("Stage 4 --- Extracting shape features ---")
    dir_input = os.path.join(build, PIPELINE, 'stage_3_normalize', 'normalized_kidney_masks_database') 
    dir_output = pipe.stage_output_dir(build, PIPELINE, __file__)
    # pipe.adjust_workers(client, min_ram_per_worker=6, overhead_ram=16)

    logging.info("Stage 4: Computing feature extraction")
    kidney_masks = npz.series(dir_input)
    dmr_files = Parallel(n_jobs=-1, verbose=10)(
        delayed(compute_shape_features)(mask, dir_output, logfile) 
        for mask in tqdm(kidney_masks, desc='Extracting features')
    )
    
    # Build single file
    dmr_files = [f for f in dmr_files if f is not None]
    if dmr_files == []:
        logging.info(f"No valid features computed")
        return
    dmr_file = os.path.join(dir_output, f'all_kidneys')
    pydmr.concat(dmr_files, dmr_file)

    logging.info("Stage 4: Completed feature extraction")


def compute_shape_features(mask_series, dir_output, logfile):

    # Setup logger inside worker
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(logfile),
            logging.StreamHandler(sys.stdout),
        ],
    )

    # Get IDs
    patient_id = mask_series[1]
    study_desc = mask_series[2][0]
    series_desc = mask_series[3][0]

    # Define outputs
    fname = f"{patient_id}_{study_desc}_{series_desc}.dmr.zip"
    dmr_file = os.path.join(dir_output, fname)

    # Skip if output exists
    if os.path.exists(dmr_file):
        return

    try:

        # Read binary mask
        vol = npz.volume(mask_series)
        mask = vol.values.astype(np.float32)
        if np.sum(mask) == 0:
            logging.info(f"{fname}: empty mask")
            return

        # Get radiomics shape features in cm (this produces a volume of 1L)
        results = npr.shape(mask, vol.spacing / 10, transpose=True, extend=True)
        units = npr.shape_units(3, 'cm')

        # Write to dmr file
        dmr = {
            'data': {f"{series_desc}-shape-{p}": [f"Shape measure {p} for {series_desc}", u, 'float'] for p, u in units.items()},
            'pars': {(patient_id, study_desc, f"{series_desc}-shape-{p}"): v for p, v in results.items()}
        }
        pydmr.write(dmr_file, dmr)
        logging.info(f"Stage 7: Successfully computed features: {fname}")
        return dmr_file

    except:
        logging.exception(f"Stage 7: Error computing features: {fname}")



if __name__=='__main__':

    BUILD = r"C:\Users\md1spsx\Documents\Data\iBEAt_Build"
    # pipe.run_client_stage(run, BUILD, PIPELINE, __file__, min_ram_per_worker=16)
    pipe.run_stage(run, BUILD, PIPELINE, __file__)
