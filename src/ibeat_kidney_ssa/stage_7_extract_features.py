import os
import logging
import argparse
from pathlib import Path

from tqdm import tqdm
import numpy as np
import pydmr
import numpyradiomics as npr
from dbdicom import npz
import dask



def run(build):

    # Define folders
    dir_input = os.path.join(build, 'kidney_ssa', 'stage_5_normalize') 
    dir_output = os.path.join(build, 'kidney_ssa', 'stage_7_extract_features')
    logging.basicConfig(
        filename=f"{dir_output}.log",
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    os.makedirs(dir_output, exist_ok=True)

    # Get kidney mask files at baseline
    kidney_masks = npz.series(dir_input)

    # Compute sequentially
    # dmr_files = [run_kidney(kidney_mask, dir_output) for kidney_mask in tqdm(kidney_masks, desc='Extracting shape features..')]

    # Compute in parallel
    logging.info("Stage 7: Scheduling feature extraction")
    tasks = [dask.delayed(compute_shape_features)(mask, dir_output) for mask in kidney_masks]

    logging.info("Stage 7: Computing feature extraction")
    dmr_files = dask.compute(*tasks)
    
    # Build single file
    dmr_files = [f for f in dmr_files if f is not None]
    if dmr_files == []:
        logging.info(f"No valid features computed")
        return
    dmr_file = os.path.join(dir_output, f'all_kidneys')
    pydmr.concat(dmr_files, dmr_file, cleanup=True)

    # # Delete source files
    # [Path(f).unlink() for f in dmr_files]


def compute_shape_features(mask_series, dir_output):

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

    BUILD = r'C:\Users\md1spsx\Documents\Data\iBEAt_Build'

    parser = argparse.ArgumentParser()
    parser.add_argument("--build", type=str, default=BUILD, help="Build folder")
    args = parser.parse_args()

    run(args.build)
