import os
import logging
import argparse
from pathlib import Path

from tqdm import tqdm
import numpy as np
import pydmr
import numpyradiomics as npr
import vreg

from ibeat_kidney_ssa.utils import data


def run(build):

    # Define folders
    dir_input = os.path.join(build, 'kidney_ssa', 'stage_3_normalize_npz') 
    dir_output = os.path.join(build, 'kidney_ssa', 'stage_5_features')
    logging.basicConfig(
        filename=f"{dir_output}.log",
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    os.makedirs(dir_output, exist_ok=True)

    # Get kidney mask files
    kidney_masks = [f for f in Path(dir_input).rglob("*") if f.is_file()]
    kidney_masks = [f for f in kidney_masks if ('Visit1' in str(f)) or ('Baseline' in str(f))]
    if kidney_masks == []:
        logging.info(f"No kidney masks in {dir_input}")
        return
    
    dmr_files = []
    for mask_npz_file in tqdm(kidney_masks, desc='Extracting shape features..'):

        # Get IDs
        _, value = data.parse_npz_dbfile(mask_npz_file)
        patient_id = value['PatientID']
        study_desc = value['StudyDescription']
        series_desc = value['SeriesDescription']

        # Define outputs
        fname = f"{patient_id}_{study_desc}_{series_desc}.dmr.zip"
        dmr_file = os.path.join(dir_output, fname)

        # Skip if output exists
        if os.path.exists(dmr_file):
            continue

        try:

            # Read binary mask
            vol = vreg.read_npz(mask_npz_file)
            mask = vol.values.astype(np.float32)
            if np.sum(mask) == 0:
                logging.info(f"{fname}: empty mask")
                continue

            # Get radiomics shape features in cm
            results = npr.shape(mask, vol.spacing / 10, transpose=True, extend=True)
            units = npr.shape_units(3, 'cm')

            # Write to dmr file
            dmr = {
                'data': {f"{series_desc}-shape-{p}": [f"Shape measure {p} for {series_desc}", u, 'float'] for p, u in units.items()},
                'pars': {(patient_id, study_desc, f"{series_desc}-shape-{p}"): v for p, v in results.items()}
            }
            pydmr.write(dmr_file, dmr)
            dmr_files.append(dmr_file)
            logging.info(f"Successfully computed shapes: {fname}")

        except:

            logging.exception(f"Error computing shapes: {fname}")

    if dmr_files != []:
        dmr_file = os.path.join(dir_output, f'all_kidneys')
        pydmr.concat(dmr_files, dmr_file)



if __name__=='__main__':

    BUILD = r'C:\Users\md1spsx\Documents\Data\iBEAt_Build'

    parser = argparse.ArgumentParser()
    parser.add_argument("--build", type=str, default=BUILD, help="Build folder")
    args = parser.parse_args()

    run(args.build)
