import os
import logging
import argparse
from pathlib import Path

import numpy as np
from dbdicom import npz
import vreg
import miblab_ssa as ssa

from ibeat_kidney_ssa.utils import data, pipe

PIPELINE = 'kidney_ssa'

def run(build):

    dir_input = os.path.join(build, PIPELINE, 'stage_1_normalize_controls')
    dir_output = pipe.setup_stage(build, PIPELINE, __file__)

    logging.info("Computation of feature vectors started")

    # Get all baseline kidneys in sorted order
    # Get kidney mask files at baseline
    kidney_masks = [k for k in npz.series(dir_input) if k[2][0] == 'Visit1']
    kidney_masks_sorted, kidney_labels_sorted = data.sort_kidney_masks(kidney_masks) 
    kidney_files_sorted = [npz.file(m) for m in kidney_masks_sorted]

    # Compute feature vectors
    feature_file = ssa.sdf_ft.compute_features(
        dir_output, 
        kidney_files_sorted, 
        kidney_labels_sorted,
    )

    with np.load(feature_file) as feature_data:
        # Load feature matrix and compute average
        X = feature_data['features']
        Xmean = np.mean(X, axis=0)

        # Compute mask and save
        mean_series = [dir_output, 'MIBL_C01', ('Visit1', 0), ('normalized_kidney_avr', 0)]
        mean_mask = ssa.sdf_ft.mask_from_features(Xmean, feature_data['shape'], feature_data['order'])
        mean_vol = vreg.volume(mean_mask.astype(int))
        npz.write_volume(mean_vol, mean_series)

    # Cleanup
    Path(feature_file).unlink()

    logging.info("Computation of feature vectors completed.")
                 






if __name__ == '__main__':

    BUILD = r"C:\Users\md1spsx\Documents\Data\iBEAt_Build"

    parser = argparse.ArgumentParser()
    parser.add_argument("--build", type=str, default=BUILD, help="Build folder")
    args = parser.parse_args()

    run(args.build)