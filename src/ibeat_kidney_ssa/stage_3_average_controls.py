import os
import logging
import argparse
from pathlib import Path

import numpy as np
from dbdicom import npz
import vreg

from ibeat_kidney_ssa.utils import sdf_ft, data


def run(build):

    dir_input = os.path.join(build, 'kidney_ssa', 'stage_1_normalize_controls')
    dir_output = os.path.join(build, 'kidney_ssa', 'stage_3_average_controls')
    os.makedirs(dir_output, exist_ok=True)

    logging.basicConfig(
        filename=f"{dir_output}.log",
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    logging.info("Computation of feature vectors started")

    # Get all baseline kidneys in sorted order
    # Get kidney mask files at baseline
    kidney_masks = [k for k in npz.series(dir_input) if k[2][0] == 'Visit1']
    kidney_masks_sorted, kidney_labels_sorted = data.sort_kidney_masks(kidney_masks) 
    kidney_files_sorted = [npz.file(m) for m in kidney_masks_sorted]

    # Compute feature vectors
    feature_file = ssa.compute_features(
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
        mean_mask = sdf_ft.mask_from_features(Xmean, feature_data['shape'], feature_data['order'])
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