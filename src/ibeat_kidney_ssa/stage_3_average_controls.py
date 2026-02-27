import os
import logging
import argparse
from pathlib import Path

import numpy as np
from dbdicom import npz
import vreg
import miblab_ssa as ssa
from miblab import pipe

from ibeat_kidney_ssa.utils import data

PIPELINE = 'kidney_ssa'

def run(build):

    logging.info("Stage 3 --- Computation of average control kidney ---")
    dir_input = os.path.join(build, PIPELINE, 'stage_1_normalize_controls')
    dir_output = pipe.stage_output_dir(build, PIPELINE, __file__)

    logging.info("Stage 3. Loading and sorting masks")
    kidney_masks = [k for k in npz.series(dir_input) if k[2][0] == 'Visit1']
    kidney_masks, kidney_labels = data.sort_kidney_masks(kidney_masks) 
    kidney_masks = [npz.volume(m).values.astype(bool) for m in kidney_masks]

    logging.info("Stage 3. Computing feature vectors")
    feature_file = os.path.join(dir_output, 'features.npz')
    ssa.features_from_dataset_in_memory(
        ssa.sdf_ft.features_from_mask, 
        kidney_masks, 
        feature_file, 
        kidney_labels,
        order=16,
    )

    logging.info("Stage 3. Computing average")
    with np.load(feature_file) as feature_data:
        # Load feature matrix and compute average
        X = feature_data['features']
        Xmean = np.mean(X, axis=0)

        # Compute mask and save
        mean_series = [dir_output, 'iBEAt_000', ('Visit1', 0), ('average_normalized_control_kidney', 0)]
        mean_mask = ssa.sdf_ft.mask_from_features(Xmean, feature_data['original_shape'], feature_data['order'])
        mean_vol = vreg.volume(mean_mask.astype(int))
        npz.write_volume(mean_vol, mean_series)

    # Cleanup
    Path(feature_file).unlink()
    logging.info("Stage 3. Computation of average control kidney completed.")
                 






if __name__ == '__main__':

    BUILD = r"C:\Users\md1spsx\Documents\Data\iBEAt_Build"
    pipe.run_script(run, BUILD, PIPELINE)