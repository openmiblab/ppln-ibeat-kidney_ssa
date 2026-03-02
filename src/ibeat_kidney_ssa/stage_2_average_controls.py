import os
import logging
import argparse
from pathlib import Path

import numpy as np
from dbdicom import npz
import vreg
import miblab_ssa as ssa
from miblab import pipe
import zarr

from ibeat_kidney_ssa.utils import display


PIPELINE = 'kidney_ssa'

def run(build, logfile):

    logging.info("Stage 2 --- Computation of average control kidney ---")
    dir_input = os.path.join(build, PIPELINE, 'stage_1_normalize_controls')
    dir_output = pipe.stage_output_dir(build, PIPELINE, __file__)

    # Input
    zarr_path = os.path.join(dir_input, 'normalized_kidney_masks.zarr')

    # Output
    features_zarr = os.path.join(dir_output, 'normalized_kidney_features.zarr')
    average_zarr = os.path.join(dir_output, 'average_kidney_mask.zarr')
    dir_png = os.path.join(dir_output, 'average_kidney_mask_imgs')
    imgs_mp4 = os.path.join(dir_output, 'average_kidney_mask_movie.mp4')

    logging.info("Stage 2. Computing feature vectors")
    # pipe.adjust_workers(client, min_ram_per_worker=2)
    ssa.features_from_dataset(
        features_from_mask=ssa.sdf_ft.features_from_mask, 
        masks_zarr_path=zarr_path, 
        output_zarr_path=features_zarr, 
        order=19,
    )

    logging.info("Stage 2. Computing average")
    z = zarr.open(features_zarr, mode='r')
    mean_feat = np.mean(z['features'][:], axis=0)
    mean_mask = ssa.sdf_ft.mask_from_features(
        mean_feat, 
        z.attrs['original_shape'], 
        **z.attrs['kwargs'],
    )
    # Save average mask as z-array
    zarr.save_group(
        average_zarr, 
        masks=mean_mask[np.newaxis, ...], 
        labels=np.array(['iBEAt_000-K']),
    )

    logging.info("Stage 2. Display average")
    # pipe.adjust_workers(client, min_ram_per_worker=16)
    display.recon(average_zarr, dir_png, imgs_mp4)
    
    logging.info("Stage 2 --- Computation of average control kidney completed ---")
                 

if __name__ == '__main__':

    BUILD = r"C:\Users\md1spsx\Documents\Data\iBEAt_Build"
    # pipe.run_client_stage(run, BUILD, PIPELINE, __file__, min_ram_per_worker=16)
    pipe.run_stage(run, BUILD, PIPELINE, __file__)