import os
import logging
import argparse

from dbdicom import npz
import miblab_ssa as ssa

from ibeat_kidney_ssa.utils import data, pipe

PIPELINE = 'kidney_ssa'


def run(build):
    
    logging.info("Stage 9 --- Stacking normalized kidneys ---")
    dir_input = os.path.join(build, PIPELINE, 'stage_5_normalize')
    dir_output = pipe.stage_output_dir(build, PIPELINE, __file__)

    # 1. Get all baseline kidneys and labels in alphabetic order
    kidney_masks = [k for k in npz.series(dir_input) if k[2][0] in ['Visit1', 'Baseline']]
    kidney_masks_sorted, kidney_labels_sorted = data.sort_kidney_masks(kidney_masks) 
    kidney_masks_sorted = [npz.file(mask) for mask in kidney_masks_sorted]

    # Save masks as zarr
    zarr_path = os.path.join(dir_output, 'normalized_kidney_masks.zarr')
    ssa.save_masks_as_zarr(zarr_path, kidney_masks_sorted, kidney_labels_sorted)
        
    logging.info(f"Stage 9: Saved stacked mask array")



if __name__ == '__main__':

    BUILD = r"C:\Users\md1spsx\Documents\Data\iBEAt_Build"
    pipe.run_dask_script(run, BUILD, PIPELINE)