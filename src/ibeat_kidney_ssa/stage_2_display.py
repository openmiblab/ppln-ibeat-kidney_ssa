import os
import logging
import argparse

import dbdicom as db

from ibeat_kidney_ssa.utils import data, pvplot


def run(build):

    # Set up folders
    dir_input = os.path.join(build, 'kidney_ssa', 'stage_1_normalize')
    dir_output = os.path.join(build, 'kidney_ssa', 'stage_2_display')
    logging.basicConfig(
        filename=f"{dir_output}.log",
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    os.makedirs(dir_output, exist_ok=True)

    # List kidney masks and select baselines
    kidney_masks = db.series(dir_input)
    kidney_masks = [k for k in kidney_masks if k[2][0] in ['Visit1', 'Baseline']]
    imagefile_front = os.path.join(dir_output, f"kidneys_stage_2_front.png")
    imagefile_top = os.path.join(dir_output, f"kidneys_stage_2_top.png")
    imagefile_side = os.path.join(dir_output, f"kidneys_stage_2_side.png")

    # Skip if empty or already done
    if kidney_masks == []:
        logging.info(f"No kidney masks in {dir_input}")
        return   
    
    # Sort kidneys and read labels
    kidney_masks_sorted = data.sort_kidney_series(kidney_masks)  
    kidney_labels = [data.label_db_series(s) for s in kidney_masks_sorted] 

    # Create mosaic and save
    pvplot.mosaic_masks_dcm(kidney_masks_sorted, imagefile_front, kidney_labels, view_vector=(1, 0, 0))
    pvplot.mosaic_masks_dcm(kidney_masks_sorted, imagefile_side, kidney_labels, view_vector=(0, -1, 0))
    pvplot.mosaic_masks_dcm(kidney_masks_sorted, imagefile_top, kidney_labels, view_vector=(0, 0, 1))
    print(f"Successfully built stage_2 mosaics.")



if __name__ == '__main__':

    BUILD = r"C:\Users\md1spsx\Documents\Data\iBEAt_Build"

    parser = argparse.ArgumentParser()
    parser.add_argument("--build", type=str, default=BUILD, help="Build folder")
    args = parser.parse_args()

    run(args.build)