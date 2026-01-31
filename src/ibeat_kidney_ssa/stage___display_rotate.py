import os
import logging
import json
import argparse
from pathlib import Path

from ibeat_kidney_ssa.utils import pvplot, data


def run(build):
    dir_input = os.path.join(build, 'kidney_ssa', 'stage_5_rotate')
    dir_output = os.path.join(build, 'kidney_ssa', 'stage_6_display')
    logging.basicConfig(
        filename=f"{dir_output}.log",
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    # Skip if the mosaic exists
    imagefile = os.path.join(dir_output, f"kidneys_stage_6.png")
    if os.path.exists(imagefile):
        logging.info(f"Mosaic already exists at: {imagefile}")
        return  
    os.makedirs(dir_output, exist_ok=True)

    # Get baseline masks
    kidney_masks = [f for f in Path(dir_input).rglob("*") if f.is_file()]
    kidney_masks = [f for f in kidney_masks if ('Visit1' in str(f)) or ('Baseline' in str(f))]
    if kidney_masks == []:
        logging.info(f"No kidney masks in {dir_input}")
        return
        
    # Sort baseline masks, get labels and plot 
    kidney_masks_sorted, kidney_labels_sorted = data.sort_kidney_npz(kidney_masks)  
    pvplot.mosaic_masks_npz(kidney_masks_sorted, imagefile, kidney_labels_sorted)
    logging.info(f"Successfully built mosaic at: {imagefile}")



if __name__ == '__main__':

    BUILD = r"C:\Users\md1spsx\Documents\Data\iBEAt_Build"

    parser = argparse.ArgumentParser()
    parser.add_argument("--build", type=str, default=BUILD, help="Build folder")
    args = parser.parse_args()

    run(args.build)