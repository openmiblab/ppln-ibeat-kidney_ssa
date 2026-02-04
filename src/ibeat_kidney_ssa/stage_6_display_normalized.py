import os
import logging
import argparse

from dbdicom import npz
from miblab_plot import pvplot, mp4

from ibeat_kidney_ssa.utils import data, pipe

PIPELINE = 'kidney_ssa'

def run(build):
    dir_input = os.path.join(build, PIPELINE, 'stage_5_normalize')
    dir_output = pipe.setup_stage(build, PIPELINE, __file__)

    dir_png = os.path.join(dir_output, 'images')
    os.makedirs(dir_png, exist_ok=True)

    # Sort kidneys and read labels
    kidney_masks = [k for k in npz.series(dir_input) if k[2][0] in ['Visit1', 'Baseline']]
    kidney_masks_sorted, kidney_labels_sorted = data.sort_kidney_masks(kidney_masks)  

    try:
        ncols, nrows = 16, 8
        pvplot.rotating_mosaics_npz(dir_png, kidney_masks_sorted, kidney_labels_sorted, chunksize=ncols * nrows, nviews=25, columns=ncols, rows=nrows)
        mp4.images_to_video(dir_png, os.path.join(dir_output, 'normalized_kidneys.mp4'), fps=16)
        logging.info(f"Stage 6: Successfully built rotating mosaics.")
    except:
        logging.exception(f"Stage 6: Error building rotating mosaics.")




if __name__ == '__main__':

    BUILD = r"C:\Users\md1spsx\Documents\Data\iBEAt_Build"

    parser = argparse.ArgumentParser()
    parser.add_argument("--build", type=str, default=BUILD, help="Build folder")
    args = parser.parse_args()

    run(args.build)