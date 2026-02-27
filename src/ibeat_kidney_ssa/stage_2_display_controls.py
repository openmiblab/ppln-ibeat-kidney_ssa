import os
import logging
import argparse

from dbdicom import npz
from miblab_plot import pvplot, mp4
from miblab import pipe

from ibeat_kidney_ssa.utils import data

PIPELINE = 'kidney_ssa'

def run(build):

    logging.info("Stage 2 --- Display controls ---")
    dir_input = os.path.join(build, 'kidney_ssa', 'stage_1_normalize_controls')
    dir_output = pipe.stage_output_dir(build, PIPELINE, __file__)
    dir_png = os.path.join(dir_output, 'images')
    os.makedirs(dir_png, exist_ok=True)

    # Sort kidneys and read labels
    kidney_masks = [k for k in npz.series(dir_input) if k[2][0] == 'Visit1']
    kidney_masks_sorted, kidney_labels_sorted = data.sort_kidney_masks(kidney_masks) 
    
    # Plot
    pvplot.rotating_mosaics_npz(dir_png, kidney_masks_sorted, kidney_labels_sorted, nviews=25)
    mp4.images_to_video(dir_png, os.path.join(dir_output, 'animation.mp4'), fps=16)

    logging.info("Stage 2. Successfully built rotating mosaics")




if __name__ == '__main__':

    BUILD = r"C:\Users\md1spsx\Documents\Data\iBEAt_Build"
    pipe.run_script(run, BUILD, PIPELINE)