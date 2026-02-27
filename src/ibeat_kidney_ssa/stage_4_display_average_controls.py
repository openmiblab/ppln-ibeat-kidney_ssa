import os
import logging
import argparse

from miblab_plot import pvplot, mp4
from miblab import pipe

PIPELINE = 'kidney_ssa'

def run(build):

    logging.info("Stage 4 --- Display of average control kidney ---")
    dir_input = os.path.join(build, PIPELINE, 'stage_3_average_controls')
    dir_output = pipe.stage_output_dir(build, PIPELINE, __file__)
    dir_png = os.path.join(dir_output, 'images')
    os.makedirs(dir_png, exist_ok=True)

    # Sort kidneys and read labels
    mean_series = [dir_input, 'iBEAt_000', ('Visit1', 0), ('average_normalized_control_kidney', 0)]
    mean_label = mean_series[1]
    
    logging.info(f"Stage 4. Plotting kidney.")
    pvplot.rotating_mosaics_npz(dir_png, [mean_series], [mean_label], nviews=36)
    mp4.images_to_video(dir_png, os.path.join(dir_output, 'average_normalized_control_kidney.mp4'), fps=16)
    
    logging.info(f"Stage 4. Successfully built rotating mosaics.")


if __name__ == '__main__':

    BUILD = r"C:\Users\md1spsx\Documents\Data\iBEAt_Build"
    pipe.run_script(run, BUILD, PIPELINE)