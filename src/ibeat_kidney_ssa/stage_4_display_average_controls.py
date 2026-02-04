import os
import logging
import argparse

from miblab_plot import pvplot, mp4
from ibeat_kidney_ssa.utils import pipe

PIPELINE = 'kidney_ssa'

def run(build):
    dir_input = os.path.join(build, PIPELINE, 'stage_3_average_controls')
    dir_output = pipe.setup_stage(build, PIPELINE, __file__)

    dir_png = os.path.join(dir_output, 'images')
    os.makedirs(dir_png, exist_ok=True)

    # Sort kidneys and read labels
    mean_series = [dir_input, 'MIBL_C01', ('Visit1', 0), ('normalized_kidney_avr', 0)]
    mean_label = 'MIBL_C01'
    
    # Plot
    pvplot.rotating_mosaics_npz(dir_png, [mean_series], [mean_label], nviews=36)
    mp4.images_to_video(dir_png, os.path.join(dir_output, 'animation.mp4'), fps=16)
    logging.info(f"Successfully built stage 4 rotating mosaics.")




if __name__ == '__main__':

    BUILD = r"C:\Users\md1spsx\Documents\Data\iBEAt_Build"

    parser = argparse.ArgumentParser()
    parser.add_argument("--build", type=str, default=BUILD, help="Build folder")
    args = parser.parse_args()

    run(args.build)