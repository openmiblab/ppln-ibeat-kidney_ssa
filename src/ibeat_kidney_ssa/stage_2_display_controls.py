import os
import logging
import argparse

from dbdicom import npz

from ibeat_kidney_ssa.utils import pvplot, data, movie


def run(build):
    dir_input = os.path.join(build, 'kidney_ssa', 'stage_1_normalize_controls')
    dir_output = os.path.join(build, 'kidney_ssa', 'stage_2_display_controls')

    logging.basicConfig(
        filename=f"{dir_output}.log",
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    dir_png = os.path.join(dir_output, 'images')
    os.makedirs(dir_png, exist_ok=True)

    # Sort kidneys and read labels
    kidney_masks = [k for k in npz.series(dir_input) if k[2][0] == 'Visit1']
    kidney_masks_sorted, kidney_labels_sorted = data.sort_kidney_masks(kidney_masks) 
    
    # Plot
    pvplot.rotating_mosaics_npz(dir_png, kidney_masks_sorted, kidney_labels_sorted, nviews=25)
    movie.images_to_video(dir_png, os.path.join(dir_output, 'animation.mp4'), fps=16)
    logging.info(f"Stage 2: successfully built rotating mosaics.")




if __name__ == '__main__':

    BUILD = r"C:\Users\md1spsx\Documents\Data\iBEAt_Build"

    parser = argparse.ArgumentParser()
    parser.add_argument("--build", type=str, default=BUILD, help="Build folder")
    args = parser.parse_args()

    run(args.build)