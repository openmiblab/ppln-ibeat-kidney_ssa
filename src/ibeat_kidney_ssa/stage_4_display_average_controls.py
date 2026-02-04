import os
import logging
import argparse

from ibeat_kidney_ssa.utils import pvplot, movie


def run(build):
    dir_input = os.path.join(build, 'kidney_ssa', 'stage_3_average_controls')
    dir_output = os.path.join(build, 'kidney_ssa', 'stage_4_display_average_controls')

    logging.basicConfig(
        filename=f"{dir_output}.log",
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    dir_png = os.path.join(dir_output, 'images')
    os.makedirs(dir_png, exist_ok=True)

    # Sort kidneys and read labels
    mean_series = [dir_input, 'MIBL_C01', ('Visit1', 0), ('normalized_kidney_avr', 0)]
    mean_label = 'MIBL_C01'
    
    # Plot
    pvplot.rotating_mosaics_npz(dir_png, [mean_series], [mean_label], nviews=36)
    movie.images_to_video(dir_png, os.path.join(dir_output, 'animation.mp4'), fps=16)
    logging.info(f"Successfully built stage 4 rotating mosaics.")




if __name__ == '__main__':

    BUILD = r"C:\Users\md1spsx\Documents\Data\iBEAt_Build"

    parser = argparse.ArgumentParser()
    parser.add_argument("--build", type=str, default=BUILD, help="Build folder")
    args = parser.parse_args()

    run(args.build)