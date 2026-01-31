import logging
import os
import argparse


import ibeat_kidney_ssa as ppln


def run(proj):

    dir_output = os.path.join(proj, 'kidney_ssa')
    os.makedirs(dir_output, exist_ok=True)

    logging.basicConfig(
        filename=f"{dir_output}.log",
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    ppln.stage_1_normalize.run(BUILD)
    ppln.stage_2_display.run(BUILD)
    ppln.stage_3_normalize_npz.run(BUILD)
    ppln.stage_4_display.run(BUILD)
    ppln.stage_5_features.run(BUILD)
    ppln.stage_6_dice.run(BUILD)


if __name__=='__main__':

    BUILD = r"C:\Users\md1spsx\Documents\Data\iBEAt_Build"

    parser = argparse.ArgumentParser()
    parser.add_argument("--build", type=str, default=BUILD, help="Build folder")
    args = parser.parse_args()

    run(args.build)

