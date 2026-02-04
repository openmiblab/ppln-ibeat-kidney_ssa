import os
import logging
import argparse
from pathlib import Path
import pandas as pd
import dask.array as da

from ibeat_kidney_ssa.utils import metrics, pipe

PIPELINE = 'kidney_ssa'


def run(build):
    # Set the stage
    dir_input = os.path.join(build, PIPELINE, 'stage_9_stack_normalized')
    dir_output = pipe.setup_stage(build, PIPELINE, __file__)

    logging.info("Stage 10 --- Computing dice matrix ---")

    # Define the input path
    zarr_path = os.path.join(dir_input, 'normalized_kidney_masks.zarr')

    # 3. Compute dice matrix
    dice_matrix = metrics.dice_matrix_zarr(zarr_path)
    logging.info(f"Stage 10: Finished computing dice matrix")

    # 4. Save as csv with column and row labels
    file = os.path.join(dir_output, f'normalized_kidney_dice.csv')
    labels = da.from_zarr(zarr_path, component='labels')
    df = pd.DataFrame(dice_matrix, columns=labels, index=labels)
    df.to_csv(file)
    logging.info(f"Stage 10: Saved dice matrix to csv")


if __name__ == '__main__':

    BUILD = r"C:\Users\md1spsx\Documents\Data\iBEAt_Build"

    parser = argparse.ArgumentParser()
    parser.add_argument("--build", type=str, default=BUILD, help="Build folder")
    args = parser.parse_args()

    run(args.build)