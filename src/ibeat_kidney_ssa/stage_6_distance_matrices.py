import os
import logging
import pandas as pd
import dask.array as da
import miblab_ssa as ssa
from miblab import pipe

PIPELINE = 'kidney_ssa'


def run(build, logfile):

    logging.info("Stage 6 --- Computing distance matrices ---")
    dir_input = os.path.join(build, PIPELINE, 'stage_3_normalize')
    dir_output = pipe.stage_output_dir(build, PIPELINE, __file__)

    # Inputs
    zarr_path = os.path.join(dir_input, 'normalized_kidney_masks.zarr')
    labels = da.from_zarr(zarr_path, component='labels')

    # Outputs
    dice_csv = os.path.join(dir_output, f'normalized_kidney_dice.csv')
    haus_csv = os.path.join(dir_output, f'normalized_kidney_hausdorff.csv')

    logging.info(f"Stage 6: Computing dice matrix")
    # dice_matrix = ssa.dice_matrix(zarr_path)
    # pd.DataFrame(dice_matrix, columns=labels, index=labels).to_csv(dice_csv)
    # logging.info(f"Stage 6: Saved dice matrix to csv")

    logging.info(f"Stage 6: Computing Hausdorff matrix")
    haus_matrix = ssa.hausdorff_matrix(zarr_path)
    pd.DataFrame(haus_matrix, columns=labels, index=labels).to_csv(haus_csv)
    logging.info(f"Stage 6: Saved Hausdorff matrix")


if __name__ == '__main__':

    BUILD = r"C:\Users\md1spsx\Documents\Data\iBEAt_Build"
    pipe.run_stage(run, BUILD, PIPELINE, __file__)