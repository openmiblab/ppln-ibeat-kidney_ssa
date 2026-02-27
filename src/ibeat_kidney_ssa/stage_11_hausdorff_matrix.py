import os
import logging

import pandas as pd
import dask.array as da
import miblab_ssa as ssa

from miblab import pipe

PIPELINE = 'kidney_ssa'

def run(build):

    logging.info("Stage 11 --- Computing Hausdorff matrix ---")
    dir_input = os.path.join(build, PIPELINE, 'stage_9_stack_normalized')
    dir_output = pipe.stage_output_dir(build, PIPELINE, __file__)

    # Define the input path
    zarr_path = os.path.join(dir_input, 'normalized_kidney_masks.zarr')

    # 3. Compute Hausdorff matrix
    matrix = ssa.hausdorff_matrix_zarr(zarr_path)
    logging.info(f"Stage 11: Finished computing Hausdorff matrix")

    # 4. Save as csv with column and row labels
    file = os.path.join(dir_output, f'normalized_kidney_hausdorff.csv')
    labels = da.from_zarr(zarr_path, component='labels')
    df = pd.DataFrame(matrix, columns=labels, index=labels)
    df.to_csv(file)
    logging.info(f"Stage 11: Saved Hausdorff matrix")



if __name__ == '__main__':

    BUILD = r"C:\Users\md1spsx\Documents\Data\iBEAt_Build"
    pipe.run_dask_script(run, BUILD, PIPELINE, min_ram_per_worker = 2.0)
