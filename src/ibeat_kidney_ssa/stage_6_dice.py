import os
import logging
import argparse
import time
from pathlib import Path
import shutil
import json

import numpy as np
import pandas as pd
import dask
import vreg


from ibeat_kidney_ssa.utils import normalize, data


def run(build):

    dir_input = os.path.join(build, 'kidney_ssa', 'stage_3_normalize_npz')
    dir_output = os.path.join(build, 'kidney_ssa', 'stage_6_dice')
    os.makedirs(dir_output, exist_ok=True)

    logging.basicConfig(
        filename=f"{dir_output}.log",
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    logging.info("Computation of correlation matrices started")

    # Get all baseline kidneys in sorted order
    kidney_masks = [f for f in Path(dir_input).rglob("*") if f.is_file()]
    kidney_masks = [f for f in kidney_masks if ('Visit1' in str(f)) or ('Baseline' in str(f))]
    if kidney_masks == []:
        logging.info(f"No kidney masks in {dir_input}")
        return
    kidney_masks_sorted, kidney_labels_sorted = data.sort_kidney_npz(kidney_masks) 

    # Define tmp path for output of parallel threads
    tmp_path = os.path.join(dir_output, 'tmp')
    os.makedirs(tmp_path, exist_ok=True)

    # Chunk output to produce less and larger tasks, and less files
    # Otherwise dask takes too long to schedule
    chunk_size = 10 # TODO change to 1000 for the full dataset
    n = len(kidney_masks_sorted)
    # Build a list of all index pairs in the sroted list that need computing
    # Since the matrix is symmetric only half needs to be computed
    pairs = [(i, j) for i in range(n) for j in range(i, n)]
    # Split the list of index pairs up into chunks
    chunks = [pairs[i:i+chunk_size] for i in range(0, len(pairs), chunk_size)]

    # Compute dice scores for each chunk in parallel
    logging.info("Started scheduling tasks..")
    tasks = [dask.delayed(correlation_matrix_chunk)(kidney_masks_sorted, tmp_path, chunk_idx, chunk) for chunk_idx, chunk in enumerate(chunks)]

    logging.info('Started computing tasks..')
    dask.compute(*tasks)

    # Gather up all the chunks to build one matrix
    build_dice_matrix(dir_output, kidney_labels_sorted)

    # Remove tmp folder
    shutil.rmtree(tmp_path)

    logging.info(f"Computation of correlation matrices completed.")


def correlation_matrix_chunk(kidney_masks_sorted, tmp_path, chunk_idx, pairs):

    filepath = os.path.join(tmp_path, f"chunk_{chunk_idx}.json")
    if os.path.exists(filepath):
        return

    data = []
    for (i,j) in pairs:

        # Load masks
        mask_i = vreg.read_npz(kidney_masks_sorted[i]).values.astype(bool)
        mask_j = vreg.read_npz(kidney_masks_sorted[j]).values.astype(bool)
        
        # Compute dice
        dice_ij = normalize.dice_coefficient(mask_i, mask_j)

        # Add to lists
        data.append(
            {
                'inds': (i,j),
                'dice': dice_ij,
            }
        )

    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)


def build_dice_matrix(db_output, labels_sorted):

    # Read the files in the tmp folder and combine in one array
    tmp_path = os.path.join(db_output, 'tmp')
    chunks = [f for f in Path(tmp_path).rglob("*") if f.is_file()]

    # Compile all results in a single matrix
    n = len(labels_sorted)
    dice_arr = np.zeros((n, n))
    for chunk in chunks:
        with open(chunk, "r") as f:
            data = json.load(f)
        for d in data:
            i, j = d['inds']
            dice_arr[i, j] = d['dice']
            dice_arr[j, i] = d['dice']

    # Save arrays as csv
    file = os.path.join(db_output, f'normalized_kidney_dice.csv')
    df = pd.DataFrame(dice_arr, columns=labels_sorted, index=labels_sorted)
    df.to_csv(file)


if __name__ == '__main__':

    BUILD = r"C:\Users\md1spsx\Documents\Data\iBEAt_Build"

    parser = argparse.ArgumentParser()
    parser.add_argument("--build", type=str, default=BUILD, help="Build folder")
    args = parser.parse_args()

    run(args.build)
