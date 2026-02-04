import os
import logging
import argparse

from tqdm import tqdm
from dbdicom import npz
import zarr

from ibeat_kidney_ssa.utils import data, pipe

PIPELINE = 'kidney_ssa'


def run(build):
    # Set the stage
    dir_input = os.path.join(build, PIPELINE, 'stage_5_normalize')
    dir_output = pipe.setup_stage(build, PIPELINE, __file__)

    logging.info("Stage 9 --- Stacking normalized kidneys ---")

    # 1. Get all baseline kidneys and labels in alphabetic order
    kidney_masks = [k for k in npz.series(dir_input) if k[2][0] in ['Visit1', 'Baseline']]
    kidney_masks_sorted, kidney_labels_sorted = data.sort_kidney_masks(kidney_masks) 

    # Define the output path (Zarr saves as a directory, not a single file)
    zarr_path = os.path.join(dir_output, 'normalized_kidney_masks.zarr')

    # 1. Open a Zarr group on disk
    # mode='w' creates a new store, overwriting if it exists
    store = zarr.DirectoryStore(zarr_path)
    root = zarr.group(store=store, overwrite=True)

    # 2. Get shapes to allocate
    n_kidneys = len(kidney_masks_sorted)
    shape = npz.volume(kidney_masks_sorted[0]).shape

    # 3. Initialize the array on disk (No RAM usage yet)
    # chunks=(1, *shape) optimizes for reading/writing one whole kidney at a time
    z_masks = root.create_dataset(
        'masks', 
        shape=(n_kidneys,) + shape, 
        chunks=(1,) + shape, 
        dtype=bool,
        compressor=zarr.Blosc(cname='zstd', clevel=3, shuffle=2) # Optional: Good compression
    )

    # 4. Save the labels immediately
    # (Assuming labels are small strings/ints, we don't need to chunk them specifically)
    root.array('labels', kidney_labels_sorted)

    # 4. Iterate and Write to Disk
    for i, mask in tqdm(enumerate(kidney_masks_sorted), desc="Stage 9: Loading masks"):
        # Load ONE mask into memory
        vol_data = npz.volume(mask).values.astype(bool)
        
        # Write it to disk immediately. 
        # Zarr handles the compression and saving behind the scenes.
        z_masks[i] = vol_data 
        
        # vol_data is now freed from RAM for the next iteration

    logging.info(f"Stage 9: Saved mask matrix")



if __name__ == '__main__':

    BUILD = r"C:\Users\md1spsx\Documents\Data\iBEAt_Build"

    parser = argparse.ArgumentParser()
    parser.add_argument("--build", type=str, default=BUILD, help="Build folder")
    args = parser.parse_args()

    run(args.build)