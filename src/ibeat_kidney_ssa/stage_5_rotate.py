import os
import logging
import argparse
from pathlib import Path

import dask
import vreg

from ibeat_kidney_ssa.utils import normalize, data

def run(build):

    dir_input = os.path.join(build, 'kidney_ssa', 'stage_3_normalize_npz')
    dir_output = os.path.join(build, 'kidney_ssa', 'stage_5_rotate')
    os.makedirs(dir_output, exist_ok=True)

    logging.basicConfig(
        filename=f"{dir_output}.log",
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    mask_ref = os.path.join(
        dir_input, 
        "Patient__3128_C03", 
        "Study__1__Visit1", 
        "Series__1__normalized_kidney_left.npz",
    )
    kidney_masks = [f for f in Path(dir_input).rglob("*") if f.is_file()]
    if kidney_masks == []:
        logging.info(f"No kidney masks in {dir_input}")
        return
    # [_normalize_rotation(mask, mask_ref, dir_output) for mask in kidney_masks]
    tasks = [dask.delayed(_normalize_rotation)(mask, mask_ref, dir_output) for mask in kidney_masks]
    dask.compute(*tasks)
    logging.info(f"Successfully performed rotations: {dir_output}")


def _normalize_rotation(kidney_mask, kidney_mask_ref, db_output):
    # Define output folder and skip if already exists
    relpath = data.relpath_npz_dbfile(kidney_mask)
    output_file = os.path.join(db_output, relpath)
    if os.path.exists(output_file):
        return

    # Read the masks
    vol_ref = vreg.read_npz(kidney_mask_ref)
    mask_ref = vol_ref.values.astype(bool)
    mask = vreg.read_npz(kidney_mask).values.astype(bool)

    # Perform rotation to reference mask
    # TODO: This needs separating out - rotate + compute dice
    _, mask_rot = normalize.invariant_dice_coefficient(mask_ref, mask, angle_step=1, return_mask=True, verbose=0)

    # Save result
    vreg.volume(mask_rot, vol_ref.affine).write_npz(output_file)

    print(f"Successfully rotated {relpath}.")
    logging.info(f"Successfully rotated {relpath}.") 


if __name__ == '__main__':

    BUILD = r"C:\Users\md1spsx\Documents\Data\iBEAt_Build"

    parser = argparse.ArgumentParser()
    parser.add_argument("--build", type=str, default=BUILD, help="Build folder")
    args = parser.parse_args()

    run(args.build)
