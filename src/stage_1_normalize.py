import os
import logging
import argparse

from tqdm import tqdm
import dbdicom as db
import vreg

from utils import normalize



def run(build):

    build_path = os.path.join(build, 'kidney-ssa')
    os.makedirs(build_path)

    logging.basicConfig(
        filename=os.path.join(build, 'kidney-ssa', 'stage_1_normalize.log'),
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    dir_masks = os.path.join(build, 'kidney-shape', 'stage_3_edit')
    dir_output = os.path.join(build, 'kidney-ssa', 'stage_1_normalize')
    os.makedirs(dir_output, exist_ok=True)

    for database in ['Controls', 'Patients']:
        db_masks = os.path.join(dir_masks, database) 
        db_normalize = os.path.join(dir_output, database) 
        run_db(db_masks, db_normalize)


def run_db(db_masks, db_normalize):

    kidney_labels = db.series(db_masks)
    for kidney_label in tqdm(kidney_labels, desc=f'Normalizing kidneys..'):
        study = [db_normalize] + kidney_label[1:3]

        # Define outputs
        lk_series = study + [('normalized_kidney_left', 0)]
        rk_series = study + [('normalized_kidney_right', 0)]
        lk_exists = db.exists(lk_series)
        rk_exists = db.exists(rk_series)

        try:
        
            # Read data
            if (not rk_exists) or (not lk_exists):
                vol = db.volume(kidney_label, verbose=0)

            # Compute right kidney
            if not rk_exists:
                rk_mask = vol.values == 2
                rk_mask_norm, _ = normalize.normalize_kidney_mask(rk_mask, vol.spacing, 'right')
                rk_vol_norm = vreg.volume(rk_mask_norm.astype(int))
                db.write_volume(rk_vol_norm, rk_series, ref=kidney_label, verbose=0)

            # Compute left kidney
            if not lk_exists:
                lk_mask = vol.values == 1
                lk_mask_norm, _ = normalize.normalize_kidney_mask(lk_mask, vol.spacing, 'left')
                lk_vol_norm = vreg.volume(lk_mask_norm.astype(int))
                db.write_volume(lk_vol_norm, lk_series, ref=kidney_label, verbose=0)

            logging.info(f"Successfully normalized kidneys in study: {study[1:]}")

        except:

            logging.exception(f"Error normalizing kidneys in study: {study[1:]}")




if __name__ == '__main__':

    BUILD = r"C:\Users\md1spsx\Documents\Data\iBEAt_Build"

    parser = argparse.ArgumentParser()
    parser.add_argument("--build", type=str, default=BUILD, help="Build folder")
    args = parser.parse_args()

    run(args.build)
