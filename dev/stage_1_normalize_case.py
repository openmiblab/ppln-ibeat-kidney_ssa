import os

import dbdicom as db
import numpy as np

from ibeat_kidney_ssa.utils import normalize, render



# A: "7128_035_R",
#     "7128_090_R",
#     "5128_036_L",
#     "7128_071_R",
#     "1128_025_L",

# B: "3128_072_R",
#     "2128_008_L",
#     "4128_035_L",
#     "5128_012_L",
#     "3128_107_R",

# A  = [0,2]
# B  = [1,3]



def display_normalized(build):

    _display_normalized(build, "7128_035", 'R')
    _display_normalized(build, "3128_072", 'R')



def _display_normalized(build, patient_id, kidney):

    dir_input = os.path.join(build, 'kidney_shape', 'stage_3_edit', 'Patients')

    study_desc = 'Baseline'
    series_desc = 'kidney_masks'

    series = [dir_input, patient_id, study_desc, (series_desc, 0)]
    vol = db.volume(series).to_right_handed()
    voxel_size = vol.spacing
    spacing_norm = 1.0
    voxel_size_norm = 3 * [spacing_norm]

    if kidney == 'R':
        rk_mask = vol.values == 2
        rk_mask_norm, _ = normalize.normalize_kidney_mask(rk_mask, voxel_size, 'right')
        return render.display_kidney_normalization(rk_mask, rk_mask_norm, voxel_size, voxel_size_norm, title='Right kidney')

    if kidney == 'L':
        lk_mask = vol.values == 1
        lk_mask_norm, _ = normalize.normalize_kidney_mask(lk_mask, voxel_size, 'left')
        return render.display_kidney_normalization(np.flip(lk_mask, 0), lk_mask_norm, voxel_size, voxel_size_norm, title='Left kidney flipped')
    

def display_two_kidneys(build):

    dir_input = os.path.join(build, 'kidney_shape', 'stage_3_edit', 'Patients')

    study_desc = 'Baseline'
    series_desc = 'kidney_masks'

    # pt, kd = "7128_035", 'R'
    # pt, kd = "7128_090", 'R'
    pt, kd = "7128_090", 'R'
    # pt, kd = "7128_035", 'R'
    series = [dir_input, pt, study_desc, (series_desc, 0)]
    vol = db.volume(series).to_right_handed()
    label = 2 if kd == 'R' else 1
    kidney1 = vol.values == label
    voxel_size_1 = vol.spacing
    title1=f"{pt}_{kd}"

    # pt, kd =  "3128_072", 'R'
    # pt, kd =  "2128_008", 'L'
    # pt, kd = "7128_090", 'L'
    pt, kd = "7128_035", 'L'
    series = [dir_input, pt, study_desc, (series_desc, 0)]
    vol = db.volume(series).to_right_handed()
    label = 2 if kd == 'R' else 1
    kidney2 = vol.values == label
    voxel_size_2 = vol.spacing
    title2=f"{pt}_{kd}"

    render.display_two_kidneys(kidney1, kidney2,
                        kidney1_voxel_size=voxel_size_1,
                        kidney2_voxel_size=voxel_size_2,
                        title1=title1, title2=title2)
    

def display_two_normalized_kidneys(build):

    dir_input = os.path.join(build, 'kidney_shape', 'stage_3_edit', 'Patients')

    study_desc = 'Baseline'
    series_desc = 'kidney_masks'

    # pt, kd = "7128_035", 'R'
    # pt, kd = "7128_090", 'R'
    pt, kd = "7128_090", 'R'
    # pt, kd = "7128_035", 'R'
    series = [dir_input, pt, study_desc, (series_desc, 0)]
    vol = db.volume(series).to_right_handed()
    label = 2 if kd == 'R' else 1
    kidney1 = vol.values == label
    voxel_size_1 = vol.spacing
    side = 'right' if kd == 'R' else 'left'
    kidney1_norm, _ = normalize.normalize_kidney_mask(kidney1, voxel_size_1, side)
    
    # pt, kd =  "3128_072", 'R'
    # pt, kd =  "2128_008", 'L'
    # pt, kd = "7128_090", 'L'
    pt, kd = "7128_035", 'L'
    series = [dir_input, pt, study_desc, (series_desc, 0)]
    vol = db.volume(series).to_right_handed()
    label = 2 if kd == 'R' else 1
    kidney2 = vol.values == label
    voxel_size_2 = vol.spacing
    side = 'right' if kd == 'R' else 'left'
    kidney2_norm, _ = normalize.normalize_kidney_mask(kidney2, voxel_size_2, side)

    render.display_two_normalized_kidneys(kidney1_norm, kidney2_norm,
                        title='Two normalized kidneys')


if __name__ == '__main__':

    BUILD = r"C:\Users\md1spsx\Documents\Data\iBEAt_Build"

    # display_normalized(BUILD)
    display_two_kidneys(BUILD)
    display_two_normalized_kidneys(BUILD)
