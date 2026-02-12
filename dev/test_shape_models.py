import os
import time
import dbdicom as db
import miblab_ssa as ssa
from miblab_plot import gui



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



def display_two_kidneys(build):

    dir_input = os.path.join(build, 'kidney_ssa', 'stage_5_normalize')

    patient = "3128_072" # "7128_090"
    study_desc = 'Baseline'
    series_desc = 'normalized_kidney_right'

    series = [dir_input, patient, (study_desc, 0), (series_desc, 0)]
    vol = db.npz.volume(series)
    mask = vol.values.astype(bool)
    spacing = vol.spacing

    t0 = time.perf_counter()
    # mask_rec = ssa.sdf_cheby.smooth_mask(mask, order=32) # 27: n=4060, dice 0.973, 14s
    mask_rec = ssa.sdf_surfleg.smooth_mask(mask, order=27)  # : n=4060, dice 0.973, 10s
    # mask_rec = sdf_ft_simple.smooth_mask(mask, order=16) # 16: n=4098, dice 0.972, 5.5s
    # mask_rec = ssa.sdf_ft.smooth_mask(mask, order=32) # 19: n=4019, dice 0.972, 5.1s
    # mask_rec = ssa.pdm.smooth_mask(mask, subdivisions=2) # 4, n=1926
    # mask_rec = ssa.sdf_wvlt.smooth_mask(mask, min_level=3, order=9) # dice 0.993, 11MB
    # mask_rec = ssa.sdf_wvlt.smooth_mask(mask, min_level=1, order=7) # dice 0.984, 9.7MB
    # mask_rec = ssa.sdf_wvlt.smooth_mask(mask, min_level=1, order=8) # dice 0.992, 70MB
    # mask_rec = ssa.sdf_rbf.smooth_mask(mask, grid_size=16, epsilon=4)  # : n=4096, dice 0.95, 16s
    
    # mask_rec = ssa.sdf_zernike.smooth_mask(mask, order=27)  # : n=4060, dice 0.973, 10s
    # mask_rec = ssa.sdf_surfleg.smooth_mask(mask, order=27)  # : n=4060, dice 0.981, 20s
    # mask_rec = ssa.sdf_spline.smooth_mask(mask, n_grid=16) # : n=4096, dice 0.982, 8.3s

    t1 = time.perf_counter()
    print(f"Computation time: {t1 - t0:.6f} s")

    dice = ssa.dice_coefficient(mask, mask_rec)
    print(f"Dice coefficient: {dice}")

    gui.display_two_kidneys(
        mask, mask_rec,
        kidney1_voxel_size=spacing, kidney2_voxel_size=spacing,
        title1='Original', title2='Reconstructed')
    


if __name__ == '__main__':
    BUILD = r"C:\Users\md1spsx\Documents\Data\iBEAt_Build"
    display_two_kidneys(BUILD)
