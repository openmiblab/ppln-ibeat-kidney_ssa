import os
import logging

import numpy as np
import zarr

import miblab_ssa as ssa
from miblab_plot import pvplot, mp4

from ibeat_kidney_ssa.utils import pipe

PIPELINE = 'kidney_ssa'

def run(build, client):

    logging.info("Stage 16 --- Wavelet PCA ---")
    dir_input = os.path.join(build, PIPELINE, 'stage_9_stack_normalized')
    dir_output = pipe.stage_output_dir(build, PIPELINE, __file__)

    # Input data - stack of normalized masks
    masks_path = os.path.join(dir_input, 'normalized_kidney_masks.zarr')

    # Outputs
    model = 'wavelet'
    feature_path = os.path.join(dir_output, f'{model}_features.zarr')
    pca_path = os.path.join(dir_output, f"{model}_components.zarr")
    scores_path = os.path.join(dir_output, f"{model}_scores.zarr")
    modes_path = os.path.join(dir_output, f"{model}_modes.zarr")
    dir_png = os.path.join(dir_output, f'images')
    movie_file = os.path.join(dir_output, f"{model}_modes.mp4")
    cumulative_dice = os.path.join(dir_output, f"{model}_performance_cumulative_dice.csv")
    marginal_dice = os.path.join(dir_output, f"{model}_performance_marginal_dice.csv")
    pca_performance_plot = os.path.join(dir_output, f"{model}_performance_plot.png")

    # Variables
    kwargs = {
        "min_level": 1,
        "order": 7,
    }

    logging.info(f"Stage 16.1 Building {model} feature matrix")
    pipe.adjust_workers(client, 18)
    ssa.features_from_dataset_zarr(
        ssa.sdf_wvlt.features_from_mask, 
        masks_path, 
        feature_path, 
        **kwargs, 
    )

    logging.info("Stage 16.2 Computing PCA")
    pipe.adjust_workers(client, 20)
    ssa.direct_pca_from_features_zarr(feature_path, pca_path)
    ssa.plot_pca_performance(pca_path, pca_performance_plot, n_components=200)

    logging.info("Stage 16.3 Compute scores")
    pipe.adjust_workers(client, 8)
    ssa.scores_from_features_zarr(feature_path, pca_path, scores_path)

    logging.info("Stage 16.4 Compute principal modes")
    pipe.adjust_workers(client, 8)
    ssa.modes_from_pca_zarr(
        ssa.sdf_wvlt.mask_from_features, 
        pca_path, 
        modes_path, 
        n_components=8, 
        max_coeff=10,
    )

    logging.info("Stage 16.5 Measure model performance")
    pipe.adjust_workers(client, 16)
    ssa.pca_performance(
        ssa.sdf_wvlt.mask_from_features, 
        pca_path, 
        scores_path, 
        masks_path, 
        marginal_dice, 
        cumulative_dice, 
        n_components=20,
    )
    ssa.plot_pca_performance(pca_path, pca_performance_plot, marginal_dice, cumulative_dice)

    logging.info("Stage 16.6 Display principal modes")
    pipe.adjust_workers(client, 16)
    display_modes(modes_path, dir_png, movie_file)

    logging.info("Stage 16: Wavelet PCA successfully completed.")


def display_modes(modes_path, dir_png, movie_file):
    modes = zarr.open(modes_path, mode='r')
    masks = modes['modes'][:]
    n_comp = masks.shape[1]
    coeffs = np.array(modes.attrs['coeffs'][:])
    labels = np.array([[f"C{y}: {round(x, 1)} x sd" for y in range(n_comp)] for x in coeffs])
    pvplot.rotating_masks_grid(dir_png, masks, labels, nviews=25)
    mp4.images_to_video(dir_png, movie_file, fps=16)



if __name__ == '__main__':

    BUILD = r"C:\Users\md1spsx\Documents\Data\iBEAt_Build"
    pipe.run_dask_script(run, BUILD, PIPELINE, min_ram_per_worker=8)