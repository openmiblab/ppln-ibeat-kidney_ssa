import os
import logging

import miblab_ssa as ssa
import miblab_ssa.sdf_ft as model

from ibeat_kidney_ssa.utils import pipe, display

PIPELINE = 'kidney_ssa'

def run(build, client):

    logging.info("Stage 12 --- Spectral PCA ---")
    dir_input = os.path.join(build, PIPELINE, 'stage_9_stack_normalized')
    dir_output = pipe.stage_output_dir(build, PIPELINE, __file__)

    # Input data
    masks = os.path.join(dir_input, 'normalized_kidney_masks.zarr')

    # Output - arrays
    features = os.path.join(dir_output, 'data_features.zarr')
    pca = os.path.join(dir_output, f"data_components.zarr")
    scores = os.path.join(dir_output, f"data_scores.zarr")
    feature_modes = os.path.join(dir_output, f"data_feature_modes.zarr")
    mask_modes = os.path.join(dir_output, f"data_mask_modes.zarr")
    feature_recon_err = os.path.join(dir_output, f"data_feature_recon_err.zarr")
    feature_recon = os.path.join(dir_output, f"data_feature_recon.zarr")
    mask_recon_err = os.path.join(dir_output, f"data_mask_recon_err.zarr")
    mask_recon = os.path.join(dir_output, f"data_mask_recon.zarr")

    # Output - tables
    cumulative_mse = os.path.join(dir_output, 'table_cumulative_mse.csv')
    marginal_mse = os.path.join(dir_output, 'table_marginal_mse.csv')

    # Output - images
    pca_performance = os.path.join(dir_output, 'imgs_performance.png')
    modes_png = os.path.join(dir_output, 'imgs_modes')
    recon_png = os.path.join(dir_output, 'imgs_recon')
    recon_err_png = os.path.join(dir_output, 'imgs_recon_err')

    # Output - movies
    modes_mp4 = os.path.join(dir_output, 'movie_modes.mp4')
    recon_mp4 = os.path.join(dir_output, 'movie_recon.mp4')
    recon_err_mp4 = os.path.join(dir_output, 'movie_recon_err.mp4')

    logging.info(f"Stage 12.1 Building feature matrix")
    ssa.features_from_dataset(
        model.features_from_mask, 
        masks, 
        features, 
        order = 19, # 4019 features
    )
    logging.info("Stage 12.2 Computing PCA")
    ssa.pca_from_features(features, pca)

    logging.info("Stage 12.3 Computing PCA scores")
    ssa.scores_from_features(features, pca, scores)

    logging.info("Stage 12.4 Computing PCA performance")
    ssa.pca_performance(
        pca, scores, 
        features, 
        marginal_mse, 
        cumulative_mse, 
        n_components=200,
    )
    ssa.plot_pca_performance(
        pca, 
        pca_performance, 
        marginal_mse, 
        cumulative_mse, 
        n_components=200,
    )
    logging.info("Stage 12.5 Computing reconstruction accuracy")
    labels = display.get_outlier_labels(cumulative_mse, n=8)
    ssa.cumulative_features_from_scores(
        pca, 
        scores, 
        feature_recon_err, 
        labels, 
        step_size=10, 
        max_components=100,
    )
    logging.info("Stage 12.6 Computing principal modes")
    ssa.modes_from_pca(
        pca, feature_modes, n_components=8, max_coeff=8,
    )
    logging.info("Stage 12.7 Computing reconstructions")
    ssa.features_from_scores(
        pca, scores, feature_recon, n_components=25,
    )
    logging.info("Stage 12.8 Converting features to masks")
    ssa.dataset_from_features(
        model.mask_from_features, feature_recon_err, mask_recon_err,
    )
    ssa.dataset_from_features(
        model.mask_from_features, feature_modes, mask_modes,
    )
    ssa.dataset_from_features(
        model.mask_from_features, feature_recon, mask_recon,
    )
    logging.info("Stage 12.9 Displaying reconstruction")
    display.recon(mask_recon, recon_png, recon_mp4)

    logging.info("Stage 12.10 Displaying reconstruction error")
    display.recon_err(mask_recon_err, recon_err_png, recon_err_mp4)

    logging.info("Stage 12.11 Displaying principal modes")
    display.modes(mask_modes, modes_png, modes_mp4)
    
    logging.info("Stage 12 --- Spectral PCA succesfully completed ---")


if __name__ == '__main__':

    BUILD = r"C:\Users\md1spsx\Documents\Data\iBEAt_Build"
    pipe.run_dask_stage(run, BUILD, PIPELINE, __file__, min_ram_per_worker=3)