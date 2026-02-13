import os
import logging

import miblab_ssa as ssa

from ibeat_kidney_ssa.utils import pipe, display

PIPELINE = 'kidney_ssa'

def run(build, client):

    logging.info("Stage 13 --- Chebyshev PCA ---")
    dir_input = os.path.join(build, PIPELINE, 'stage_9_stack_normalized')
    dir_output = pipe.stage_output_dir(build, PIPELINE, __file__)

    # Input data - stack of normalized masks
    masks_path = os.path.join(dir_input, 'normalized_kidney_masks.zarr')

    # Outputs
    feature_path = os.path.join(dir_output, 'features.zarr')
    pca_path = os.path.join(dir_output, f"components.zarr")
    scores_path = os.path.join(dir_output, f"scores.zarr")
    feature_modes_path = os.path.join(dir_output, f"feature_modes.zarr")
    mask_modes_path = os.path.join(dir_output, f"mask_modes.zarr")
    feature_recon = os.path.join(dir_output, f"feature_recon.zarr")
    mask_recon = os.path.join(dir_output, f"mask_recon.zarr")
    dir_png = os.path.join(dir_output, 'modes_png')
    movie_file = os.path.join(dir_output, 'modes.mp4')
    cumulative_mse = os.path.join(dir_output, 'cumulative_mse.csv')
    marginal_mse = os.path.join(dir_output, 'marginal_mse.csv')
    pca_performance_plot = os.path.join(dir_output, 'performance_plot.png')
    recon_png = os.path.join(dir_output, 'recon_png')
    recon_movie = os.path.join(dir_output, 'recon.mp4')

    logging.info(f"Stage 13.1 Building feature matrix")
    pipe.adjust_workers(client, 4)
    ssa.features_from_dataset_zarr(
        ssa.sdf_cheby.features_from_mask, 
        masks_path, 
        feature_path, 
        order=27, # 4060 features
    )

    logging.info("Stage 13.2 Computing PCA")
    pipe.adjust_workers(client, 16)
    ssa.pca_from_features_zarr(feature_path, pca_path)

    logging.info("Stage 13.3 Compute chebyshev scores")
    pipe.adjust_workers(client, 16)
    ssa.scores_from_features_zarr(feature_path, pca_path, scores_path)

    logging.info("Stage 13.4 Compute principal modes")
    pipe.adjust_workers(client, 16)
    ssa.modes_from_pca_zarr(pca_path, feature_modes_path, n_components=8, max_coeff=8)
    ssa.dataset_from_features_zarr(
        ssa.sdf_cheby.mask_from_features, 
        feature_modes_path, 
        mask_modes_path,
    )

    logging.info("Stage 13.5 Measure model performance")
    pipe.adjust_workers(client, 16)
    ssa.pca_performance(
        pca_path, 
        scores_path, 
        feature_path, 
        marginal_mse, 
        cumulative_mse, 
        n_components=200,
        overwrite=True,
    )
    ssa.plot_pca_performance(
        pca_path, 
        pca_performance_plot, 
        marginal_mse, 
        cumulative_mse, 
        n_components=200,
    )

    logging.info("Stage 13.6 Display principal modes")
    pipe.adjust_workers(client, 16)
    display.modes(mask_modes_path, dir_png, movie_file)

    logging.info("Stage 13.7 Display reconstruction accuracy")
    labels = display.get_outlier_labels(cumulative_mse, n=8)
    ssa.cumulative_features_from_scores_zarr(
        pca_path, 
        scores_path, 
        feature_recon, 
        labels, 
        step_size=5, 
        max_components=50,
    )
    ssa.dataset_from_features_zarr(
        ssa.sdf_cheby.mask_from_features, 
        feature_recon, 
        mask_recon,
    )
    display.recon(mask_recon, recon_png, recon_movie)

    logging.info("Stage 13: Chebyshev PCA successfully completed.")


if __name__ == '__main__':

    BUILD = r"C:\Users\md1spsx\Documents\Data\iBEAt_Build"
    pipe.run_dask_stage(run, BUILD, PIPELINE, __file__, min_ram_per_worker=4)