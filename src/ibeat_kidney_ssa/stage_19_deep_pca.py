import os
import logging

import miblab_ssa as ssa
import miblab_ssa.sdf_ft as model

from ibeat_kidney_ssa.utils import pipe, display

PIPELINE = 'kidney_ssa'
DEBUG = False

def run(build, client):

    logging.info("Stage 19 --- Deep spectral PCA ---")
    dir_input_features = os.path.join(build, PIPELINE, 'stage_12_spectral_pca')
    dir_output = pipe.stage_output_dir(build, PIPELINE, __file__)

    # Input data
    features = os.path.join(dir_input_features, 'data_features.zarr')

    # Output - arrays
    feature_modes = os.path.join(dir_output, f"data_feature_modes.zarr")
    mask_modes = os.path.join(dir_output, f"data_mask_modes.zarr")
    feature_recon_err = os.path.join(dir_output, f"data_feature_recon_err.zarr")
    feature_recon = os.path.join(dir_output, f"data_feature_recon.zarr")
    mask_recon_err = os.path.join(dir_output, f"data_mask_recon_err.zarr")
    mask_recon = os.path.join(dir_output, f"data_mask_recon.zarr")

    # Output - tables
    cumulative_mse = os.path.join(dir_output, 'table_cumulative_mse.csv')
    marginal_mse = os.path.join(dir_output, 'table_marginal_mse.csv')
    dice_recon_error = os.path.join(dir_output, 'table_dice_recon_error.csv')
    haus_recon_error = os.path.join(dir_output, 'table_hausdorff_recon_error.csv')
    scores = os.path.join(dir_output, f"table_scores.csv")
    normalized_scores = os.path.join(dir_output, f"table_normalized_scores.csv")
    modes_shape_features = os.path.join(dir_output, f"table_modes_shape_features")

    # Output - images
    pca_performance = os.path.join(dir_output, 'imgs_performance.png')
    modes_png = os.path.join(dir_output, 'imgs_modes')
    modes_sections_png = os.path.join(dir_output, 'imgs_modes_sections')
    modes_fingerprints = os.path.join(dir_output, 'imgs_modes_fingerprints')
    recon_png = os.path.join(dir_output, 'imgs_recon')
    recon_err_png = os.path.join(dir_output, 'imgs_recon_err')
    recon_performance_img = os.path.join(dir_output, 'imgs_recon_performance.png')

    # Output - movies
    modes_mp4 = os.path.join(dir_output, 'movie_modes.mp4')
    recon_mp4 = os.path.join(dir_output, 'movie_recon.mp4')
    recon_err_mp4 = os.path.join(dir_output, 'movie_recon_err.mp4')

    # Output - models
    model_checkpoint = os.path.join(dir_output, 'model.pth')

    # Hyperparameters
    n_comp = 128
    n_outliers_display = 9
    if DEBUG:
        epochs = 100
        n_outliers = 25
        max_comp = 20
    else:
        epochs = 1000
        n_outliers = None
        max_comp = 64

    logging.info("Stage 19.1 Computing PCA")
    ssa.deep_pca_from_features(
        features_zarr_path=features, 
        model_pth_path=model_checkpoint, 
        n_components=n_comp, 
        epochs=epochs
    )
    ssa.add_deep_pca_metrics(
        features_zarr_path=features, 
        model_pth_path=model_checkpoint,
    )

    logging.info("Stage 19.2 Computing PCA reconstructions")
    ssa.deep_scores_from_features(
        features_zarr_path=features, 
        model_pth_path=model_checkpoint, 
        scores_csv_path=scores,
        normalized_scores_csv_path=normalized_scores,
    )
    ssa.deep_features_from_scores(
        model_pth_path=model_checkpoint, 
        scores_csv_path=scores, 
        features_zarr_path=feature_recon, 
        n_components=n_comp
    )
    pipe.adjust_workers(client, min_ram_per_worker=2)
    ssa.dataset_from_features(
        mask_from_features_func=model.mask_from_features, 
        features_zarr_path=feature_recon, 
        dataset_zarr_path=mask_recon
    )
    pipe.adjust_workers(client, min_ram_per_worker=16)
    display.recon(mask_recon, recon_png, recon_mp4)

    logging.info("Stage 19.3 Computing PCA performance")
    ssa.deep_pca_performance(
        model_pth_path=model_checkpoint, 
        scores_csv_path=scores, 
        gt_features_zarr_path=features, 
        marginal_mse_csv_path=marginal_mse,
        cumulative_mse_csv_path=cumulative_mse, 
        n_components=n_comp,
    )
    ssa.plot_deep_pca_performance(
        model_pth_path=model_checkpoint, 
        output_image_path=pca_performance, 
        marginal_mse_path=marginal_mse,  
        cumulative_mse_path=cumulative_mse, 
        n_components=n_comp,
    )

    logging.info("Stage 19.4 Computing reconstruction accuracy")
    labels = display.get_outlier_labels(cumulative_mse, n=n_outliers)
    ssa.deep_cumulative_features_from_scores(
        model_pth_path=model_checkpoint, 
        scores_csv_path=scores, 
        gt_features_zarr_path=features,
        output_zarr_path=feature_recon_err, 
        target_labels=labels, 
        step_size=1, 
        max_components=max_comp,
    )
    pipe.adjust_workers(client, min_ram_per_worker=2)
    ssa.dataset_from_features(
        mask_from_features_func=model.mask_from_features, 
        features_zarr_path=feature_recon_err, 
        dataset_zarr_path=mask_recon_err,
    )
    ssa.recon_error(
        dataset_zarr_path=mask_recon_err, 
        dice_csv_path=dice_recon_error, 
        hausdorff_csv_path=haus_recon_error
    )
    ssa.plot_pca_reconstruction_performance(
        dice_csv_path=dice_recon_error, 
        hausdorff_csv_path=haus_recon_error, 
        output_image_path=recon_performance_img
    )
    pipe.adjust_workers(client, min_ram_per_worker=16)
    display.recon_err(
        mask_zarr_path=mask_recon_err, 
        dir_png=recon_err_png, 
        movie_file=recon_err_mp4, 
        n_samples=n_outliers_display,
        n_components=15,
    )

    logging.info("Stage 19.5 Computing principal modes")
    ssa.deep_modes_from_pca(
        model_pth_path=model_checkpoint, 
        modes_zarr_path=feature_modes, 
        n_components=8, 
        n_coeffs=15,
        max_coeff=6,
    )
    pipe.adjust_workers(client, min_ram_per_worker=2)
    ssa.dataset_from_features(
        mask_from_features_func=model.mask_from_features, 
        features_zarr_path=feature_modes, 
        dataset_zarr_path=mask_modes
    )
    pipe.adjust_workers(client, min_ram_per_worker=16)
    ssa.plot_mask_sections(
        dataset_zarr_path=mask_modes, 
        dir_png=modes_sections_png, 
    )
    display.modes(
        modes_zarr_path=mask_modes, 
        dir_png=modes_png, 
        movie_file=modes_mp4,
    )
    pipe.adjust_workers(client, min_ram_per_worker=2)
    ssa.dataset_shapes(
        dataset_zarr_path=mask_modes, 
        dir_csv=modes_shape_features, 
    )
    pipe.adjust_workers(client, min_ram_per_worker=16)
    ssa.plot_shape_fingerprints(
        dir_csv=modes_shape_features,
        dir_png=modes_fingerprints,
    )

    logging.info("Stage 19 --- Deep PCA succesfully completed ---")



if __name__ == '__main__':

    BUILD = r"C:\Users\md1spsx\Documents\Data\iBEAt_Build"
    pipe.run_dask_stage(run, BUILD, PIPELINE, __file__, min_ram_per_worker=16)