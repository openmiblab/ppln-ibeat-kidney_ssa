import os
import logging
import miblab_ssa as ssa

from ibeat_kidney_ssa.utils import pipe

PIPELINE = 'kidney_ssa'


def run(build, client):

    logging.info("Stage 12 --- Starting computation of representations ---")

    models = {
        'spectral':{
            'module': ssa.sdf_ft,
            'kwargs': {
                'order': 19 # 4019 features
            },
            'min_order': 10,
            'max_order': 24,
        },
        'chebyshev':{
            'module': ssa.sdf_cheby,
            'kwargs': {
                'order': 27
            },
            'min_order': 10,
            'max_order': 36,
        },
    }
    for model, props in models.items():
        run_model(build, client, model, props)

    logging.info("Stage 12 --- Finished computation of representations ---")


def run_model(build, client, model, props):

    logging.info(f"Stage 12[{model}] Setup")

    module = props['module']
    kwargs = props['kwargs']

    dir_input = os.path.join(build, PIPELINE, 'stage_9_stack_normalized')
    dir_output = pipe.stage_output_dir(build, PIPELINE, __file__)
    dir_model_output = os.path.join(dir_output, model)
    os.makedirs(dir_model_output, exist_ok=True)

    # Input data
    masks = os.path.join(dir_input, 'normalized_kidney_masks.zarr')

    # Output
    features = os.path.join(dir_model_output, 'data_features.zarr')
    dice_feature_selection = os.path.join(dir_model_output, 'table_feature_sel_dice.csv')
    haus_feature_selection = os.path.join(dir_model_output, 'table_feature_sel_hausdorff.csv')
    recon_fidelity = os.path.join(dir_model_output, 'reconstruction_fidelity.png')

    logging.info(f"Stage 12[{model}].1 Selecting number of features")
    pipe.adjust_workers(client, min_ram_per_worker=2)
    ssa.reconstruction_fidelity(
        smooth_mask_func=module.smooth_mask,
        dataset_zarr_path=masks,
        dice_csv_path=dice_feature_selection,
        hausdorff_csv_path=haus_feature_selection,
        min_order=props['min_order'], 
        max_order=props['max_order'],
    )
    pipe.adjust_workers(client, min_ram_per_worker=16)
    ssa.plot_reconstruction_fidelity(
        dice_csv_path=dice_feature_selection,
        hausdorff_csv_path=haus_feature_selection,  
        output_image_path=recon_fidelity,    
    )

    logging.info(f"Stage 12[{model}].2 Building feature matrix")
    ssa.features_from_dataset(
        module.features_from_mask, 
        masks, 
        features, 
        **kwargs
    )
    
    logging.info(f"Stage 12[{model}] Completed")


if __name__ == '__main__':

    BUILD = r"C:\Users\md1spsx\Documents\Data\iBEAt_Build"
    pipe.run_dask_stage(run, BUILD, PIPELINE, __file__, min_ram_per_worker=3)