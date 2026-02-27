import os
import logging
import miblab_ssa as ssa
from miblab import pipe

from ibeat_kidney_ssa.utils.models import MODELS
from ibeat_kidney_ssa.utils import display


PIPELINE = 'kidney_ssa'


def run(build, dir_output, client, model='spectral'):

    logging.info(f"Stage 12[{model}] Setup")

    dir_input = os.path.join(build, PIPELINE, 'stage_9_stack_normalized')
    dir_model_output = os.path.join(dir_output, model)
    os.makedirs(dir_model_output, exist_ok=True)

    # Input data
    masks = os.path.join(dir_input, 'normalized_kidney_masks.zarr')

    # Output data
    dice_feature_selection = os.path.join(dir_model_output, 'table_feature_sel_dice.csv')
    haus_feature_selection = os.path.join(dir_model_output, 'table_feature_sel_hausdorff.csv')
    recon_fidelity = os.path.join(dir_model_output, 'reconstruction_fidelity.png')
    recon_masks = os.path.join(dir_model_output, 'reconstructed_kidney_masks.zarr')

    # Output - images
    recon_png = os.path.join(dir_model_output, 'imgs_recon_orders')

    # Output - movies
    recon_mp4 = os.path.join(dir_model_output, 'movie_recon_orders.mp4')

    # Hyperparameters
    n_outliers = 10

    logging.info(f"Stage 12[{model}] Selecting number of features")
    pipe.adjust_workers(client, min_ram_per_worker=2)
    module = MODELS[model]['module']
    kwargs = MODELS[model]['kwargs']
    # ssa.reconstruction_fidelity(
    #     smooth_mask_func=module.smooth_mask,
    #     dataset_zarr_path=masks,
    #     dice_csv_path=dice_feature_selection,
    #     hausdorff_csv_path=haus_feature_selection,
    #     min_order=MODELS[model]['min_order'], 
    #     max_order=MODELS[model]['max_order'],
    #     **kwargs,
    # )

    # logging.info(f"Stage 12[{model}].2 Plotting reconstruction fidelity")
    # pipe.adjust_workers(client, min_ram_per_worker=16)
    # ssa.plot_reconstruction_fidelity(
    #     dice_csv_path=dice_feature_selection,
    #     hausdorff_csv_path=haus_feature_selection,  
    #     output_image_path=recon_fidelity,    
    # )

    logging.info(f"Stage 12[{model}] Displaying reconstruction fidelity")
    labels = display.get_outlier_labels(
        haus_feature_selection, 
        n=n_outliers, 
        column_idx=-1,
    )
    ssa.save_reconstructed_masks(
        smooth_mask_func=module.smooth_mask,
        dataset_zarr_path=masks,
        output_zarr_path=recon_masks,
        target_labels=labels,
        min_order=MODELS[model]['min_order'], 
        max_order=MODELS[model]['max_order'],
        **kwargs
    )
    pipe.adjust_workers(client, min_ram_per_worker=16)
    display.recon(recon_masks, recon_png, recon_mp4)

    logging.info(f"Stage 12[{model}] Completed")


if __name__ == '__main__':

    build = r"C:\Users\md1spsx\Documents\Data\iBEAt_Build"
    kwargs = {
        "model": {'type': str, 'default': 'spectral', 'help': 'Model'}
    }
    pipe.run_client_stage(run, build, PIPELINE, __file__, min_ram_per_worker=16, **kwargs)