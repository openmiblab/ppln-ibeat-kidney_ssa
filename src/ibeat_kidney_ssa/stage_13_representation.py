import os
import logging
import miblab_ssa as ssa

from miblab import pipe
from ibeat_kidney_ssa.utils.models import MODELS

PIPELINE = 'kidney_ssa'


def run(build, client, model='spectral'):

    logging.info(f"Stage 13[{model}] Setup")

    dir_input = os.path.join(build, PIPELINE, 'stage_9_stack_normalized')
    dir_output = pipe.stage_output_dir(build, PIPELINE, __file__)
    dir_model_output = os.path.join(dir_output, model)
    os.makedirs(dir_model_output, exist_ok=True)

    # Input data
    masks = os.path.join(dir_input, 'normalized_kidney_masks.zarr')

    # Output data
    features = os.path.join(dir_model_output, 'data_features.zarr')

    logging.info(f"Stage 13[{model}].1 Computing features")
    module = MODELS[model]['module']
    kwargs = MODELS[model]['kwargs']
    pipe.adjust_workers(client, min_ram_per_worker=2)
    ssa.features_from_dataset(
        features_from_mask=module.features_from_mask, 
        masks_zarr_path=masks, 
        output_zarr_path=features, 
        **kwargs,
    )
    logging.info(f"Stage 13[{model}] Completed")


if __name__ == '__main__':

    BUILD = r"C:\Users\md1spsx\Documents\Data\iBEAt_Build"
    kwargs = {
        "model": {'type': str, 'default': 'spectral', 'help': 'Model'}
    }
    pipe.run_dask_stage(run, BUILD, PIPELINE, __file__, min_ram_per_worker=16, **kwargs)