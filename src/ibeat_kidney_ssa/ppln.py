import argparse

import ibeat_kidney_ssa as ppln
from ibeat_kidney_ssa.utils import pipe

PIPELINE = 'kidney_ssa'

def run(build):

    pipe.setup_pipeline(build, PIPELINE)

    # # Compute stages
    # ppln.stage_1_normalize_controls.run(build)
    # ppln.stage_2_display_controls.run(build)
    # ppln.stage_3_average_controls.run(build)
    # ppln.stage_4_display_average_controls.run(build)
    # ppln.stage_5_normalize.run(build)
    # ppln.stage_6_display_normalized.run(build)
    # ppln.stage_7_extract_features.run(build)
    # ppln.stage_8_export.run(build)
    ppln.stage_9_stack_normalized.run(build)
    ppln.stage_10_dice_matrix.run(build)
    ppln.stage_11_hausdorff_matrix.run(build)
    ppln.stage_12_spectral_pca.run(build)
    ppln.stage_13_chebyshev_pca.run(build)
    
    
if __name__=='__main__':

    BUILD = r"C:\Users\md1spsx\Documents\Data\iBEAt_Build"

    parser = argparse.ArgumentParser()
    parser.add_argument("--build", type=str, default=BUILD, help="Build folder")
    args = parser.parse_args()

    run(args.build)

