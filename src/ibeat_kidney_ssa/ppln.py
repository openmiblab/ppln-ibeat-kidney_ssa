import ibeat_kidney_ssa as ppln
from ibeat_kidney_ssa.utils import pipe

PIPELINE = 'kidney_ssa'

def run(build, client):
    
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
    ppln.stage_12_spectral_pca.run(build, client)
    ppln.stage_13_chebyshev_pca.run(build, client)


if __name__=='__main__':

    BUILD = r"C:\Users\md1spsx\Documents\Data\iBEAt_Build"
    pipe.run_dask_script(run, BUILD, PIPELINE, min_ram_per_worker = 4.0)