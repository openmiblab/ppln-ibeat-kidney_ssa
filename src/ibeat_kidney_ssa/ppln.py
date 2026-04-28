import ibeat_kidney_ssa as ppln
from miblab import pipe

PIPELINE = 'kidney_ssa'

def run(build, logfile):
    
    # ppln.stage_1_normalize_controls.run(build, logfile)
    # ppln.stage_2_average_controls.run(build, logfile)
    # ppln.stage_3_normalize.run(build, logfile)

    # ppln.stage_4_extract_features.run(build, logfile)
    # ppln.stage_5_export.run(build, logfile)
    # ppln.stage_6_distance_matrices.run(build, logfile)

    # ppln.stage_7_features.run(build, logfile, model='spectral')
    # ppln.stage_8_representation.run(build, logfile, model='spectral')
    # ppln.stage_9_pca.run(build, logfile, model='spectral')
    ppln.stage_10_deep_pca.run(build, logfile, model='spectral')

    # ppln.stage_7_features.run(build, logfile, model='chebyshev')
    # ppln.stage_8_representation.run(build, logfile, model='chebyshev')
    # ppln.stage_9_pca.run(build, logfile, model='chebyshev')
    # ppln.stage_10_deep_pca.run(build, logfile, model='chebyshev')

    # ppln.stage_7_features.run(build, logfile, model='spectral')
    # ppln.stage_8_representation.run(build, logfile, model='spectral')
    # ppln.stage_9_pca.run(build, logfile, model='spectral')
    ppln.stage_10_deep_pca.run(build, logfile, model='legendre')

if __name__=='__main__':

    BUILD = r"C:\Users\md1jdsp\Documents\Data\iBEAt_Build"
    # pipe.run_client_ppln(run, BUILD, PIPELINE, min_ram_per_worker = 16)
    pipe.run_ppln(run, BUILD, PIPELINE)