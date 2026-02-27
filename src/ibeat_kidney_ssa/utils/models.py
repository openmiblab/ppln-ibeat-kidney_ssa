import miblab_ssa as ssa


MODELS = { 
    'spectral':{
        'module': ssa.sdf_ft,
        'kwargs': {
            'order': 19 
        },
        'min_order': 5,
        'max_order': 32, 
    },
    'chebyshev':{
        'module': ssa.sdf_cheby,
        'kwargs': {
            'order': 27 
        },
        'min_order': 10,
        'max_order': 36, 
    },
    'legendre':{
        'module': ssa.sdf_legendre,
        'kwargs': {
            'order': 27
        },
        'min_order': 10,
        'max_order': 36,
    },
    'spline':{
        'module': ssa.sdf_spline,
        'kwargs': {
            'order': 16
        },
        'min_order': 10,
        'max_order': 20,
    },
    'pspline':{
        'module': ssa.sdf_pspline,
        'kwargs': {
            'order': 18,
            'div': 2/3,
            'degree': 3,
        },
        'min_order': 10,
        'max_order': 20,
    },
    'rbf':{
        'module': ssa.sdf_rbf,
        'kwargs': {
            'order': 16,
            'epsilon': 4,
        },
        'min_order': 10,
        'max_order': 20,
    },
    'wavelet':{
        'module': ssa.sdf_wvlt,
        'kwargs': {
            'order': 7,
            "min_level": 2,
        },
        'min_order': 3,
        'max_order': 9,
    },
}
