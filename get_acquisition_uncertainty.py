import numpy as np
from predict_ensemble import predict_ensemble

def get_acquisition_uncertainty(X_unscaled, e2nn_models, emulator_functions, xscale_obj, yscale_obj):
    """
    UNCERTAINTY: ACQUISITION FUNCTION TO MAXIMIZE FOR GLOBAL SURROGATE ACCURACY
    """
    # if X is 1d, assume it is a single sample and reshape to 2D
    if len(np.shape(X_unscaled)) == 1:
        X_unscaled = X_unscaled.reshape(1, -1)

    mu, sig_tdist, df = predict_ensemble(X_unscaled, e2nn_models,
                                         emulator_functions,
                                         xscale_obj, yscale_obj)
    sig_tdist = sig_tdist.flatten()
    return(sig_tdist)
