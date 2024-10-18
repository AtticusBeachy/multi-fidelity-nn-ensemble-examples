import numpy as np
from scipy.stats import t

from predict_ensemble import predict_ensemble

def get_acquisition_ei(X_unscaled, e2nn_models, emulator_functions, f_best,
                       xscale_obj, yscale_obj):
    """
    EXPECTED IMPROVEMENT: ACQUISITION FUNCTION TO MAXIMIZE FOR OPTIMIZATION
    For T-distribution:
        z = (f_best-mu)/sig
        EI = (f_best-mu)*CDF(z) + df/(df-1) * (1+z**2/df)*sig*PDF(z)
    """
    # if X is 1d, assume it is a single sample and reshape to 2D
    if len(np.shape(X_unscaled)) == 1:
        X_unscaled = X_unscaled.reshape(1, -1)

    mu, sig, df = predict_ensemble(X_unscaled, e2nn_models, emulator_functions,
                                   xscale_obj, yscale_obj)
    z = (f_best-mu)/sig
    EI = (f_best-mu)*t.cdf(z, df) + df/(df-1) * (1+z**2/df)*sig*t.pdf(z, df)
    EI = EI.flatten()
    return(EI)
