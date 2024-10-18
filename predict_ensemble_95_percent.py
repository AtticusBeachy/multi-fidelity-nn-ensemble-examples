from scipy.stats import t

from predict_ensemble import predict_ensemble

def predict_ensemble_95_percent(x_raw, e2nn_models, EMULATOR_FUNCTIONS, xscale_obj, yscale_obj):
    mu_tdist, sig_tdist, df = predict_ensemble(
        x_raw, e2nn_models, EMULATOR_FUNCTIONS, xscale_obj, yscale_obj
    )
    # percent point function (inverse cdf)
    y_95_lower = t.ppf(0.025, df, mu_tdist, sig_tdist) 
    # inverse survival function
    y_95_upper = t.isf(0.025, df, mu_tdist, sig_tdist) 
    return(mu_tdist, y_95_lower, y_95_upper)

