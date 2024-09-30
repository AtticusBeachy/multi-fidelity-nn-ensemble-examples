import numpy as np

from predict_e2nn import predict_e2nn

def predict_ensemble(x_raw, e2nn_models, EMULATOR_FUNCTIONS, xscale_obj, yscale_obj):
    """
    Predict the response of an ensemble of e2nn models
    """
    y_models = []
    for model in e2nn_models:
        y_model = predict_e2nn(x_raw, model, EMULATOR_FUNCTIONS, 
                               xscale_obj, yscale_obj) #1d output
        y_models.append(y_model.reshape(-1, 1))
    y_models = np.concatenate(y_models, axis=1)
    
    mu_tdist = np.mean(y_models, axis=1)
    mu_tdist = mu_tdist.reshape(-1, 1)

    s = np.std(y_models, axis=1, ddof=1)
    s = s.reshape(-1, 1)
    n_mdl = len(e2nn_models)

    sig_tdist = s*np.sqrt((n_mdl+1)/n_mdl)  #s/np.sqrt(n_mdl)
    df = n_mdl-1

    return(mu_tdist, sig_tdist, df)

