import numpy as np
import tensorflow as tf
# physical_devices = tf.config.list_physical_devices("GPU")
# from tensorflow.python.framework.ops import disable_eager_execution
# disable_eager_execution()
from scipy.stats import t

class Tdists:
    def __init__(self, mu, sig, df):
        self.mu = mu
        self.sig = sig
        self.df = df

def call_emulators(X_unscaled, Fe, yscale_obj):
    """
    Return emulator values
    """
    Nemulator = len(Fe)
    for ii in range(Nemulator):
        emulator = Fe[ii]
        Ylf_unscaled = emulator(X_unscaled)
        Ylf_scaled = yscale_obj.transform(Ylf_unscaled.reshape(-1, 1))

        if ii==0:
            Emulator_vals = Ylf_scaled
        else:
            Emulator_vals = np.hstack([Emulator_vals, Ylf_scaled])
    return(Emulator_vals)

def predict_e2nn(X_unscaled, E2NN_model, Fe, xscale_obj, yscale_obj):
    """
    Predict the response of an e2nn model
    """

    X_scaled = xscale_obj.transform(X_unscaled)

    Y_lf_scaled = call_emulators(X_unscaled, Fe, yscale_obj)
    Emulator_vals = Y_lf_scaled

    # # optional
    # X_scaled = tf.convert_to_tensor(X_scaled)
    # Emulator_vals = tf.convert_to_tensor(Emulator_vals)

    if X_scaled.shape[0] >= 100_000: #10: # good for bigger predictions
        Y_E2NN_pred = E2NN_model.predict([X_scaled, Emulator_vals])
        # print("fast")
    else: # quick for small predictions
        Y_E2NN_pred = E2NN_model([X_scaled, Emulator_vals])
        # print("quick")

    Y_E2NN_pred = yscale_obj.inverse_transform(Y_E2NN_pred)
    Y_E2NN_pred = Y_E2NN_pred.flatten()

    return(Y_E2NN_pred)

def predict_ensemble(X_unscaled, E2NN_MODELS, Fe, xscale_obj, yscale_obj):
    """
    Predict the response of an ensemble of e2nn models
    """

    n_pt = X_unscaled.shape[0]
    n_mdl = len(E2NN_MODELS)

    Y_E2NN_preds = []
    for model in E2NN_MODELS:
        E2NN_pred = predict_e2nn(X_unscaled, model, Fe, xscale_obj, yscale_obj) #1d output
        Y_E2NN_preds.append(E2NN_pred.reshape(-1, 1))
    Y_E2NN_preds = np.concatenate(Y_E2NN_preds, axis=1)
    Y_ensemble = np.mean(Y_E2NN_preds, axis = 1)
    Y_ensemble = Y_ensemble.reshape(-1, 1)

    s = np.std(Y_E2NN_preds, axis = 1, ddof = 1)
    s = s.reshape(-1, 1)

    mu_tdist = Y_ensemble
    sig_tdist = s*np.sqrt((n_mdl+1)/n_mdl)#s/np.sqrt(n_mdl)
    df = n_mdl-1

    t_dists = Tdists(mu_tdist, sig_tdist, df)

    x_95_lower = t.ppf(0.025, df, mu_tdist, sig_tdist) # percent point function (inverse cdf)
    x_95_upper = t.isf(0.025, df, mu_tdist, sig_tdist) # inverse survival function
    return(Y_ensemble, t_dists, x_95_lower, x_95_upper)#s_95)

def ensemble_fn(X_unscaled, E2NN_MODELS, Fe, xscale_obj, yscale_obj):
    """
    At each point return mean, uncertainty, and t-distribution degrees of freedom
    """
    (Y_ensemble, t_dists, *__) = predict_ensemble(X_unscaled, E2NN_MODELS, Fe, xscale_obj, yscale_obj)
    return(Y_ensemble, t_dists.sig, t_dists.df)