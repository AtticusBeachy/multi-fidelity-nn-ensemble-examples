import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

from build_e2nn_one_layer import build_e2nn_one_layer
from build_e2nn_two_layer import build_e2nn_two_layer
from evaluate_emulators import evaluate_emulators
from err_nrmse import err_nrmse

def build_ensemble(EMULATOR_FUNCTIONS, x_train_raw, y_train_raw, LB, UB,
                   ensemble_settings=None, dtype = "float64", ERR_TOL = 0.001,
                   WEIGHT_TOL = 100):
    """
    Build an ensemble of e2nn models.
    
    Arguments
    ---------

    EMULATOR_FUNCTIONS (list): Inexpensive information sources

    x_train_raw, y_train_raw (numpy array): Unscaled training data

    LB, UB (numpy array): Lower and upper bounds for the domain

    ensemble_settings (list): List of dictionaries of e2nn training parameters

    dtype (string): Type of TensorFlow weights. Using float64 instead of 
                    float32 offers better accuracy and stability during rapid 
                    training.

    ERR_TOL (float): Maximum allowable NRMSE for an e2nn model

    WEIGHT_TOL (float or int): Maximum allowable value for largest weight
                               magnitude in e2nn model

    Returns
    -------

    E2NN_MODELS (list): A list of acceptable e2nn models

    xscale_obj, yscale_obj (MinMaxScaler object): Objects to scale and unscale 
                                                   e2nn inputs and outputs

    fail_1L, fail_2L (bool): Whether training failed too often for 1 and 2 
                             layer models with Fourier activations

    ALL_E2NN_MODELS (list): all e2nn models, including the bad ones

    bad_idxs (numpy array): The indices of the bad e2nn models


    NOTE: Emulators are scaled the same way as the HF training data. If 
    emulators have a much different scale than the HF data, but good 
    correlation, it is better to scale them seperately.
    """

    ################################################################################
    """SCALE TRAINING DATA"""
    BOUNDS = np.vstack([LB, UB])

    # scale data from [-1, 1] using sk-learn
    xscale_obj = MinMaxScaler(feature_range=(-1, 1)).fit(BOUNDS)
    yscale_obj = MinMaxScaler(feature_range=(-1, 1)).fit(y_train_raw)

    x_train = xscale_obj.transform(x_train_raw)
    y_train = yscale_obj.transform(y_train_raw)

    ################################################################################
    """GET EMULATOR VALUES FOR TRAINING"""
    emulator_train = evaluate_emulators(x_train_raw, EMULATOR_FUNCTIONS, yscale_obj)

    ############################################################################
    """SET NN ARCHITECTURE PARAMETERS"""

    if ensemble_settings is None:
        N_COPY_ARCH = 2  # number of copies of each NN architecture

        ensemble_settings = N_COPY_ARCH*[
            {"n_layers": 1, "activation_name": "fourier_low_1layer", "activation_function": lambda x: tf.sin(1.0*x)},
            {"n_layers": 1, "activation_name": "fourier_med_1layer", "activation_function": lambda x: tf.sin(1.1*x)},
            {"n_layers": 1, "activation_name": "fourier_high_1layer", "activation_function": lambda x: tf.sin(1.2*x)},
            {"n_layers": 1, "activation_name": "swish", "activation_function": "swish"},
        
            {"n_layers": 2, "activation_name": "fourier_low_2layer", "activation_function": lambda x: tf.sin(1.0*x)},
            {"n_layers": 2, "activation_name": "fourier_med_2layer", "activation_function": lambda x: tf.sin(1.1*x)},
            {"n_layers": 2, "activation_name": "fourier_high_2layer", "activation_function": lambda x: tf.sin(1.2*x)},
            {"n_layers": 2, "activation_name": "swish", "activation_function": "swish"},
        ]

    NUM_MODELS_TOTAL = len(ensemble_settings)

    ################################################################################
    """GET TRAINED E2NN MODELS"""

    ALL_E2NN_MODELS = []
    MAX_E2NN_WEIGHTS = []

    for settings in ensemble_settings:
        n_layers = settings["n_layers"]
        activation_name = settings["activation_name"]
        activation_function = settings["activation_function"]
        if n_layers==1:
            nn, wt = build_e2nn_one_layer(x_train, y_train, emulator_train, 
                                          dtype=dtype, 
                                          activation_name=activation_name,
                                          activation_function = activation_function)
        elif n_layers==2:
            nn, wt = build_e2nn_two_layer(x_train, y_train, emulator_train, 
                                          dtype=dtype, 
                                          activation_name=activation_name,
                                          activation_function = activation_function)
        else:
            Error(f"Number of hidden layers must be 1 or 2, not {n_layers}")
        ALL_E2NN_MODELS.append(nn)
        MAX_E2NN_WEIGHTS.append(wt)

    MAX_E2NN_WEIGHTS = np.array(MAX_E2NN_WEIGHTS)

    ################################################################################
    """EXTRACT INDICES OF MODELS WITH FOURIER ACTIVATIONS"""

    fourier_1_layer_idx = np.array([], dtype="int32")
    fourier_2_layer_idx = np.array([], dtype="int32")

    for i, setting in enumerate(settings):
        n_layers = settings["n_layers"]
        activation_name = settings["activation_name"]
        
        if "fourier" in activation_name:
            if n_layers==1:
                fourier_1_layer_idx = np.append(fourier_1_layer_idx, i)
            elif n_layers==2: 
                fourier_2_layer_idx = np.append(fourier_2_layer_idx, i)

    ################################################################################
    """GET ERROR OF EACH MODEL ON TRAINING DATA"""

    e2nn_nrmses = np.empty(NUM_MODELS_TOTAL)         # accuracy check
    for ii in range(NUM_MODELS_TOTAL):
        model = ALL_E2NN_MODELS[ii]

        ################################################################################
        """SAVE E2NN MODEL"""
        # model.save("saved_models/model_"+str(ii))

        ################################################################################
        """CHECK ERROR"""
        # Train Error
        y_train_pred = model.predict([x_train, emulator_train])
        y_train_pred = y_train_pred.flatten()
        nrmse = err_nrmse(y_train_pred, y_train)

        e2nn_nrmses[ii] = nrmse

    ################################################################################
    """CHECK FITS"""

    # nrmse not above tolerance (needed when many bad fits exist)
    bad_raw_err = e2nn_nrmses > ERR_TOL

    bad_weights = MAX_E2NN_WEIGHTS > WEIGHT_TOL

    # bad indices true/false
    bad_idxs_tf = bad_raw_err | bad_weights
    bad_idxs = np.flatnonzero(bad_idxs_tf)

    # bad fraction of 1L fourier and 2L fourier
    bad_idxs_tf_1L_fourier = bad_idxs_tf[fourier_1_layer_idx]
    bad_frac_1L_fourier = np.sum(bad_idxs_tf_1L_fourier)/bad_idxs_tf_1L_fourier.size
    bad_idxs_tf_2L_fourier = bad_idxs_tf[fourier_2_layer_idx]
    bad_frac_2L_fourier = np.sum(bad_idxs_tf_2L_fourier)/bad_idxs_tf_2L_fourier.size

    # check fitting failure individually for 2L fourier and 1L fourier (if most bad will adjust fourier frequency)
    if bad_frac_1L_fourier > 0.4999:  # Failure if half or more fits are bad
        fail_1L = True
    else:
        fail_1L = False

    if bad_frac_2L_fourier > 0.4999:  # Failure if half or more fits are bad
        fail_2L = True
    else:
        fail_2L = False

    # remove bad fits (if any exist and if we are keeping this model)
    if len(bad_idxs) and not (fail_1L or fail_2L):
        good_idxs = np.flatnonzero([ii not in bad_idxs for ii in range(NUM_MODELS_TOTAL)])
        E2NN_MODELS = [ALL_E2NN_MODELS[i] for i in good_idxs]
    else:
        E2NN_MODELS = ALL_E2NN_MODELS

    return(E2NN_MODELS, xscale_obj, yscale_obj, fail_1L, fail_2L,
           ALL_E2NN_MODELS, bad_idxs)
