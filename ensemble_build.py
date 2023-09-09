import numpy as np

from rann_training import train_E2NN_RaNN_1L, train_E2NN_RaNN_2L
from ensemble_predict import call_emulators
from accuracy_metrics import NRMSE

def fit_ensemble(Nsamp, Ndim, Fe, X_train_unscaled, Y_train_unscaled, X_test_unscaled, Y_test_unscaled, azim_2d, elev_2d):
    """
    WARNING: EMULATORS ARE SCALED THE SAME WAY AS THE HF TRAIING DATA. If EMULATORS
    HAVE A MUCH DIFFERENT SCALE THAN THE HF DATA, IT IS LIKELY BETTER TO SCALE THEM 
    SEPERATELY.
    """

    ################################################################################
    """SCALE TRAINING DATA"""

    # scale data from [-1, 1] using sk-learn
    from sklearn.preprocessing import MinMaxScaler
    xscale_obj = MinMaxScaler(feature_range=(-1, 1)).fit(X_train_unscaled) #feature_range=(-1, 1)
    yscale_obj = MinMaxScaler(feature_range=(-1, 1)).fit(Y_train_unscaled) #feature_range=(-1, 1)

    X_train = xscale_obj.transform(X_train_unscaled)
    Y_train = yscale_obj.transform(Y_train_unscaled)

#     ################################################################################
#     """SCALE TEST DATA"""
# 
#     # scale to fit sample data
#     X_test_scaled = xscale_obj.transform(X_test_unscaled)
#     Y_test_scaled = yscale_obj.transform(Y_test_unscaled.reshape(-1, 1))
#     Y_test_scaled = Y_test_scaled.flatten()

    ################################################################################
    """GET EMULATOR VALUES FOR TRAINING"""
    Emulator_train = call_emulators(X_train_unscaled, Fe, yscale_obj)

    ################################################################################
    ################################################################################
    """                                 (2) E2NN                                 """
    ################################################################################
    ################################################################################

    ############################################################################
    """SET PARAMETERS AND INITIALIZE VARIABLES"""

    num_models = 16 #32 #4 #3 #10 #32 #5 #64 #16 #5 #

    # opt = keras.optimizers.Adam(learning_rate=0.001) #0.01)# 0.001 default
    # reg = keras.regularizers.l2(1e-6) #1e-5) #0)#1e-3) #1e-13)#REGULARIZATION) #keras.regularizers.l1(1e-6) #default 0.01
    # for 1 hidden layer I found 1e-4 too high (500 neuron, 500 samp, 10D rosenbrock)
    dtype = "float64" #"float32" #

    Nemulator = len(Fe)
    E2NN_MODELS = [None]*num_models
    BAD_E2NN_MODELS = []
    E2NN_NRMSES = np.nan*np.ones(num_models)         # accuracy check
    E2NN_MODEL_LOSSES = np.nan*np.ones(num_models)   # convergence check

    run_idxs = list(range(num_models))
    NUM_MODELS_TOTAL = num_models
    

    MDL1, WT1 = train_E2NN_RaNN_1L(X_train, Y_train, Emulator_train, Nemulator=Nemulator, dtype=dtype, Neurons="few", Activation="fourier_low_1layer")
    MDL2, WT2 = train_E2NN_RaNN_1L(X_train, Y_train, Emulator_train, Nemulator=Nemulator, dtype=dtype, Neurons="few", Activation="fourier_med_1layer")
    MDL3, WT3 = train_E2NN_RaNN_1L(X_train, Y_train, Emulator_train, Nemulator=Nemulator, dtype=dtype, Neurons="few", Activation="fourier_high_1layer")
    MDL4, WT4 = train_E2NN_RaNN_1L(X_train, Y_train, Emulator_train, Nemulator=Nemulator, dtype=dtype, Neurons="few", Activation="swish")
    MDL5, WT5 = train_E2NN_RaNN_2L(X_train, Y_train, Emulator_train, Nemulator=Nemulator, dtype=dtype, Neurons="many", Activation="fourier_low_2layer")
    MDL6, WT6 = train_E2NN_RaNN_2L(X_train, Y_train, Emulator_train, Nemulator=Nemulator, dtype=dtype, Neurons="many", Activation="fourier_med_2layer")
    MDL7, WT7 = train_E2NN_RaNN_2L(X_train, Y_train, Emulator_train, Nemulator=Nemulator, dtype=dtype, Neurons="many", Activation="fourier_high_2layer")
    MDL8, WT8 = train_E2NN_RaNN_2L(X_train, Y_train, Emulator_train, Nemulator=Nemulator, dtype=dtype, Neurons="many", Activation="swish")

    MDL9,  WT9  = train_E2NN_RaNN_1L(X_train, Y_train, Emulator_train, Nemulator=Nemulator, dtype=dtype, Neurons="few", Activation="fourier_low_1layer")
    MDL10, WT10 = train_E2NN_RaNN_1L(X_train, Y_train, Emulator_train, Nemulator=Nemulator, dtype=dtype, Neurons="few", Activation="fourier_med_1layer")
    MDL11, WT11 = train_E2NN_RaNN_1L(X_train, Y_train, Emulator_train, Nemulator=Nemulator, dtype=dtype, Neurons="few", Activation="fourier_high_1layer")
    MDL12, WT12 = train_E2NN_RaNN_1L(X_train, Y_train, Emulator_train, Nemulator=Nemulator, dtype=dtype, Neurons="few", Activation="swish")
    MDL13, WT13 = train_E2NN_RaNN_2L(X_train, Y_train, Emulator_train, Nemulator=Nemulator, dtype=dtype, Neurons="many", Activation="fourier_low_2layer")
    MDL14, WT14 = train_E2NN_RaNN_2L(X_train, Y_train, Emulator_train, Nemulator=Nemulator, dtype=dtype, Neurons="many", Activation="fourier_med_2layer")
    MDL15, WT15 = train_E2NN_RaNN_2L(X_train, Y_train, Emulator_train, Nemulator=Nemulator, dtype=dtype, Neurons="many", Activation="fourier_high_2layer")
    MDL16, WT16 = train_E2NN_RaNN_2L(X_train, Y_train, Emulator_train, Nemulator=Nemulator, dtype=dtype, Neurons="many", Activation="swish")


    fourier_1_layer_idx = np.array([0,1,2,8,9,10])
    fourier_2_layer_idx = np.array([4,5,6,12,13,14])

    E2NN_MODELS = [MDL1, MDL2, MDL3, MDL4, MDL5, MDL6, MDL7, MDL8, MDL9, MDL10, MDL11, MDL12, MDL13, MDL14, MDL15, MDL16]
    MAX_E2NN_WEIGHTS = np.array([WT1, WT2, WT3, WT4, WT5, WT6, WT7, WT8, WT9, WT10, WT11, WT12, WT13, WT14, WT15, WT16])

    assert len(E2NN_MODELS)==num_models, f"num_models given as {num_models}, but {len(E2NN_MODELS)} models found in list"

    for jj in range(len(run_idxs)):

        ii = run_idxs[jj]
        E2NN_model = E2NN_MODELS[ii]

        ################################################################################
        """SAVE E2NN MODEL"""
        # E2NN_model.save("saved_models/model_"+str(ii))

        ################################################################################
        """CHECK ERROR"""
        # Train Error
        Y_E2NN_train = E2NN_model.predict([X_train, Emulator_train])
        Y_E2NN_train = Y_E2NN_train.flatten()
        E2NN_train_NRMSE = NRMSE(Y_E2NN_train, Y_train)

        E2NN_NRMSES[ii] = E2NN_train_NRMSE

    ################################################################################
    """SELECT E2NN MODELS TO RETRAIN (BAD_E2NN_MODELS) (if none, converge)"""

    # nrmse not above tolerance (needed when many bad fits exist)
    RAW_ERR_TOL = 0.001 #0.005 #0.01 #
    bad_raw_err = E2NN_NRMSES > RAW_ERR_TOL

    WEIGHT_TOL = 100 #50 #1000
    bad_weights = MAX_E2NN_WEIGHTS > WEIGHT_TOL

    # bad indices true/false
    bad_idxs_tf = bad_raw_err | bad_weights
    bad_idxs = np.flatnonzero(bad_idxs_tf)

    # bad fraction or 1L fourier and 2L fourier
    bad_idxs_tf_1L_fourier = bad_idxs_tf[fourier_1_layer_idx]
    bad_frac_1L_fourier = np.sum(bad_idxs_tf_1L_fourier)/bad_idxs_tf_1L_fourier.size
    bad_idxs_tf_2L_fourier = bad_idxs_tf[fourier_2_layer_idx]
    bad_frac_2L_fourier = np.sum(bad_idxs_tf_2L_fourier)/bad_idxs_tf_2L_fourier.size

    # for plotting 
    ALL_E2NN_MODELS = E2NN_MODELS.copy()

    # check fitting failure individually for 2L fourier and 1L fourier (if most bad will adjust fourier frequency)
    if bad_frac_1L_fourier > 0.4999: # Failure if half or more fits are bad
        fail_1L = True
    else:
        fail_1L = False

    if bad_frac_2L_fourier > 0.4999: # Failure if half or more fits are bad
        fail_2L = True
    else:
        fail_2L = False

    # remove bad fits (if any exist and if we are keeping this model)
    if len(bad_idxs) and not (fail_1L or fail_2L):
        BAD_E2NN_MODELS += [E2NN_MODELS[ii] for ii in bad_idxs]
        good_idxs = np.flatnonzero([ii not in bad_idxs for ii in range(num_models)])
        E2NN_MODELS = [E2NN_MODELS[ii] for ii in good_idxs]

    # return(E2NN_MODELS, E2NN_train_NRMSE, E2NN_test_NRMSE, xscale_obj, yscale_obj, fail_1L, fail_2L)
    return(E2NN_MODELS, xscale_obj, yscale_obj, fail_1L, fail_2L, ALL_E2NN_MODELS, bad_idxs)