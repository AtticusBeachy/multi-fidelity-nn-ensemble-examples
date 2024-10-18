# Imports
import numpy as np
import pickle
import time
import tensorflow as tf
# add custom activation functions
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras.layers import Activation

from build_ensemble import build_ensemble
from predict_ensemble import predict_ensemble
from minimize_global import minimize_global

from get_training_doe import get_training_doe
from get_test_doe import get_test_doe

from get_acquisition_ei import get_acquisition_ei
from get_acquisition_uncertainty import get_acquisition_uncertainty

from err_nrmse import err_nrmse
from err_sum_log_lh_tdist import err_sum_log_lh_tdist

from plot_ensemble_details import plot_ensemble_details
from plot_1d_or_2d_acquisition import plot_1d_or_2d_acquisition
from plot_convergence_history import plot_convergence_history

from user_defined_test_functions import rosenbrock, rosen_univariate, \
    get_rosen_emulator, return_sum_emulator, scaled_sphere_rosenbrock, \
    nonstationary_1d_hf, nonstationary_1d_lf, uninformative_1d, \
    nonstationary_2d_hf, nonstationary_2d_lf, uninformative_nd_lf, \
    viscous_simulation_cl_cd

################################################################################
################################################################################
"""                       (0) USER SPECIFIED VARIABLES                       """
################################################################################
################################################################################

resume_saved_state = True #False # #If True, will try to load saved data. Generates data on failure

# Rejection tolerances for individual e2nn models
ERR_TOL = 0.001 # max error
WEIGHT_TOL = 100    # max weight

TEST_TRUE_FUNCTION = True #False #  # set to False if HF function is too expensive to gather test data

N_TEST = 10_000 #1000 #1000 # 
N_SCATTER_GAUSS = 128 #32 #64 #256 #

ACQUIS_TOL = 1e-4 #1e-5 #0.001 #1e-7 #0.001 #np.inf #0.01 # This depends on the data scale
# SOBOL_TOL = 1e-3 #4e-3 #1e-2 #  # good value depends on Ndim
# EM_GRID_DENSITY = 7 #11 #
MAX_SAMP = 1000 #500 #200 #150 #100 #300 #101 #320 #0 #50 #10 #np.inf
STEPS_TO_CONVERGE = 3 #1 #2 #
converged_steps = 0
adaptive_sampling_converged = False

N_COPY_ARCH = 2  # number of copies of each NN architecture

ensemble_settings = N_COPY_ARCH*[
    {"n_layers": 1, "activation_name": "fourier_low_1layer"},
    {"n_layers": 1, "activation_name": "fourier_med_1layer"},
    {"n_layers": 1, "activation_name": "fourier_high_1layer"},
    {"n_layers": 1, "activation_name": "swish"},

    {"n_layers": 2, "activation_name": "fourier_low_2layer"},
    {"n_layers": 2, "activation_name": "fourier_med_2layer"},
    {"n_layers": 2, "activation_name": "fourier_high_2layer"},
    {"n_layers": 2, "activation_name": "swish"},
]

OPTIMIZATION_PROBLEM = True #False #



################################################################################
################################################################################
"""                       (0) USER SPECIFIED VARIABLES                       """
################################################################################
################################################################################



# """ 1D problem """
# N_DIM = 1
# n_train = 3
# N_TEST = 201
# TRUE_FUNCTION = nonstationary_1d_hf
# # emulator_functions = [nonstationary_1d_lf]
# emulator_functions = [uninformative_1d]
# LB = np.zeros([N_DIM]) #0.0 # -2 + np.zeros([N_DIM]) #np.array([-2, -2])
# UB = np.ones([N_DIM]) #1.0 #  2 + np.zeros([N_DIM]) #np.array([ 2,  2])
# X_TRUE_OPT = np.array([[0.75724875784185587]]) # 0.7572487578418558700469473965334781194850722869852168660523
# Y_TRUE_OPT = -6.02074005576708278655 #TRUE_FUNCTION(X_TRUE_OPT) #-6.02074005576708278655396973488822950268123099501973746419 #
# 


""" 2D problem """
N_DIM = 2 #10 #1 #20 #3 #3 #1 #2 #
n_train = 8 #(N_DIM+1)*(N_DIM+2)//2 #3 #100 #3000 #2000 #1000 #200 #10*N_DIM 
TRUE_FUNCTION = nonstationary_2d_hf
emulator_functions = [nonstationary_2d_lf]
# emulator_functions = [uninformative_nd_lf]
LB = np.array([0.05, 0]) #0.0 # -2 + np.zeros([N_DIM]) #np.array([-2, -2])
UB = np.array([1.05, 1])
X_TRUE_OPT = np.array([[0.221173397155865137351525548116, 0.0]]) #np.array([[0.22117339715586513735, 0.0]]) #
Y_TRUE_OPT = -0.44420104


# """ Rosenbrock problem """
# # Rosenbrock (nd)
# N_DIM = 6 #4 #20 #10 #1 #3 #3 #1 #2 #
# n_train = (N_DIM+1)*(N_DIM+2)//2 #8 #
# TRUE_FUNCTION = rosenbrock
# # emulator_functions = [uninformative_nd_lf]
# # emulator_functions = [scaled_sphere_rosenbrock]
# emulator_functions = []
# for ii in range(N_DIM):
#     emulator = get_rosen_emulator(dim=ii, N_DIM=N_DIM)
#     emulator_functions.append(emulator)
# 
# # sum emulator
# emulator_sum = return_sum_emulator(emulator_functions.copy())
# emulator_functions.append(emulator_sum)
# 
# LB = -2*np.ones([N_DIM]) #0.0 # -2 + np.zeros([N_DIM]) #np.array([-2, -2])
# UB = 2*np.ones([N_DIM]) #1.0 #  2 + np.zeros([N_DIM]) #np.array([ 2,  2])
# 
# X_TRUE_OPT = np.ones(N_DIM).reshape(1, -1)
# Y_TRUE_OPT = 0.0



# """ Fun3d problem """
# 
# N_DIM = 3
# n_train = 10
# 
# LB = np.zeros([N_DIM])
# UB = np.ones([N_DIM])
# 
# X_TRUE_OPT = []
# Y_TRUE_OPT = []
# 
# subfolder2 = "GHV_300k_v" 
# out_name2  = "GHV02_300k" 
# 
# fun_v_ratio = lambda x : -viscous_simulation_cl_cd(x, out_name2, subfolder2)
# TRUE_FUNCTION = fun_v_ratio
# 
# # Get LF model from CFD training data
# filename_train = 'x_lf_inviscid_3d.pkl'
# infile = open(filename_train, 'rb')
# X_lf_train = pickle.load(infile)
# infile.close()
# 
# # LD Ratio
# filename_train = 'ycl_cd_lf_inviscid_3d.pkl'
# infile = open(filename_train, 'rb')
# y_lf_cl_cd = pickle.load(infile)
# infile.close()
# 
# # construct gpr model and train on data
# from sklearn.gaussian_process import GaussianProcessRegressor
# from sklearn.gaussian_process.kernels import Matern, RBF
# 
# GPR = GaussianProcessRegressor( # kernel = Matern()
#         kernel=1.0*RBF(1.0), alpha=1e-10, optimizer='fmin_l_bfgs_b',
#         n_restarts_optimizer=25) #250) #25
# 
# GPR.fit(X_lf_train, y_lf_cl_cd) # LD Ratio
# 
# LF_GPR = lambda x : GPR.predict(x, return_std=False).flatten()
# emulator_functions = [LF_GPR]


################################################################################
""" OTHER SET-UP """

physical_devices = tf.config.list_physical_devices("GPU")
print("physical devices: ", physical_devices)
# Don't crash if something else is also using the GPU
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

# 2d figures
azim_2d = -120 #-150 #-60 # -60 (default) #
elev_2d = 20 # 30 (default) #

SAVE_PATH = "./images/"


times_filename = "times.txt"


############################################################################
""" INITIALISE """

if resume_saved_state:
    try:
        # load state
        filename = "state_to_load.pkl"
        with open(filename, 'rb') as infile:
            current_state = pickle.load(infile)

        x_train_raw = current_state["x_train_raw"]
        y_train_raw = current_state["y_train_raw"]
        x_test_raw = current_state["x_test_raw"]
        y_test_raw = current_state["y_test_raw"]
        global_parallel_acquisitions = current_state["global_parallel_acquisitions"]
        EIs = current_state["EIs"]
        EIs_at_true_opt = current_state["EIs_at_true_opt"]
        nrmse_s = current_state["nrmse_s"]
        sum_log_likelihoods = current_state["sum_log_likelihoods"]
        samples_each_iter = current_state["samples_each_iter"]
        min_idx = current_state["min_idx"]
        Yopts = current_state["Yopts"]
        Xopts = current_state["Xopts"]
        fourier_factor_1L = current_state["fourier_factor_1L"]
        fourier_factor_2L = current_state["fourier_factor_2L"]

        n_train = len(y_train_raw)

    except:
        print("Loading run state failed. Generating fresh run state.")
        resume_saved_state = False

if not resume_saved_state:

    ###########################################################################
    """GET TRAINING DATA"""

    x_train_raw = get_training_doe(n_train, N_DIM, LB, UB)
    y_train_raw = TRUE_FUNCTION(x_train_raw)
    y_train_raw = y_train_raw.reshape(-1, 1)

    ###########################################################################
    """GET TEST DATA"""

    x_test_raw = get_test_doe(N_TEST, N_DIM, LB, UB)

    if TEST_TRUE_FUNCTION:
        y_test_raw = TRUE_FUNCTION(x_test_raw)
    else:
        y_test_raw = None
    
    global_parallel_acquisitions = 0
    EIs = np.empty([0,1]) #[]
    EIs_at_true_opt = np.empty([0,1])
    nrmse_s = np.empty([0,1]) #[]
    sum_log_likelihoods = np.empty([0,1]) #[]
    samples_each_iter = np.empty([0,1])
    min_idx = np.argmin(y_train_raw)
    Yopts = np.array(np.min(y_train_raw))
    Xopts = x_train_raw[min_idx,:].reshape(1,-1)
    fourier_factor_1L = 1
    fourier_factor_2L = 1


################################################################################
""" RUN ACTIVE LEARNING """
################################################################################


class Tdists:
    def __init__(self, mu, sig, df):
        self.mu = mu
        self.sig = sig
        self.df = df

def fourier_activation_lambda(coeff):
    fn = lambda x : tf.sin(coeff*x)
    return(fn)

while not adaptive_sampling_converged:

    tic = time.perf_counter()
    samples_each_iter = np.vstack([samples_each_iter, n_train])

    # Set Fourier activation function frequencies such that at least half of 
    # NNs using the frequncy pass checks.
    frequencies_good = False
    while not frequencies_good:
        # # Add scaled fourier functions as custom activation functions
        # # (This works in Tensorflow 2.10.0 but is broken in TensorFlow 2.17.0)
        # # (Instead, create lambda functions for the activations)

        # fourier_low_1layer = fourier_activation_lambda(1*fourier_factor_1L)
        # fourier_med_1layer = fourier_activation_lambda(1.1*fourier_factor_1L)
        # fourier_high_1layer = fourier_activation_lambda(1.2*fourier_factor_1L)

        # get_custom_objects().update(
        #     {"fourier_low_1layer": Activation(fourier_low_1layer)})
        # get_custom_objects().update(
        #     {"fourier_med_1layer": Activation(fourier_med_1layer)})
        # get_custom_objects().update(
        #     {"fourier_high_1layer": Activation(fourier_high_1layer)})


        # fourier_low_2layer = fourier_activation_lambda(1*fourier_factor_2L)
        # fourier_med_2layer = fourier_activation_lambda(1.1*fourier_factor_2L)
        # fourier_high_2layer = fourier_activation_lambda(1.2*fourier_factor_2L)

        # get_custom_objects().update(
        #     {"fourier_low_2layer": Activation(fourier_low_2layer)})
        # get_custom_objects().update(
        #     {"fourier_med_2layer": Activation(fourier_med_2layer)})
        # get_custom_objects().update(
        #     {"fourier_high_2layer": Activation(fourier_high_2layer)})

        
        # Add activation functions to settings:
        for settings in ensemble_settings:
            activation_name = settings["activation_name"]
            if activation_name == "swish":
                settings["activation_function"] = "swish"

            elif activation_name == "fourier_low_1layer":
                settings["activation_function"] = fourier_activation_lambda(1.0*fourier_factor_1L)
            elif activation_name == "fourier_med_1layer":
                settings["activation_function"] = fourier_activation_lambda(1.1*fourier_factor_1L)
            elif activation_name == "fourier_high_1layer":
                settings["activation_function"] = fourier_activation_lambda(1.2*fourier_factor_1L)

            elif activation_name == "fourier_low_2layer":
                settings["activation_function"] = fourier_activation_lambda(1.0*fourier_factor_2L)
            elif activation_name == "fourier_med_2layer":
                settings["activation_function"] = fourier_activation_lambda(1.1*fourier_factor_2L)
            elif activation_name == "fourier_high_2layer":
                settings["activation_function"] = fourier_activation_lambda(1.2*fourier_factor_2L)


        ########################################################################
        """ TRAIN ENSEMBLE """

        # important to prevent slowdown and memory bloat over iterations
        tf.keras.backend.clear_session() 

        print("Training E2NN ensemble")

        (e2nn_models, xscale_obj, yscale_obj, fail_1L, fail_2L, ALL_E2NN_MODELS,
        bad_idxs) = build_ensemble(emulator_functions, x_train_raw, y_train_raw,
                                   LB, UB, ensemble_settings=ensemble_settings, 
                                   ERR_TOL=ERR_TOL, WEIGHT_TOL=WEIGHT_TOL)

        fitting_failure = fail_1L or fail_2L
        print("Done training E2NN ensemble")

        ########################################################################
        """ PLOT FIT """
        print("Plotting ensemble details")

        plot_ensemble_details(e2nn_models, ALL_E2NN_MODELS, bad_idxs, 
                              emulator_functions, LB, UB, 
                              xscale_obj, yscale_obj, 
                              x_train_raw, y_train_raw, 
                              x_test_raw, y_test_raw, 
                              fourier_factor_1L, fourier_factor_2L, 
                              azim_2d, elev_2d)

        print("End plotting ensemble details")

        if fail_1L:
            fourier_factor_1L += 1 #*= 2 #*=1.2 #
        if fail_2L:
            fourier_factor_2L += 1 #*= 2 #*=1.2 #

        if not (fail_1L or fail_2L):
            frequencies_good = True

    toc = time.perf_counter()

    with open(times_filename,"a") as time_doc:
        time_doc.write(f"time for training ensemble: {toc-tic} s"+"\n")
    
    ############################################################################
    """ CALCULATE ENSEMBLE NRMSE AND LIKELIHOOD RATIO (TRAINING DATA) """

    tic = time.perf_counter()

    y_train_pred, sig_train_pred, df = predict_ensemble(
        x_train_raw, e2nn_models, emulator_functions, xscale_obj, yscale_obj
    )

    residuals = y_train_pred - y_train_raw
    # 1e-6 term avoids divide by zero and allows small errors
    t_scores = abs(residuals)/(sig_train_pred + 1e-6)  

    e2nn_train_nrmse = err_nrmse(y_train_pred, y_train_raw)

    toc = time.perf_counter()

    with open(times_filename,"a") as time_doc:
        time_doc.write(f"time getting training performance: {toc-tic} s"+"\n")

    ############################################################################
    """ ACQUISITION TO GET NEW SAMPLE """
    print("Beginning acquisition optimization")

    tic = time.perf_counter()

    if OPTIMIZATION_PROBLEM:
        # expected improvement for minimization
        obj_fn = lambda x : -get_acquisition_ei(x, e2nn_models,
                                                emulator_functions, 
                                                np.min(y_train_raw), 
                                                xscale_obj, yscale_obj)
    else:
        # add data at location of highest uncertainty
        obj_fn = lambda x : -get_acquisition_uncertainty(x, e2nn_models,
                                                         emulator_functions,
                                                         xscale_obj, yscale_obj)

    # Perform global optimization
    x_new, f_opt, x_opts, f_opts, *__ = minimize_global(
        obj_fn, LB, UB, N_DIM, global_parallel_acquisitions, 
        n_scatter_init=100_000, n_scatter_check=1000, n_local_opts=10, 
        previous_local_xopt=np.array([]), n_scatter_gauss=N_SCATTER_GAUSS
    )

    toc = time.perf_counter()

    with open(times_filename,"a") as time_doc:
        time_doc.write(f"optimizing acquisition function: {toc-tic} s"+"\n")

    ############################################################################
    """ MISC (RUNNING TESTS AND COMPARISONS, SAVING RESULTS)"""

    tic = time.perf_counter()

    acquis_opt = -f_opt
    EIs = np.vstack([EIs, acquis_opt])

    if len(X_TRUE_OPT):
        EIs_at_true_opt = np.vstack([
            EIs_at_true_opt, float(-obj_fn(X_TRUE_OPT)[0])
        ])

        mu_opt, sig_opt, df = predict_ensemble(X_TRUE_OPT, e2nn_models, 
                                               emulator_functions, 
                                               xscale_obj, yscale_obj)

        t_dist_opt = Tdists(mu_opt, sig_opt, df)
    else:
        EIs_at_true_opt = []
        t_dist_opt = []


    if TEST_TRUE_FUNCTION:
        y_e2nn_test, sig_e2nn_test, df = predict_ensemble(x_test_raw, 
                                                          e2nn_models, 
                                                          emulator_functions, 
                                                          xscale_obj, 
                                                          yscale_obj)

        t_dists = Tdists(y_e2nn_test, sig_e2nn_test, df)
        e2nn_test_nrmse = err_nrmse(y_e2nn_test, y_test_raw)
        e2nn_test_sum_log_likelihood = err_sum_log_lh_tdist(y_test_raw, t_dists)

        nrmse_s = np.vstack([nrmse_s, e2nn_test_nrmse])
        sum_log_likelihoods = np.vstack([
            sum_log_likelihoods, float(e2nn_test_sum_log_likelihood)
        ])

    # write training results to text document
    with open(SAVE_PATH+f"Train results ({n_train} samples).txt","w") as doc:
        doc.write("e2nn_test_nrmse: "+str(e2nn_test_nrmse)+"\n")
        doc.write("e2nn_train_nrmse: "+str(e2nn_train_nrmse)+"\n")
        doc.write("T_distributions:"+"\n")
        doc.write("y_train_pred: "+str(y_train_pred)+"\n")
        doc.write("sig_train_pred: "+str(sig_train_pred)+"\n")
        doc.write("df: "+str(df)+"\n")
        doc.write("residuals: "+str(residuals)+"\n")
        doc.write("t_scores: "+str(t_scores)+2*"\n")


    ############################################################################
    """ PLOTTING ACQUISITION """
    print("Plotting acquisition")
    if N_DIM == 1 or N_DIM == 2:
        plot_1d_or_2d_acquisition(obj_fn, x_train_raw, x_test_raw, x_new, acquis_opt, x_opts, azim_2d, elev_2d)
    print("End plotting acquisition")

    toc = time.perf_counter()

    with open(times_filename,"a") as time_doc:
        time_doc.write(f"plotting acquisition: {toc-tic} s"+"\n")



    ############################################################################
    """ ADD NEW SAMPLE UNLESS CONVERGED"""

    tic = time.perf_counter()

    data_range = np.max(y_train_raw) - np.min(y_train_raw)
    if acquis_opt < ACQUIS_TOL*data_range:
        converged_steps += 1
    else:
        converged_steps = 0

    if converged_steps >= STEPS_TO_CONVERGE or n_train >= MAX_SAMP:
        adaptive_sampling_converged = True
    else:

        if N_DIM != 1:
            x_new = x_new.reshape(1, -1) # make x_new 2D

        y_new = TRUE_FUNCTION(x_new)
        if ~np.isnan(y_new):  # if simulation succeeded and returned a value
            x_train_raw = np.vstack([x_train_raw, x_new])
            y_train_raw = np.vstack([y_train_raw, y_new])
            n_train = y_train_raw.size

        min_idx = np.argmin(y_train_raw)
        Yopts = np.hstack([Yopts, np.min(y_train_raw)])
        Xopts = np.vstack([Xopts, x_train_raw[min_idx,:]])

    doc1 = open(SAVE_PATH+"Ultimate results.txt","w")
    doc1.write(f"x_train_raw: {x_train_raw}\n")
    doc1.write(f"y_train_raw: {y_train_raw}\n")
    doc1.write(f"EIs: {EIs}\n")
    doc1.write(f"Yopts: {Yopts}\n")
    doc1.write(f"nrmse_s: {nrmse_s}\n")
    doc1.write(f"sum_log_likelihoods: {sum_log_likelihoods}\n")
    doc1.write(f"samples_each_iter: {samples_each_iter}\n")
    doc1.close()


    ############################################################################
    """ PLOT CONVERGENCE HISTORY """
    print("Plotting convergence history")
    plot_convergence_history(
        Xopts, Yopts, x_train_raw, y_train_raw, EIs, nrmse_s, 
        sum_log_likelihoods, samples_each_iter, X_TRUE_OPT, Y_TRUE_OPT,
        t_dist_opt, EIs_at_true_opt)
    print("End plotting convergence history")

    toc = time.perf_counter()

    with open(times_filename,"a") as time_doc:
        time_doc.write(f"short finishing stuff: {toc-tic} s"+2*"\n")

    ############################################################################
    """ SAVE CURRENT PROGRAM STATE """

    filename = "state_to_load.pkl"
    
    current_state = {
        "x_train_raw":x_train_raw,
        "y_train_raw":y_train_raw,
        "x_test_raw":x_test_raw,
        "y_test_raw":y_test_raw,
        "global_parallel_acquisitions":global_parallel_acquisitions,
        "EIs":EIs,
        "EIs_at_true_opt":EIs_at_true_opt,
        "nrmse_s":nrmse_s,
        "sum_log_likelihoods":sum_log_likelihoods,
        "min_idx":min_idx,
        "Yopts":Yopts,
        "Xopts":Xopts,
        "fourier_factor_1L":fourier_factor_1L,
        "fourier_factor_2L":fourier_factor_2L,
    }

    with open(filename, 'wb') as outfile:
        pickle.dump(current_state, outfile)

print("Adaptive Sampling Complete")


