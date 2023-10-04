# Imports
import numpy as np
import math
import pickle
import tensorflow as tf
from tensorflow import keras
# add custom activation functions
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras.layers import Activation
# predictions from trained ensembles
from scipy.stats import t

from numpy.random import default_rng
rng = default_rng()

from rann_training import train_E2NN_RaNN_1L, train_E2NN_RaNN_2L
from ensemble_build import fit_ensemble
from ensemble_predict import predict_ensemble, ensemble_fn
from user_defined_test_functions import rosenbrock, rosen_univariate, get_rosen_emulator, sum_emulator, return_sum_emulator, nonstationary_1d_hf, nonstationary_1d_lf, uninformative_1d, nonstationary_2d_hf, nonstationary_2d_lf, uninformative_nd_lf, viscous_simulation_cl_cd
from optimization_functions import global_optimization
from accuracy_metrics import NRMSE, sum_log_likelihood_tdist
from plotting_functions import plot_ensemble_details, plot_1d_or_2d_acquisition, plot_convergence_history
from design_of_experiments_functions import get_training_doe, get_test_doe

################################################################################
################################################################################
"""                       (1) USER SPECIFIED VARIABLES                       """
################################################################################
################################################################################

load_data = False #True # #If True, will try to load saved data. Generates data on failure

# Rejection tolerances for individual e2nn models
RAW_ERR_TOL = 0.001 # max error
WEIGHT_TOL = 100    # max weight

TEST_HF_FUNCTION = True #False # set to False if HF function is too expensive to gather test data

Ntest = 10_000 #1000 #1000 # use fewer when plotting test error convergence
n_scatter_gauss = 128 #32 #64 #256 #

EI_TOL = 1e-4 #1e-5 #0.25 #3 #0.001 #1e-7 #0.001 #np.inf #0.01 # This depends on the data scale
MAX_SAMP = 1000 #500 #200 #150 #100 #300 #101 #320 #0 #50 #10 #np.inf
STEPS_TO_CONVERGE = 3 #1 #2 #


# """ 1D problem """
# Ndim = 1
# Nsamp = 3
# Ntest = 201
# F = nonstationary_1d_hf
# Fe = [nonstationary_1d_lf]
# # Fe = [uninformative_1d]
# lb = np.zeros([Ndim]) #0.0 # -2 + np.zeros([Ndim]) #np.array([-2, -2])
# ub = np.ones([Ndim]) #1.0 #  2 + np.zeros([Ndim]) #np.array([ 2,  2])
# x_true_opt = np.array([[0.75724875784185587]]) # 0.7572487578418558700469473965334781194850722869852168660523
# y_true_opt = -6.02074005576708278655 #F(x_true_opt) #-6.02074005576708278655396973488822950268123099501973746419 #


""" 2D problem """
Ndim = 2 #10 #1 #20 #3 #3 #1 #2 #
Nsamp = 8 #(Ndim+1)*(Ndim+2)//2 #3 #100 #3000 #2000 #1000 #200 #10*Ndim 
F = nonstationary_2d_hf
Fe = [nonstationary_2d_lf]
# Fe = [uninformative_nd_lf]
lb = np.array([0.05, 0]) #0.0 # -2 + np.zeros([Ndim]) #np.array([-2, -2])
ub = np.array([1.05, 1])
x_true_opt = np.array([[0.221173397155865137351525548116, 0.0]]) #np.array([[0.22117339715586513735, 0.0]]) #
y_true_opt = -0.44420104


# """ Rosenbrock problem """
# # Rosenbrock (nd)
# Ndim = 20 #10 #2 #1 #3 #3 #1 #2 #
# Nsamp = (Ndim+1)*(Ndim+2)//2 #8 #
# F = rosenbrock
# # Fe = [uninformative_nd_lf]
# # Fe = [scaled_sphere_rosenbrock]
# Fe = []
# for ii in range(Ndim):
#     # emulator = lambda X : rosen_univariate(X, dim = ii, Ndim = Ndim)
#     emulator = get_rosen_emulator(dim=ii, Ndim=Ndim)
#     Fe.append(emulator)
# 
# # sum emulator
# emulator_sum = return_sum_emulator(Fe.copy())
# Fe.append(emulator_sum)
# 
# lb = -2*np.ones([Ndim]) #0.0 # -2 + np.zeros([Ndim]) #np.array([-2, -2])
# ub = 2*np.ones([Ndim]) #1.0 #  2 + np.zeros([Ndim]) #np.array([ 2,  2])
# 
# x_true_opt = np.ones(Ndim).reshape(1,-1)
# y_true_opt = 0.0


# """ Fun3d problem """
# 
# Ndim = 3
# Nsamp = 10
# 
# lb = np.zeros([Ndim])
# ub = np.ones([Ndim])
# 
# x_true_opt = []
# y_true_opt = []
# 
# subfolder2 = "GHV_300k_v" 
# out_name2  = "GHV02_300k" 
# 
# fun_v_ratio = lambda x : -viscous_simulation_cl_cd(x, out_name2, subfolder2)
# F = fun_v_ratio
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
# Fe = [LF_GPR]

################################################################################
################################################################################
"""                            (2) PROBLEM SET UP                            """
################################################################################
################################################################################


################################################################################
""" SET UP TENSORFLOW """

# Use GPU
physical_devices = tf.config.list_physical_devices("GPU")
print("physical devices: ", physical_devices)
# Don't crash if something else is also using the GPU
tf.config.experimental.set_memory_growth(physical_devices[0], True)


################################################################################
""" OTHER SET-UP """

Nemulator = len(Fe)
converged_steps = 0
adaptive_sampling_converged = False

azim_2d = -120 #-150 #-60 # -60 (default) #
elev_2d = 20 # 30 (default) #


eps64 = np.finfo("float").eps
eps32 = np.finfo("float32").eps
eps = 1e-6 #1e-3 #

SAVE_PATH = "./images/"


############################################################################
""" ADAPTIVE SAMPLING ACQUISITION FUNCTIONS """

def ensemble_ei(X_unscaled, E2NN_MODELS, Fe, xscale_obj, yscale_obj):
    """
    MAXIMIZE ACQUISITION
    For T-distribution:
        z = (f_best-mu)/sig
        EI = (f_best-mu)*CDF(z) + df/(df-1) * (1+z**2/df)*sig*PDF(z)
    """
    # if X is 1d, assume it is a single sample and reshape to 2D
    if len(np.shape(X_unscaled)) == 1:
        X_unscaled = X_unscaled.reshape(1, -1)

    #from scipy.stats import t
    f_best = np.min(Y_train_unscaled)
    mu, sig, df = ensemble_fn(X_unscaled, E2NN_MODELS, Fe, xscale_obj, yscale_obj)
    z = (f_best-mu)/sig
    EI = (f_best-mu)*t.cdf(z, df) + df/(df-1) * (1+z**2/df)*sig*t.pdf(z, df)
    EI = EI.flatten()
    return(EI)


############################################################################
""" INITIALISE """

if load_data:
    try:

        # load state
        filename = "state_to_load.pkl"
        
        with open(filename, 'rb') as infile:
            current_state = pickle.load(infile)

        X_train_unscaled = current_state["X_train_unscaled"]
        Y_train_unscaled = current_state["Y_train_unscaled"]
        X_test_unscaled = current_state["X_test_unscaled"]
        Y_test_unscaled = current_state["Y_test_unscaled"]
        global_parallel_acquisitions = current_state["global_parallel_acquisitions"]
        EIs = current_state["EIs"]
        EIs_at_true_opt = current_state["EIs_at_true_opt"]
        NRMSEs = current_state["NRMSEs"]
        SumLogLikelihoods = current_state["SumLogLikelihoods"]
        min_idx = current_state["min_idx"]
        Yopts = current_state["Yopts"]
        Xopts = current_state["Xopts"]
        fourier_factor_1L = current_state["fourier_factor_1L"]
        fourier_factor_2L = current_state["fourier_factor_2L"]

        Nsamp = len(Y_train_unscaled)
    except:
        load_data = False

if not load_data:

    ################################################################################
    """GET TRAINING DATA"""

    X_train = get_training_doe(Nsamp, Ndim, sample_corners=False)
    X_train_unscaled = (ub-lb)*X_train + lb
    Y_train_unscaled = F(X_train_unscaled)
    # reshape to 2D for scaling
    Y_train_unscaled = Y_train_unscaled.reshape(-1, 1)

    ################################################################################
    """GET TEST DATA"""

    X_test = get_test_doe(Ntest, Ndim)
    X_test_unscaled = (ub-lb)*X_test + lb

    if TEST_HF_FUNCTION:
        Y_test_unscaled = F(X_test_unscaled)
    else:
        Y_test_unscaled = None

    
    global_parallel_acquisitions = 0
    EIs = np.empty([0,1]) #[]
    EIs_at_true_opt = np.empty([0,1])
    NRMSEs = np.empty([0,1]) #[]
    SumLogLikelihoods = np.empty([0,1]) #[]
    min_idx = np.argmin(Y_train_unscaled)
    Yopts = np.array(np.min(Y_train_unscaled))
    Xopts = X_train_unscaled[min_idx,:].reshape(1,-1)
    fourier_factor_1L = 1
    fourier_factor_2L = 1
    


################################################################################
################################################################################
"""                  (3) ACTIVE LEARNING WITH E2NN ENSEMBLE                  """
################################################################################
################################################################################


def fourier_activation_lambda(coeff):
    fn = lambda x : tf.sin(coeff*x)
    return(fn)

while not adaptive_sampling_converged:

    good_frequencies = False
    while not good_frequencies:
        
        # Add scaled fourier functions as custom activation functions

        fourier_low_1layer = fourier_activation_lambda(1*fourier_factor_1L)
        fourier_med_1layer = fourier_activation_lambda(1.1*fourier_factor_1L) #1.5*fourier_factor_1L) #
        fourier_high_1layer = fourier_activation_lambda(1.2*fourier_factor_1L) #2*fourier_factor_1L) #

        get_custom_objects().update({"fourier_low_1layer": Activation(fourier_low_1layer)})
        get_custom_objects().update({"fourier_med_1layer": Activation(fourier_med_1layer)})
        get_custom_objects().update({"fourier_high_1layer": Activation(fourier_high_1layer)})


        fourier_low_2layer = fourier_activation_lambda(1*fourier_factor_2L)
        fourier_med_2layer = fourier_activation_lambda(1.1*fourier_factor_2L) #1.5*fourier_factor) #
        fourier_high_2layer = fourier_activation_lambda(1.2*fourier_factor_2L) #2*fourier_factor) #

        get_custom_objects().update({"fourier_low_2layer": Activation(fourier_low_2layer)})
        get_custom_objects().update({"fourier_med_2layer": Activation(fourier_med_2layer)})
        get_custom_objects().update({"fourier_high_2layer": Activation(fourier_high_2layer)})

        ################################################################################
        """ TRAIN ENSEMBLE """

        # important to prevent slowdown and memory bloat over iterations
        tf.keras.backend.clear_session() 

        print("Training E2NN ensemble")
        E2NN_MODELS, xscale_obj, yscale_obj, fail_1L, fail_2L, ALL_E2NN_MODELS, bad_idxs = fit_ensemble(Nsamp, Ndim, Fe, X_train_unscaled, Y_train_unscaled)

        fitting_failure = fail_1L or fail_2L
        print("Done training E2NN ensemble")

        ################################################################################
        """ PLOT FIT """
        print("Plotting ensemble details")
        plot_ensemble_details(E2NN_MODELS, ALL_E2NN_MODELS, bad_idxs, Fe, lb, ub, xscale_obj, yscale_obj, X_train_unscaled, Y_train_unscaled, X_test_unscaled, Y_test_unscaled, fourier_factor_1L, fourier_factor_2L, azim_2d, elev_2d)
        print("End plotting ensemble details")

        if fail_1L:
            fourier_factor_1L += 1 #*= 2 #
        if fail_2L:
            fourier_factor_2L += 1 #*= 2 #

        if not (fail_1L or fail_2L):
            good_frequencies = True

    
    ################################################################################
    """ CALCULATE NRMSE OF ENSEMBLE AND LIKELIHOOD RATIO """

    Y_e2nn_train, t_dists, *__ = predict_ensemble(X_train_unscaled, E2NN_MODELS, Fe, xscale_obj, yscale_obj)

    residuals = Y_e2nn_train - Y_train_unscaled
    t_scores = abs(residuals)/(t_dists.sig + eps) # Avoid divide by zero and allow small errors

    e2nn_train_nrmse = NRMSE(Y_e2nn_train, Y_train_unscaled)

    # write results to text document
    doc1 = open(SAVE_PATH+f"Results ({Nsamp} samples).txt","w")
    doc1.write("e2nn_train_nrmse: "+str(e2nn_train_nrmse)+"\n")
    doc1.write("T_distributions:"+"\n")
    doc1.write("t_dists.mu: "+str(t_dists.mu)+"\n")
    doc1.write("t_dists.sig: "+str(t_dists.sig)+"\n")
    doc1.write("t_dists.df: "+str(t_dists.df)+"\n")
    doc1.write("residuals: "+str(residuals)+"\n")
    doc1.write("t_scores: "+str(t_scores)+2*"\n")
    doc1.close()

    ################################################################################
    """ ACQUISITION TO GET NEW SAMPLE """
    print("Beginning acquisition optimization")
    # Expected Improvement !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    obj_fn = lambda x : -ensemble_ei(x, E2NN_MODELS, Fe, xscale_obj, yscale_obj)
    # preacquisition_fn = lambda x : ensemble_preacquisition_ei(x, E2NN_MODELS, Fe, xscale_obj, yscale_obj)

    # # Maximum uncertainty
    # def get_uncertainty(X_unscaled, E2NN_MODELS, Fe, xscale_obj, yscale_obj):
    #     (Y_ensemble, t_dists, *__) = predict_ensemble(X_unscaled, E2NN_MODELS, Fe, xscale_obj, yscale_obj)
    #     return(t_dists.sig)

    # obj_fn = lambda x : -get_uncertainty(x, E2NN_MODELS, Fe, xscale_obj, yscale_obj)



    # BEGIN ACQUISITION OPTIMIZATION
    
    # method originally intended to be parallel
    x_new, f_opt, X_EI_opts, Y_EI_opts, *__ = global_optimization(obj_fn, lb, ub, Ndim, global_parallel_acquisitions, n_scatter_init=100_000, n_scatter_check=1000, n_local_opts=10, previous_local_xopt=np.array([]), n_scatter_gauss=n_scatter_gauss) #128)
    # x_opt, y_opt, x_opts, y_opts, exclusion_points, X_scatter, Y_scatter, local_optima_x_init, local_optima_y_init
    print("Acquisition optimization complete")

    # END ACQUISITION OPTIMIZATION

    EI_opt = -f_opt

    # EIs.append(float(EI_opt)) # sometimes EI_opt is a numpy array, sometimes it is a float
    # NRMSEs.append(e2nn_test_nrmse)
    # SumLogLikelihoods.append(e2nn_test_sum_log_likelihood[0])

    if len(x_true_opt):
        EIs_at_true_opt = np.vstack([EIs_at_true_opt, float(-obj_fn(x_true_opt))])
        __, t_dist_opt, *__ = predict_ensemble(x_true_opt, E2NN_MODELS, Fe, xscale_obj, yscale_obj)
    else:
        EIs_at_true_opt = []
        t_dist_opt = []
    

    EIs = np.vstack([EIs, float(EI_opt)]) # sometimes EI_opt is a numpy array, sometimes it is a float

    if TEST_HF_FUNCTION:
        Y_e2nn_test, t_dists, *__ = predict_ensemble(X_test_unscaled, E2NN_MODELS, Fe, xscale_obj, yscale_obj)
        e2nn_test_nrmse = NRMSE(Y_e2nn_test, Y_test_unscaled)
        e2nn_test_sum_log_likelihood = sum_log_likelihood_tdist(Y_test_unscaled, t_dists)

        NRMSEs = np.vstack([NRMSEs, e2nn_test_nrmse])
        SumLogLikelihoods = np.vstack([SumLogLikelihoods, float(e2nn_test_sum_log_likelihood[0])])

    ################################################################################
    """ PLOTTING ACQUISITION """
    print("Plotting acquisition")
    if Ndim == 1 or Ndim == 2:
        plot_1d_or_2d_acquisition(obj_fn, X_train_unscaled, X_test_unscaled, x_new, EI_opt, X_EI_opts, azim_2d, elev_2d)
    print("End plotting acquisition")

    ################################################################################
    """ ADD NEW SAMPLE UNLESS CONVERGED"""

    data_range = np.max(Y_train_unscaled)-np.min(Y_train_unscaled)
    if EI_opt < EI_TOL*data_range:
        converged_steps += 1
    else:
        converged_steps = 0

    if converged_steps >= STEPS_TO_CONVERGE or Nsamp >= MAX_SAMP:
        adaptive_sampling_converged = True
    else:

        if Ndim != 1:
            x_new = x_new.reshape(1, -1) # make x_new 2D

        y_new = F(x_new)
        X_train_unscaled = np.vstack([X_train_unscaled, x_new])
        Y_train_unscaled = np.vstack([Y_train_unscaled, y_new])
        Nsamp += 1

        min_idx = np.argmin(Y_train_unscaled)
        # Yopts.append(np.min(Y_train_unscaled))
        # Xopts.append(X_train_unscaled[min_idx,:])
        Yopts = np.hstack([Yopts, np.min(Y_train_unscaled)])
        Xopts = np.vstack([Xopts, X_train_unscaled[min_idx,:]])

    doc1 = open(SAVE_PATH+"sample data.txt","w")
    doc1.write("X_train_unscaled: "+str(X_train_unscaled)+"\n")
    doc1.write("Y_train_unscaled: "+str(Y_train_unscaled)+"\n")
    doc1.write("EIs: "+str(EIs)+"\n")
    doc1.write("Yopts: "+str(Yopts)+"\n")
    doc1.write("NRMSEs: "+str(NRMSEs)+"\n")
    doc1.write("SumLogLikelihoods: "+str(SumLogLikelihoods)+"\n")
    doc1.close()


    ################################################################################
    """ PLOT CONVERGENCE HISTORY """

    print("Plotting convergence history")
    plot_convergence_history(Xopts, Yopts, X_train_unscaled, Y_train_unscaled, EIs, NRMSEs,  x_true_opt, y_true_opt, t_dist_opt, EIs_at_true_opt)
    print("End plotting convergence history")

    ################################################################################
    """ SAVE CURRENT PROGRAM STATE """

    filename = "state_to_load.pkl"
    
    current_state = {"X_train_unscaled":X_train_unscaled,
                     "Y_train_unscaled":Y_train_unscaled,
                     "X_test_unscaled":X_test_unscaled,
                     "Y_test_unscaled":Y_test_unscaled,
                     "global_parallel_acquisitions":global_parallel_acquisitions,
                     "EIs":EIs,
                     "EIs_at_true_opt":EIs_at_true_opt,
                     "NRMSEs":NRMSEs,
                     "SumLogLikelihoods":SumLogLikelihoods,
                     "min_idx":min_idx,
                     "Yopts":Yopts,
                     "Xopts":Xopts,
                     "fourier_factor_1L":fourier_factor_1L,
                     "fourier_factor_2L":fourier_factor_2L,
                     }

    with open(filename, 'wb') as outfile:
        pickle.dump(current_state, outfile)


print("Adaptive Sampling Complete")




#
