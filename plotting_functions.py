
import numpy as np
from scipy.stats import t
from ensemble_predict import predict_e2nn, predict_ensemble
from accuracy_metrics import NRMSE

################################################################################
"""HANDLING FIGURES"""

# enable plotting
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.tri as mtri

# Improve figure appearence
import matplotlib as mpl
mpl.rc("axes", labelsize=14)
mpl.rc("xtick", labelsize=12)
mpl.rc("ytick", labelsize=12)

# enable saving figures
import os
SAVE_PATH = "./images/" # "." represents current path
os.makedirs(SAVE_PATH, exist_ok=True)

def string(val, dig=5):
    """ For rounding error measures when included in plot titles """
    # val = np.round(val, dig)
    # s = str(val)
    if val < 0.01:
        s = np.format_float_scientific(val, precision=dig)
    else:
        s = np.format_float_positional(val, precision=dig)
    return(s)










def plot_ensemble_details(E2NN_MODELS, ALL_E2NN_MODELS, bad_idxs, Fe, lb, ub, xscale_obj, yscale_obj, X_train_unscaled, Y_train_unscaled, X_test_unscaled, Y_test_unscaled, fourier_factor_1L, fourier_factor_2L, azim_2d, elev_2d):
    """  """

    NUM_MODELS = len(E2NN_MODELS)
    NUM_MODELS_TOTAL = len(ALL_E2NN_MODELS)

    ################################################################################
    """ GET VALUES FOR PLOTTING """
    
    Nsamp, Ndim = X_train_unscaled.shape

    X_plot = X_test_unscaled #np.linspace(lb, ub, 500)

    y_hf_plot = Y_test_unscaled
    Y_LF_PLOTS = [lf(X_plot) for lf in Fe]

    Y_E2NN_PLOTS = []
    for model in E2NN_MODELS:
        y_e2nn_plot = predict_e2nn(X_plot, model, Fe, xscale_obj, yscale_obj)
        Y_E2NN_PLOTS.append(y_e2nn_plot)

    Y_E2NN_PLOTS_ALL = []
    for model in ALL_E2NN_MODELS:
        y_e2nn_plot = predict_e2nn(X_plot, model, Fe, xscale_obj, yscale_obj)
        Y_E2NN_PLOTS_ALL.append(y_e2nn_plot)

    [Y_ensemble, t_dists, x_95_lower, x_95_upper] = predict_ensemble(X_plot, E2NN_MODELS, Fe, xscale_obj, yscale_obj)

    ################################################################################
    """ SUBPLOT DIAGONAL SWEEPS FOR EACH MODEL IN THE ENSEMBLE (ANY DIMENSIONALITY) """

    x_diag_sweep = np.linspace(lb, ub, 1000*int(np.ceil(np.sqrt(Ndim)))) 
    x_plot_1d = np.linspace(-1, 1, 1000*int(np.ceil(np.sqrt(Ndim))))

    fig = plt.figure(figsize=(4*8, 4*NUM_MODELS_TOTAL//8))
    for ii in range(NUM_MODELS_TOTAL):
        plt.subplot(NUM_MODELS_TOTAL//8, 8, ii+1)

        y_diag_sweep = predict_e2nn(x_diag_sweep, ALL_E2NN_MODELS[ii], Fe, xscale_obj, yscale_obj)
        
        if ii in bad_idxs: #bad_idxs_tf[ii]: #
            plt.plot(x_plot_1d, y_diag_sweep, "r-", linewidth=1, alpha=1.0)
        else: 
            plt.plot(x_plot_1d, y_diag_sweep, "g-", linewidth=1, alpha=1.0)
    plt.title(f"(Fourier factor 1L {fourier_factor_1L}) (Fourier factor 2L {fourier_factor_2L})")
    plt.tight_layout()
    fig_name = f"Diagonal Sweeps for each model ({Nsamp} samples) (Fourier factor 1L {fourier_factor_1L}) (Fourier factor 2L {fourier_factor_2L}).png"
    plt.savefig(SAVE_PATH+fig_name, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    ################################################################################
    """ LARGE DIAGONAL SWEEP OF ENSEMBLE TO CHECK STABILITY (ANY DIMENSIONALITY) """
    x_diag_sweep = np.linspace(lb, ub, 1000*int(np.ceil(np.sqrt(Ndim))))
    y_diag_sweep, __, sweep_95_lower, sweep_95_upper = predict_ensemble(x_diag_sweep, E2NN_MODELS, Fe, xscale_obj, yscale_obj)
    x_plot_1d = np.linspace(0, 1, 1000*int(np.ceil(np.sqrt(Ndim))))

    fig = plt.figure(figsize=(3*6.4, 3*4.8))
    ax = fig.add_subplot(111)
    ax.plot(x_plot_1d, y_diag_sweep, "b-", linewidth=2)
    ax.plot(x_plot_1d, sweep_95_lower, "b-", linewidth=2, alpha=0.5)
    ax.plot(x_plot_1d, sweep_95_upper, "b-", linewidth=2, alpha=0.5)
    ax.set_title("Diagonal Sweep to Check Stability")
    plt.tight_layout()
    fig_name = f"Stability Check ({Nsamp} samples).png"
    plt.savefig(SAVE_PATH+fig_name, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    ################################################################################
    """ ACTUAL VS PREDICTED (TEST DATA) """

    if Y_test_unscaled is not None: #if TEST_HF_FUNCTION: # 

        e2nn_test_nrmse = NRMSE(Y_ensemble, Y_test_unscaled)
        yerr=x_95_upper-Y_ensemble
        fig = plt.figure(figsize=(6.4, 4.8))
        ax = fig.add_subplot(111)
        ax.errorbar(Y_test_unscaled.flatten(), Y_ensemble.flatten(), yerr=yerr.flatten(), fmt = "none", color = "g", marker=None, capsize = 4, elinewidth=0.5, alpha = 0.2, label="95% error bars", zorder=1) #capthick= , elinewidth=
        ax.scatter(Y_test_unscaled, Y_ensemble, c="b", marker=".", label="test predictions", zorder=2)
        ax.plot(Y_test_unscaled, Y_test_unscaled, "k-", linewidth=1, label = "true", zorder=3)
        ax.set_title(f"E2NN fit (Test NRMSE={e2nn_test_nrmse})")
        ax.set_xlabel("Actual", fontsize=18)
        ax.set_ylabel("Predicted", fontsize=18)
        ax.legend(loc="lower right")
        plt.tight_layout()
        fig_name = f"Actual vs Predicted Ensemble ({Nsamp} samples).png"
        plt.savefig(SAVE_PATH+fig_name, format="png", dpi=300, bbox_inches="tight")
        # plt.draw()
        plt.close(fig)

    ################################################################################
    """ ACTUAL VS PREDICTED (TRAINING DATA) """

    # TRAINING DATA ACTUAL VS PREDICTED
    [Y_train_predicted, *__, x_95_upper_train] = predict_ensemble(X_train_unscaled, E2NN_MODELS, Fe, xscale_obj, yscale_obj)
    yerr_train = x_95_upper_train - Y_train_predicted 
    e2nn_train_nrmse = NRMSE(Y_train_predicted, Y_train_unscaled)

    fig = plt.figure(figsize=(6.4, 4.8))
    ax = fig.add_subplot(111)
    ax.errorbar(Y_train_unscaled.flatten(), Y_train_predicted.flatten(), yerr=yerr_train.flatten(), fmt = "none", color = "g", marker=None, capsize = 4, elinewidth=0.5, alpha = 0.2, label="95% error bars", zorder=1) #capthick= , elinewidth=
    ax.scatter(Y_train_unscaled, Y_train_predicted, c="b", marker=".", label="test predictions", zorder=2)
    ax.plot(Y_train_unscaled, Y_train_unscaled, "k-", linewidth=1, label = "true", zorder=3)
    ax.set_title(f"E2NN ensemble fit (Train NRMSE={e2nn_train_nrmse})")
    ax.set_xlabel("Actual", fontsize=18)
    ax.set_ylabel("Predicted", fontsize=18)
    ax.legend(loc="lower right")
    plt.tight_layout()
    fig_name = f"Actual vs Predicted Training Data ({Nsamp} samples).png"
    plt.savefig(SAVE_PATH+fig_name, format="png", dpi=300, bbox_inches="tight")
    # plt.draw()
    plt.close(fig)


    if Ndim == 1:

        ################################################################################
        """ 1D HF DATA AND E2NN MODELS (GOOD AND BAD) """

        # PLOT SURROGATES IN ENSEMBLE (NO OTHER INFORMATION EXCEPT HF DATA)

        fig = plt.figure(figsize=(6.4, 4.8))
        ax = fig.add_subplot(111)

        ax.scatter(X_train_unscaled, Y_train_unscaled, c="k", marker="o", label="data")
        
        good_label = "E2NN"
        bad_label = "bad E2NN"
        for ii in range(NUM_MODELS_TOTAL):
            if ii in bad_idxs:
                ax.plot(X_plot, Y_E2NN_PLOTS_ALL[ii], "r-", label = bad_label, linewidth=1, alpha=0.2)
                bad_label = None
            else:
                ax.plot(X_plot, Y_E2NN_PLOTS_ALL[ii], "g-", label = good_label, linewidth=1, alpha=0.2) 
                good_label = None

        ax.set_title("Problem Visualization")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.legend(loc="upper left")
        plt.tight_layout()
        fig_name = f"Scatter visualization for paper ({Nsamp} samples).png"
        plt.savefig(SAVE_PATH+fig_name, format="png", dpi=300, bbox_inches="tight")
        # plt.draw()
        plt.close(fig)

        ################################################################################
        """ SUBPLOT 1D SWEEPS FOR EACH MODEL IN THE ENSEMBLE (PLUS HF DATA) """

        fig = plt.figure(figsize=(2*8, 2*NUM_MODELS_TOTAL//8))
        for ii in range(NUM_MODELS_TOTAL):
            plt.subplot(NUM_MODELS_TOTAL//8, 8, ii+1)
            plt.scatter(X_train_unscaled, Y_train_unscaled, c="k", marker="o", label="data")

            if ii in bad_idxs: #bad_idxs_tf[ii]: #
                plt.plot(X_plot, Y_E2NN_PLOTS_ALL[ii], "r-", linewidth=2, alpha=1.0)
            else: 
                plt.plot(X_plot, Y_E2NN_PLOTS_ALL[ii], "g-", linewidth=2, alpha=1.0)
        plt.tight_layout()
        fig_name = f"Subplot visualizations for each model ({Nsamp} samples).png"
        plt.savefig(SAVE_PATH+fig_name, format="png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        
        ################################################################################
        """ 1D PROBLEM STATEMENT (HF, LF AND DATA) """

        fig = plt.figure(figsize=(6.4, 4.8))
        ax = fig.add_subplot(111)
        if y_hf_plot is not None:
            ax.plot(X_plot, y_hf_plot, "k-", linewidth=2, label="HF")

        label = "LF"
        for y_lf_plot in Y_LF_PLOTS:
            ax.plot(X_plot, y_lf_plot, "k--", linewidth=1, label=label)
            label = None

        ax.scatter(X_train_unscaled, Y_train_unscaled, c="k", marker="o", label="data")

        ax.set_title("Problem Visualization")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.legend(loc="upper left")
        plt.tight_layout()
        fig_name = f"Problem Visualization ({Nsamp} samples).png"
        plt.savefig(SAVE_PATH+fig_name, format="png", dpi=300, bbox_inches="tight")
        # plt.draw()
        plt.close(fig)
        
        ################################################################################
        """ 1D SOLUTION (HF, DATA, ENSEMBLE PREDICTION LINES) """
        fig = plt.figure(figsize=(6.4, 4.8))
        ax = fig.add_subplot(111)
        if y_hf_plot is not None:
            ax.plot(X_plot, y_hf_plot, "k-", label = "HF", linewidth=2)
        ax.scatter(X_train_unscaled, Y_train_unscaled, c="k", marker="o", label="data")

        ax.plot(X_plot, Y_ensemble, "b-", label = "E2NN Ensemble", linewidth=2)
        ax.plot(X_plot, x_95_lower, "b--", label = "95% probability dist.", linewidth=2)
        ax.plot(X_plot, x_95_upper, "b--", linewidth=2)

        ax.set_title(f"E2NN prediction ({Nsamp} samples) (Fourier factor 1L {fourier_factor_1L}) (Fourier factor 2L {fourier_factor_2L})")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.legend(loc="upper left")
        plt.tight_layout()
        fig_name = f"E2NN prediction ({Nsamp} samples) (Fourier factor 1L {fourier_factor_1L}) (Fourier factor 2L {fourier_factor_2L}).png"
        plt.savefig(SAVE_PATH+fig_name, format="png", dpi=300, bbox_inches="tight")
        # plt.draw()
        plt.close(fig)


        ################################################################################
        """ 1D SOLUTION (HF, DATA, GOOD & BAD E2NN, ENSEMBLE PREDICTION LINES) """

        fig = plt.figure(figsize=(6.4, 4.8))
        ax = fig.add_subplot(111)
        if y_hf_plot is not None:
            ax.plot(X_plot, y_hf_plot, "k-", label = "HF", linewidth=2)

        ax.scatter(X_train_unscaled, Y_train_unscaled, c="k", marker="o", label="data")

        good_label = "E2NN"
        bad_label = "bad E2NN"
        for ii in range(NUM_MODELS_TOTAL):
            if ii in bad_idxs:
                ax.plot(X_plot, Y_E2NN_PLOTS_ALL[ii], "r-", label = bad_label, linewidth=1, alpha=0.2)
                bad_label = None
            else:
                ax.plot(X_plot, Y_E2NN_PLOTS_ALL[ii], "g-", label = good_label, linewidth=1, alpha=0.2) 
                good_label = None

        ax.plot(X_plot, Y_ensemble, "b-", label = "Ensemble", linewidth=2) #"E2NN Ensemble"
        ax.plot(X_plot, x_95_lower, "b--", label = "95% probability dist.", linewidth=2)
        ax.plot(X_plot, x_95_upper, "b--", linewidth=2)

        ax.set_title("Problem Visualization")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.legend(loc="upper left")
        plt.tight_layout()
        fig_name = f"Problem with individual NNs ({Nsamp} samples).png"
        plt.savefig(SAVE_PATH+fig_name, format="png", dpi=300, bbox_inches="tight")
        # plt.draw()
        plt.close(fig)

    elif Ndim == 2:

        def fix_legend_crash(surf):
            # Fixes a bug where the legend tries to call the surface color and crashes
            surf._facecolors2d=surf._facecolor3d
            surf._edgecolors2d=surf._edgecolor3d
            return(surf)

        X1_test_unscaled = X_test_unscaled[:,0]#.flatten()
        X2_test_unscaled = X_test_unscaled[:,1]#.flatten()
        tri = mtri.Triangulation(X1_test_unscaled, X2_test_unscaled)

        ################################################################################
        """ 2D PROBLEM STATEMENT (HF, LF AND DATA) """
        # PLOT PROBLEM STATEMENT (NO SURROGATE)

        fig = plt.figure(figsize=(6.4, 4.8))
        ax = fig.add_subplot(111, projection="3d")
        ax.grid(False)

        if y_hf_plot is not None:
            surf = ax.plot_trisurf(X1_test_unscaled, X2_test_unscaled, y_hf_plot, \
                            triangles=tri.triangles, color="k", alpha=0.5, linewidth=0.2, label="HF")
            surf = fix_legend_crash(surf)

        label = "LF"
        for y_lf_plot in Y_LF_PLOTS:
            surf = ax.plot_trisurf(X1_test_unscaled, X2_test_unscaled, y_lf_plot, \
                                triangles=tri.triangles, color="orange", alpha=0.5, linewidth=0.2, label=label) #gray #orange
            surf = fix_legend_crash(surf)
            label = None

        ax.scatter(X_train_unscaled[:,0], X_train_unscaled[:,1], Y_train_unscaled, c="r", marker="o", label="data")

        ax.set_title("Problem Visualization")
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        ax.set_zlabel("y")
        ax.legend(loc="upper right")
        ax.azim = azim_2d
        ax.elev = elev_2d

        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor("w")
        ax.yaxis.pane.set_edgecolor("w")
        ax.zaxis.pane.set_edgecolor("w")

        plt.tight_layout()
        fig_name = f"Problem Visualization ({Nsamp} samples).png"
        plt.savefig(SAVE_PATH+fig_name, format="png", dpi=300, bbox_inches="tight")
        # plt.draw()
        plt.close(fig)


        ################################################################################
        """ 2D SOLUTION (HF, DATA, ENSEMBLE SURFACES) """

        X1_test_unscaled = X_test_unscaled[:,0]#.flatten()
        X2_test_unscaled = X_test_unscaled[:,1]#.flatten()
        tri = mtri.Triangulation(X1_test_unscaled, X2_test_unscaled)

        fig = plt.figure(figsize=(6.4, 4.8))
        ax = fig.add_subplot(111, projection="3d")
        ax.grid(False)
        # ax.set_facecolor("white") # CHECK RESULTS OF THIS # no effect

        if y_hf_plot is not None:
            surf = ax.plot_trisurf(X1_test_unscaled, X2_test_unscaled, y_hf_plot, \
                            triangles=tri.triangles, color="k", alpha = 0.3, linewidth=0.2, label = "HF")
            surf = fix_legend_crash(surf)

        ax.scatter(X_train_unscaled[:,0], X_train_unscaled[:,1], Y_train_unscaled, c="r", marker="o", label="data")

        surf = ax.plot_trisurf(X1_test_unscaled, X2_test_unscaled, Y_ensemble.flatten(), \
                        triangles=tri.triangles, color="b", alpha = 0.3, linewidth=0.2, label = "E2NN Ensemble")
        surf = fix_legend_crash(surf)

        surf = ax.plot_trisurf(X1_test_unscaled, X2_test_unscaled, x_95_lower.flatten(), \
                        triangles=tri.triangles, color="b", alpha = 0.1, linewidth=0.2, \
                        linestyle = "--", label = "95% probability dist.")
        surf = fix_legend_crash(surf)

        surf = ax.plot_trisurf(X1_test_unscaled, X2_test_unscaled, x_95_upper.flatten(), \
                        triangles=tri.triangles, color="b", alpha = 0.1, linewidth=0.2, linestyle = "--")
        surf = fix_legend_crash(surf)

        ax.set_title(f"E2NN prediction ({Nsamp} samples)")

        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        ax.set_zlabel("y")
        ax.legend(loc="upper right")
        ax.azim = azim_2d
        ax.elev = elev_2d

        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor("w")
        ax.yaxis.pane.set_edgecolor("w")
        ax.zaxis.pane.set_edgecolor("w")
        plt.tight_layout()
        fig_name = f"E2NN prediction ({Nsamp} samples).png"
        plt.savefig(SAVE_PATH+fig_name, format="png", dpi=300, bbox_inches="tight")
        # plt.draw()
        plt.close(fig)

        ################################################################################
        """ 2D SOLUTION (HF, DATA, GOOD & BAD E2NN, ENSEMBLE SURFACES) """
        fig = plt.figure(figsize=(6.4, 4.8))
        ax = fig.add_subplot(111, projection="3d")
        ax.grid(False)

        if y_hf_plot is not None:
            surf = ax.plot_trisurf(X1_test_unscaled, X2_test_unscaled, y_hf_plot, \
                            triangles=tri.triangles, color="k", alpha = 0.5, linewidth=0.2, label = "HF")
            surf = fix_legend_crash(surf)

        ax.scatter(X_train_unscaled[:,0], X_train_unscaled[:,1], Y_train_unscaled, c="r", marker="o", label="data")


        good_label = "E2NN"
        bad_label = "bad E2NN"
        for ii in range(NUM_MODELS_TOTAL):

            if ii in bad_idxs:
                # bad e2nn models
                surf = ax.plot_trisurf(X1_test_unscaled, X2_test_unscaled, Y_E2NN_PLOTS_ALL[ii], \
                                triangles=tri.triangles, color="r", alpha=0.2, linewidth=0.1, label=bad_label)
                surf = fix_legend_crash(surf)
                bad_label = None
            else:
                # good e2nn models
                surf = ax.plot_trisurf(X1_test_unscaled, X2_test_unscaled, Y_E2NN_PLOTS_ALL[ii], \
                                triangles=tri.triangles, color="g", alpha = 0.2, linewidth=0.1, label=good_label)
                surf = fix_legend_crash(surf)
                good_label = None

        surf = ax.plot_trisurf(X1_test_unscaled, X2_test_unscaled, Y_ensemble.flatten(), \
                        triangles=tri.triangles, color="b", alpha=0.5, linewidth=0.2, label = "E2NN Ensemble")
        surf = fix_legend_crash(surf)

        surf = ax.plot_trisurf(X1_test_unscaled, X2_test_unscaled, x_95_lower.flatten(), \
                        triangles=tri.triangles, color="b", alpha=0.4, linewidth=0.2, \
                        linestyle = "--", label = "95% probability dist.")
        surf = fix_legend_crash(surf)

        surf = ax.plot_trisurf(X1_test_unscaled, X2_test_unscaled, x_95_upper.flatten(), \
                        triangles=tri.triangles, color="b", alpha=0.4, linewidth=0.2, linestyle="--")
        surf = fix_legend_crash(surf)

        ax.set_title("Problem Visualization")
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        ax.set_zlabel("y")
        ax.legend(loc="upper right")
        ax.azim = azim_2d
        ax.elev = elev_2d

        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor("w")
        ax.yaxis.pane.set_edgecolor("w")
        ax.zaxis.pane.set_edgecolor("w")

        plt.tight_layout()
        fig_name = f"Problem with individual NNs ({Nsamp} samples).png"
        plt.savefig(SAVE_PATH+fig_name, format="png", dpi=300, bbox_inches="tight")
        # plt.draw()
        plt.close(fig)

    return(None)







def plot_1d_or_2d_acquisition(obj_fn, X_train_unscaled, X_test_unscaled, x_new, EI_opt, X_EI_opts, azim_2d, elev_2d):
    """  """
    
    Nsamp, Ndim = X_train_unscaled.shape
    X_plot = X_test_unscaled
    EI_plot = -obj_fn(X_plot)
    EI_local_opts = -obj_fn(X_EI_opts)

    # 1d plots
    if Ndim == 1:
        fig = plt.figure(figsize=(6.4, 4.8))
        ax = fig.add_subplot(111)
        ax.plot(X_plot, EI_plot, "k-", label = "EI", linewidth=2)

        ax.scatter(x_new, EI_opt, c="r", marker="o", label="optimum")

        # ax.scatter(X_preacq_opt.flatten(), EI_init, c="y", marker="x", label="opts init")
        ax.scatter(X_EI_opts.flatten(), EI_local_opts, c="g", marker="x", label="opts final")

        ax.scatter(X_train_unscaled, np.zeros(np.shape(X_train_unscaled)), c="gray", marker="o", label="data locations")
        ax.set_title(f"Expected Improvement ({Nsamp} samples)")
        ax.set_xlabel("x")
        ax.set_ylabel("EI")
        ax.legend(loc="upper left")
        fig_name = f"Expected Improvement ({Nsamp} samples).png"
        plt.savefig(SAVE_PATH+fig_name, format="png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    # 2d plots
    if Ndim == 2:

        def fix_legend_crash(surf):
            # Fixes a bug where the legend tries to call the surface color and crashes
            surf._facecolors2d=surf._facecolor3d
            surf._edgecolors2d=surf._edgecolor3d
            return(surf)

        X1_test_unscaled = X_test_unscaled[:,0]#.flatten()
        X2_test_unscaled = X_test_unscaled[:,1]#.flatten()
        tri = mtri.Triangulation(X1_test_unscaled, X2_test_unscaled)

        # plot acquisition
        fig = plt.figure(figsize=(6.4, 4.8))
        ax = fig.add_subplot(111, projection="3d")
        ax.grid(False)

        surf = ax.plot_trisurf(X1_test_unscaled, X2_test_unscaled, EI_plot.flatten(), \
                        triangles=tri.triangles, color="k", alpha = 0.5, linewidth=0.2, label = "EI")
        surf = fix_legend_crash(surf)

        ax.scatter(x_new[0], x_new[1], EI_opt, c="r", marker="o", label="optimum")

        ax.scatter(X_EI_opts[:,0], X_EI_opts[:,1], EI_local_opts, c="g", marker="x", label="opts final")

        ax.scatter(X_train_unscaled[:,0], X_train_unscaled[:,1], np.zeros(np.shape(X_train_unscaled[:,0])), c="gray", marker="o", label="data locations")

        ax.set_title(f"Expected Improvement ({Nsamp} samples)")
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        ax.set_zlabel("EI")
        ax.legend(loc="upper right")
        ax.azim = azim_2d
        ax.elev = elev_2d

        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor("w")
        ax.yaxis.pane.set_edgecolor("w")
        ax.zaxis.pane.set_edgecolor("w")
        fig_name = f"Expected Improvement ({Nsamp} samples).png"
        plt.savefig(SAVE_PATH+fig_name, format="png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    return(None)








def plot_convergence_history(Xopts, Yopts, X_train_unscaled, Y_train_unscaled, EIs, NRMSEs,  x_true_opt, y_true_opt, t_dist_opt, EIs_at_true_opt):
    """  """
    Nsamp, Ndim = X_train_unscaled.shape

    Niters = Yopts.size
    iters = np.array(range(1, Niters+1)) #list(range(1, Niters+1)) #

    # Optimal xs
    fig = plt.figure(figsize=(6.4, 4.8))
    ax = fig.add_subplot(111)
    for ii in range(Ndim):
        ax.plot(iters, Xopts[:,ii], "-", label = "Opt $X_"+str(ii)+"$", linewidth=2)
    ax.set_title("X history")
    ax.set_xlabel("iteration")
    ax.set_ylabel("Optimums")
    ax.legend(loc="upper left")
    plt.savefig(SAVE_PATH+"History X.png", format="png", dpi=300, bbox_inches="tight")
    # plt.draw()
    plt.close(fig)


    # Optimal ys
    fig = plt.figure(figsize=(6.4, 4.8))
    ax = fig.add_subplot(111)
    ax.plot(iters, Yopts, "k.-", label = "Yopt", linewidth=2)
    ax.set_title("Y history")
    ax.set_xlabel("iteration")
    ax.set_ylabel("Optimums")
    ax.legend(loc="upper right")
    plt.savefig(SAVE_PATH+"History Y.png", format="png", dpi=300, bbox_inches="tight")
    # plt.draw()
    plt.close(fig)


    # Optimal ys log
    if np.all(Yopts>0):
        fig = plt.figure(figsize=(6.4, 4.8))
        ax = fig.add_subplot(111)
        ax.plot(iters, Yopts, "k.-", label = "Yopt", linewidth=2)
        ax.set_title("Y history")
        ax.set_xlabel("iteration")
        ax.set_ylabel("Optimums")
        ax.legend(loc="upper right")
        plt.yscale("log")
        plt.savefig(SAVE_PATH+"History Y log.png", format="png", dpi=300, bbox_inches="tight")
        # plt.draw()
        plt.close(fig)

    if len(x_true_opt):
        fig = plt.figure(figsize=(6.4, 4.8))
        ax = fig.add_subplot(111)
        ax.plot(iters, Yopts-y_true_opt, "k.-", label = "Yopt", linewidth=2)
        ax.set_title("Y history")
        ax.set_xlabel("iteration")
        ax.set_ylabel("inferiority to optimum")
        ax.legend(loc="upper right")
        plt.yscale("log")
        plt.savefig(SAVE_PATH+"History Y log loss.png", format="png", dpi=300, bbox_inches="tight")
        # plt.draw()
        plt.close(fig)


    # print("iters: \n", iters)
    # print("Yopts: \n", Yopts)
    # print("EIs: \n", EIs)
    
    # after convergence iters_eval will be equal to iters, because terms are added to EIs and NRMSEs in the last iteration but not to Yopts or Xopts
    iters_eval = np.array(range(1, EIs.size+1))

    # EI
    fig = plt.figure(figsize=(6.4, 4.8))
    ax = fig.add_subplot(111)
    ax.plot(iters_eval, EIs, "r.-", label = "EI", linewidth=2)
    ax.set_title("EI history (or other acquisition)")
    ax.set_xlabel("iteration")
    ax.set_ylabel("EI")
    ax.legend(loc="upper right")
    plt.savefig(SAVE_PATH+"History EI.png", format="png", dpi=300, bbox_inches="tight")
    # plt.draw()
    plt.close(fig)

    # EI log
    fig = plt.figure(figsize=(6.4, 4.8))
    ax = fig.add_subplot(111)
    ax.plot(iters_eval, EIs, "r.-", label = "EI", linewidth=2)
    ax.set_title("EI history (or other acquisition)")
    ax.set_xlabel("iteration")
    ax.set_ylabel("EI")
    ax.legend(loc="upper right")
    plt.yscale("log")
    plt.savefig(SAVE_PATH+"History EI log.png", format="png", dpi=300, bbox_inches="tight")
    # plt.draw()
    plt.close(fig)

    # NRMSE
    if len(NRMSEs):
        fig = plt.figure(figsize=(6.4, 4.8))
        ax = fig.add_subplot(111)
        ax.plot(iters_eval, NRMSEs, "g.-", label = "NRMSE", linewidth=2)
        ax.set_title("NRMSE history")
        ax.set_xlabel("iteration")
        ax.set_ylabel("NRMSE")
        ax.legend(loc="upper right")
        plt.savefig(SAVE_PATH+"History NRMSE.png", format="png", dpi=300, bbox_inches="tight")
        # plt.draw()
        plt.close(fig)
    
    # NRMSE log
    if len(NRMSEs):
        fig = plt.figure(figsize=(6.4, 4.8))
        ax = fig.add_subplot(111)
        ax.plot(iters_eval, NRMSEs, "g.-", label = "NRMSE", linewidth=2)
        ax.set_title("NRMSE history")
        ax.set_xlabel("iteration")
        ax.set_ylabel("NRMSE")
        ax.legend(loc="upper right")
        plt.yscale("log")
        plt.savefig(SAVE_PATH+"History NRMSE log.png", format="png", dpi=300, bbox_inches="tight")
        # plt.draw()
        plt.close(fig)

    # Added xs
    fig = plt.figure(figsize=(6.4, 4.8))
    ax = fig.add_subplot(111)
    for ii in range(Ndim):
        ax.plot(X_train_unscaled[:,ii], "-", label = f"Adaptive $X_{ii}$", linewidth=1)
    ax.set_title("X adaptive (includes initial sampling)")
    ax.set_xlabel("iteration")
    ax.set_ylabel("Added point")
    ax.legend(loc="upper left")
    plt.savefig(SAVE_PATH+"History adaptive X.png", format="png", dpi=300, bbox_inches="tight")
    # plt.draw()
    plt.close(fig)

    # Added ys
    fig = plt.figure(figsize=(6.4, 4.8))
    ax = fig.add_subplot(111)
    ax.plot(Y_train_unscaled, "k.-", label = "Yopt", linewidth=2)
    ax.set_title("Y adaptive (includes initial sampling)")
    ax.set_xlabel("iteration")
    ax.set_ylabel("Added point")
    ax.legend(loc="upper right")
    plt.savefig(SAVE_PATH+"History adaptive Y.png", format="png", dpi=300, bbox_inches="tight")
    # plt.draw()
    plt.close(fig)

    # Added ys log scale
    if np.all(Y_train_unscaled>0):
        fig = plt.figure(figsize=(6.4, 4.8))
        ax = fig.add_subplot(111)
        ax.plot(Y_train_unscaled, "k.-", label = "Yopt", linewidth=2)
        ax.set_title("Y adaptive (includes initial sampling)")
        ax.set_xlabel("iteration")
        ax.set_ylabel("Added point")
        ax.legend(loc="upper right")
        plt.yscale("log")
        plt.savefig(SAVE_PATH+"History adaptive Y logscale.png", format="png", dpi=300, bbox_inches="tight")
        # plt.draw()
        plt.close(fig)
    if len(x_true_opt):
        fig = plt.figure(figsize=(6.4, 4.8))
        ax = fig.add_subplot(111)
        ax.plot(Y_train_unscaled-y_true_opt, "k.-", label = "Yopt", linewidth=2)
        ax.set_title("Y adaptive inferiority (includes initial sampling)")
        ax.set_xlabel("iteration")
        ax.set_ylabel("Added point")
        ax.legend(loc="upper right")
        plt.yscale("log")
        plt.savefig(SAVE_PATH+"History adaptive Y log loss.png", format="png", dpi=300, bbox_inches="tight")
        # plt.draw()
        plt.close(fig)

    # prediction at known optimum
    if len(x_true_opt):
        mu_opt = t_dist_opt.mu.flatten()
        sig_opt = t_dist_opt.sig.flatten()
        df_opt = t_dist_opt.df

        t_dist_plot = np.linspace(mu_opt-3*sig_opt, mu_opt+3*sig_opt, 200)
        t_dist_pdf = t.pdf(t_dist_plot, df_opt, loc=mu_opt, scale=sig_opt)

        fig = plt.figure(figsize=(6.4, 4.8))
        ax = fig.add_subplot(111)
        ax.plot(t_dist_pdf, t_dist_plot, "b-", linewidth=2) #, label="pred dist") #
        ax.scatter(0.0, mu_opt, c="b", marker="o", label="pred")
        ax.scatter(0.0, y_true_opt, c="g", marker="o", label="true")

        ax.set_xlabel("pdf")
        ax.set_ylabel("optimum")
        ax.legend(loc="upper right")
        fig_name = f"Accuracy at optimum ({Nsamp} samples).png"
        plt.savefig(SAVE_PATH+fig_name, format="png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    if len(x_true_opt):
        # EI log
        fig = plt.figure(figsize=(6.4, 4.8))
        ax = fig.add_subplot(111)
        ax.plot(iters_eval, EIs, "r.-", label = "maximum EI", linewidth=2)
        ax.plot(iters_eval, EIs_at_true_opt, "g.-", label = "EI at true opt", linewidth=2)
        ax.set_title("EI history (or other acquisition)")
        ax.set_xlabel("iteration")
        ax.set_ylabel("EI")
        ax.legend(loc="upper right")
        plt.yscale("log")
        plt.savefig(SAVE_PATH+"History EI maximized and at optimum", format="png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    return(None)
