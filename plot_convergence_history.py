import numpy as np
from scipy.stats import t
from matplotlib import pyplot as plt

# Improve figure appearence
import matplotlib as mpl
mpl.rc("axes", labelsize=14)
mpl.rc("xtick", labelsize=12)
mpl.rc("ytick", labelsize=12)

# enable saving figures
import os
SAVE_PATH = "./images/" # "." represents current path
os.makedirs(SAVE_PATH, exist_ok=True)


def plot_convergence_history(Xopts, Yopts, X_train_unscaled, Y_train_unscaled, 
                             EIs, NRMSEs, sum_log_likelihoods, 
                             samples_each_iter, x_true_opt, y_true_opt, 
                             t_dist_opt, EIs_at_true_opt):
    """  """
    Nsamp, Ndim = X_train_unscaled.shape

    # prevent shape mismatch plotting errors
    Y_train_unscaled = Y_train_unscaled.flatten()


    Niters = Yopts.size
    iters = np.array(range(1, Niters+1)) #list(range(1, Niters+1)) #

    # # Optimal xs
    # fig = plt.figure(figsize=(6.4, 4.8))
    # ax = fig.add_subplot(111)
    # for ii in range(Ndim):
    #     ax.plot(iters, Xopts[:,ii], "-", label = "Opt $X_"+str(ii)+"$", linewidth=2)
    # ax.set_title("X history")
    # ax.set_xlabel("iteration")
    # ax.set_ylabel("Optimums")
    # ax.legend(loc="upper left")
    # plt.savefig(SAVE_PATH+"History X.png", format="png", dpi=300, bbox_inches="tight")
    # # plt.draw()
    # plt.close(fig)


    # # Optimal ys
    # fig = plt.figure(figsize=(6.4, 4.8))
    # ax = fig.add_subplot(111)
    # ax.plot(iters, Yopts, "k.-", label = "Yopt", linewidth=2)
    # ax.set_title("Y history")
    # ax.set_xlabel("iteration")
    # ax.set_ylabel("Optimums")
    # ax.legend(loc="upper right")
    # plt.savefig(SAVE_PATH+"History Y.png", format="png", dpi=300, bbox_inches="tight")
    # # plt.draw()
    # plt.close(fig)


    # # Optimal ys log
    # if np.all(Yopts>0):
    #     fig = plt.figure(figsize=(6.4, 4.8))
    #     ax = fig.add_subplot(111)
    #     ax.plot(iters, Yopts, "k.-", label = "Yopt", linewidth=2)
    #     ax.set_title("Y history")
    #     ax.set_xlabel("iteration")
    #     ax.set_ylabel("Optimums")
    #     ax.legend(loc="upper right")
    #     plt.yscale("log")
    #     plt.savefig(SAVE_PATH+"History Y log.png", format="png", dpi=300, bbox_inches="tight")
    #     # plt.draw()
    #     plt.close(fig)

    # if len(x_true_opt):
    #     fig = plt.figure(figsize=(6.4, 4.8))
    #     ax = fig.add_subplot(111)
    #     ax.plot(iters, Yopts-y_true_opt, "k.-", label = "Yopt", linewidth=2)
    #     ax.set_title("Y history")
    #     ax.set_xlabel("iteration")
    #     ax.set_ylabel("inferiority to optimum")
    #     ax.legend(loc="upper right")
    #     plt.yscale("log")
    #     plt.savefig(SAVE_PATH+"History Y log loss.png", format="png", dpi=300, bbox_inches="tight")
    #     # plt.draw()
    #     plt.close(fig)


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

    # # EI log
    # fig = plt.figure(figsize=(6.4, 4.8))
    # ax = fig.add_subplot(111)
    # ax.plot(iters_eval, EIs, "r.-", label = "EI", linewidth=2)
    # ax.set_title("EI history (or other acquisition)")
    # ax.set_xlabel("iteration")
    # ax.set_ylabel("EI")
    # ax.legend(loc="upper right")
    # plt.yscale("log")
    # plt.savefig(SAVE_PATH+"History EI log.png", format="png", dpi=300, bbox_inches="tight")
    # # plt.draw()
    # plt.close(fig)

    # EI log vs samples
    fig = plt.figure(figsize=(6.4, 4.8))
    ax = fig.add_subplot(111)
    ax.plot(samples_each_iter, EIs, "r.-", label = "EI", linewidth=2)
    # ax.set_title("EI history (or other acquisition)")
    ax.set_xlabel("training samples")
    ax.set_ylabel("acquisition")
    ax.legend(loc="upper right")
    plt.yscale("log")
    plt.savefig(SAVE_PATH+"History acquisition log vs samples.png", format="png", dpi=300, bbox_inches="tight")
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
    
    # # NRMSE log
    # if len(NRMSEs):
    #     fig = plt.figure(figsize=(6.4, 4.8))
    #     ax = fig.add_subplot(111)
    #     ax.plot(iters_eval, NRMSEs, "g.-", label = "NRMSE", linewidth=2)
    #     ax.set_title("NRMSE history")
    #     ax.set_xlabel("iteration")
    #     ax.set_ylabel("NRMSE")
    #     ax.legend(loc="upper right")
    #     plt.yscale("log")
    #     plt.savefig(SAVE_PATH+"History NRMSE log.png", format="png", dpi=300, bbox_inches="tight")
    #     # plt.draw()
    #     plt.close(fig)

    # NRMSE log vs samples
    if len(NRMSEs):
        fig = plt.figure(figsize=(6.4, 4.8))
        ax = fig.add_subplot(111)
        ax.plot(samples_each_iter, NRMSEs, "g.-", label = "NRMSE", linewidth=2)
        # ax.set_title("NRMSE history")
        ax.set_xlabel("samples")
        ax.set_ylabel("NRMSE")
        ax.legend(loc="upper right")
        plt.yscale("log")
        plt.savefig(SAVE_PATH+"History NRMSE log vs samples.png", format="png", dpi=300, bbox_inches="tight")
        # plt.draw()
        plt.close(fig)

    # # Sum-log-likelihood
    # if len(sum_log_likelihoods):
    #     fig = plt.figure(figsize=(6.4, 4.8))
    #     ax = fig.add_subplot(111)
    #     ax.plot(iters_eval, sum_log_likelihoods, "g.-", label = "sum-log-likelihoods", linewidth=2)
    #     ax.set_title("sum-log-likelihood history")
    #     ax.set_xlabel("iteration")
    #     ax.set_ylabel("sum-log-likelihoods")
    #     # ax.legend(loc="upper right")
    #     plt.savefig(SAVE_PATH+"History sum-log-likelihood.png", format="png", dpi=300, bbox_inches="tight")
    #     plt.close(fig)

    # # Sum-log-likelihood vs samples
    # if len(sum_log_likelihoods):
    #     fig = plt.figure(figsize=(6.4, 4.8))
    #     ax = fig.add_subplot(111)
    #     ax.plot(samples_each_iter, sum_log_likelihoods, "g.-", label="sum-log-likelihoods", linewidth=2)
    #     ax.set_title("sum-log-likelihood history")
    #     ax.set_xlabel("samples")
    #     ax.set_ylabel("sum-log-likelihoods")
    #     # ax.legend(loc="upper right")
    #     plt.savefig(SAVE_PATH+"History sum-log-likelihood vs samples.png", format="png", dpi=300, bbox_inches="tight")
    #     plt.close(fig)


    # # Added xs
    # fig = plt.figure(figsize=(6.4, 4.8))
    # ax = fig.add_subplot(111)
    # for ii in range(Ndim):
    #     ax.plot(X_train_unscaled[:,ii], "-", label = f"Adaptive $X_{ii}$", linewidth=1)
    # ax.set_title("X adaptive (includes initial sampling)")
    # ax.set_xlabel("iteration")
    # ax.set_ylabel("Added point")
    # ax.legend(loc="upper left")
    # plt.savefig(SAVE_PATH+"History adaptive X.png", format="png", dpi=300, bbox_inches="tight")
    # # plt.draw()
    # plt.close(fig)

    # # Added ys
    # fig = plt.figure(figsize=(6.4, 4.8))
    # ax = fig.add_subplot(111)
    # ax.plot(Y_train_unscaled, "k.-", label = "Yopt", linewidth=2)
    # ax.set_title("Y adaptive (includes initial sampling)")
    # ax.set_xlabel("iteration")
    # ax.set_ylabel("Added point")
    # ax.legend(loc="upper right")
    # plt.savefig(SAVE_PATH+"History adaptive Y.png", format="png", dpi=300, bbox_inches="tight")
    # # plt.draw()
    # plt.close(fig)

    # # Added ys log scale
    # if np.all(Y_train_unscaled>0):
    #     fig = plt.figure(figsize=(6.4, 4.8))
    #     ax = fig.add_subplot(111)
    #     ax.plot(Y_train_unscaled, "k.-", label = "Yopt", linewidth=2)
    #     ax.set_title("Y adaptive (includes initial sampling)")
    #     ax.set_xlabel("iteration")
    #     ax.set_ylabel("Added point")
    #     ax.legend(loc="upper right")
    #     plt.yscale("log")
    #     plt.savefig(SAVE_PATH+"History adaptive Y logscale.png", format="png", dpi=300, bbox_inches="tight")
    #     # plt.draw()
    #     plt.close(fig)
    # if len(x_true_opt):
    #     fig = plt.figure(figsize=(6.4, 4.8))
    #     ax = fig.add_subplot(111)
    #     ax.plot(Y_train_unscaled-y_true_opt, "k.-", label = "Yopt", linewidth=2)
    #     ax.set_title("Y adaptive inferiority (includes initial sampling)")
    #     ax.set_xlabel("iteration")
    #     ax.set_ylabel("Added point")
    #     ax.legend(loc="upper right")
    #     plt.yscale("log")
    #     plt.savefig(SAVE_PATH+"History adaptive Y log loss.png", format="png", dpi=300, bbox_inches="tight")
    #     # plt.draw()
    #     plt.close(fig)

    # # prediction at known optimum
    # if len(x_true_opt):
    #     mu_opt = t_dist_opt.mu.flatten()
    #     sig_opt = t_dist_opt.sig.flatten()
    #     df_opt = t_dist_opt.df

    #     t_dist_plot = np.linspace(mu_opt-3*sig_opt, mu_opt+3*sig_opt, 200)
    #     t_dist_pdf = t.pdf(t_dist_plot, df_opt, loc=mu_opt, scale=sig_opt)

    #     fig = plt.figure(figsize=(6.4, 4.8))
    #     ax = fig.add_subplot(111)
    #     ax.plot(t_dist_pdf, t_dist_plot, "b-", linewidth=2) #, label="pred dist") #
    #     ax.scatter(0.0, mu_opt, c="b", marker="o", label="pred")
    #     ax.scatter(0.0, y_true_opt, c="g", marker="o", label="true")

    #     ax.set_xlabel("pdf")
    #     ax.set_ylabel("optimum")
    #     ax.legend(loc="upper right")
    #     fig_name = f"Accuracy at optimum ({Nsamp} samples).png"
    #     plt.savefig(SAVE_PATH+fig_name, format="png", dpi=300, bbox_inches="tight")
    #     plt.close(fig)

    # if len(x_true_opt):
    #     # EI log
    #     fig = plt.figure(figsize=(6.4, 4.8))
    #     ax = fig.add_subplot(111)
    #     ax.plot(iters_eval, EIs, "r.-", label = "maximum EI", linewidth=2)
    #     ax.plot(iters_eval, EIs_at_true_opt, "g.-", label = "EI at true opt", linewidth=2)
    #     ax.set_title("EI history (or other acquisition)")
    #     ax.set_xlabel("iteration")
    #     ax.set_ylabel("EI")
    #     ax.legend(loc="upper right")
    #     plt.yscale("log")
    #     plt.savefig(SAVE_PATH+"History EI maximized and at optimum", format="png", dpi=300, bbox_inches="tight")
    #     plt.close(fig)

    return(None)
