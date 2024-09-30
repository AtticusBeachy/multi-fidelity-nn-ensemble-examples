import numpy as np
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


def plot_1d_or_2d_acquisition(obj_fn, X_train_unscaled, X_test_unscaled, 
                              x_new, EI_opt, X_EI_opts, azim_2d, elev_2d):
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
        ax.scatter(X_EI_opts.flatten(), EI_local_opts, c="g", marker="x", 
                   label="opts final")

        ax.scatter(X_train_unscaled, np.zeros(np.shape(X_train_unscaled)), 
                   c="gray", marker="o", label="data locations")
        ax.set_title(f"Expected Improvement ({Nsamp} samples)")
        ax.set_xlabel("x")
        ax.set_ylabel("EI")
        ax.legend(loc="upper left")
        fig_name = f"Expected Improvement ({Nsamp} samples).png"
        plt.savefig(SAVE_PATH+fig_name, format="png", dpi=300, 
                    bbox_inches="tight")
        plt.close(fig)

    # 2d plots
    if Ndim == 2:

        def fix_legend_crash(surf):
            # Fixes a bug where the legend tries to call the surface color
            # and crashes
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

        surf = ax.plot_trisurf(
            X1_test_unscaled, X2_test_unscaled, EI_plot.flatten(),
            triangles=tri.triangles, color="k", alpha = 0.5, linewidth=0.2, 
            label = "EI"
        )
        surf = fix_legend_crash(surf)

        ax.scatter(x_new[0], x_new[1], EI_opt, c="r", marker="o", 
                   label="optimum")

        ax.scatter(X_EI_opts[:,0], X_EI_opts[:,1], EI_local_opts, c="g", 
                   marker="x", label="opts final")

        ax.scatter(X_train_unscaled[:,0], X_train_unscaled[:,1], 
                   np.zeros(np.shape(X_train_unscaled[:,0])), c="gray",
                   marker="o", label="data locations")

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
        plt.savefig(SAVE_PATH+fig_name, format="png", dpi=300, 
                    bbox_inches="tight")
        plt.close(fig)

    return(None)

