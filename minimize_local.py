import numpy as np
from pyDOE3 import lhs
from scipy.stats import multivariate_normal

def minimize_local(obj_fn, x0, step0, bounds, total_parallel_acquisitions, 
                   Nscatter=10_000, ftol=1e-6, xtol=1e-6):
    """
    Local optimization using Gaussian Scatter method
    ftol: if all evaluations yield the same value to within ftol, converge
          (region flat)
    xtol: if the standard deviation drops to this level, converge 
          (optimum approached)
    """
    import time
    from scipy.stats import norm

    Ndim = len(x0)
    stdev = step0
    y0 = obj_fn(x0)

    pdfs_at_x0 = np.array([])

    # bounds were scaled from [[0, 1],[0, 1], [0, 1], ... ]
    lb = bounds[:,0]
    ub = bounds[:,1]

    lb = np.tile(lb, (Nscatter, 1))
    ub = np.tile(ub, (Nscatter, 1))


    converged = False
    # global total_parallel_acquisitions

    while not converged:

        # find optimum

        # # old random
        # x_scatter = rng.normal(loc = x0, scale = stdev/np.sqrt(Ndim), size = (Nscatter, Ndim))

        # # new lhs
        # tic = time.perf_counter()
        # for tt in range(10_000):
        x_scatter = lhs(Ndim, samples=Nscatter, criterion="maximin", 
                        iterations=5)#100_000)
        x_scatter = norm.ppf(x_scatter, loc = x0, scale = stdev/np.sqrt(Ndim)) #inverse cdf
        # toc = time.perf_counter()
        # print("Average time for lhs normal 5 iter: ", (toc-tic)/10_000)
        # print("Mean: ", np.mean(x_scatter, axis=0))
        # print("Std dev: ", np.std(x_scatter, axis=0))

        low = x_scatter<lb
        high = x_scatter>ub
        x_scatter[low] = lb[low]
        x_scatter[high] = ub[high]

        # x_scatter = np.vstack([x_scatter, x0])
        y_scatter = obj_fn(x_scatter)

        idx = np.argmin(y_scatter)
        x_opt = x_scatter[idx,:]
        y_opt = y_scatter[idx]

        # check convergence
        if stdev < xtol:
            converged = True
        if np.max(y_scatter) - np.min(y_scatter) < ftol:
            converged = True

        total_parallel_acquisitions += 1
        print("total_parallel_acquisitions: ", total_parallel_acquisitions)

        if obj_fn(x_opt) < obj_fn(x0):
            # PDF_opt = multivariate_normal.pdf(x_opt, x0, stdev*np.eye(Ndim))
            PDF_opt = multivariate_normal.pdf(x_opt, x0, 
                                              stdev/np.sqrt(Ndim)*np.eye(Ndim))
            stdev = (PDF_opt * Nscatter)**-(1/Ndim) # density-based
            # # stdev = np.sum((x_opt - x0)**2)**(1/Ndim) # step-based
            x0 = x_opt
            y0 = y_opt
        else:
            # if no improvement, shrink search area
            stdev *= 0.5 # halve distance
            # stdev *= 0.5**(1/Ndim) # halve hypervolume


        x_dist = np.sqrt(np.sum((x0 - 0.75)**2))

        # x_dists_history = np.hstack([x_dists_history, x_dist])
        # y_dists_history = np.hstack([y_dists_history, y0])
        # r_step_sizes = np.hstack([r_step_sizes, stdev])

    return(x_opt, y_opt, total_parallel_acquisitions)

