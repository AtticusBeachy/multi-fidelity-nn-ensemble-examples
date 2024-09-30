import numpy as np
from pyDOE3 import lhs

from minimize_local import minimize_local


def minimize_global(obj_fn, lb, ub, Ndim, total_parallel_acquisitions, 
                    n_scatter_init = 100_000, n_scatter_check = 1000, 
                    n_local_opts = 10, previous_local_xopt = np.array([]), 
                    n_scatter_gauss = 128):
    """
    Meant for optimizing functions that are cheap to evaluate in parallel but 
    difficult with many local minima
    """

    ###########################################################################
    """Scatter points"""
    print("Scattering "+str(n_scatter_init)+" points")
    print("Start LHS")  # sometimes runs out of memory and crashes here
    if Ndim * n_scatter_init <= 10_000:
        X_scatter = lhs(Ndim, samples=n_scatter_init, 
                        criterion="maximin", iterations=10)
        X_scatter = (ub-lb)*X_scatter + lb
        Y_scatter = obj_fn(X_scatter)
    else:
        divisor = 10
        n_scatter_step = n_scatter_init // divisor
        X_scatters = []
        for ii in range(divisor):
            print("iter", ii+1, "of", divisor)
            X_scatter_new = lhs(Ndim, samples=n_scatter_step, 
                                criterion="maximin", iterations=5)
            X_scatter_new = (ub-lb)*X_scatter_new + lb
            X_scatters.append(X_scatter_new)
        Y_scatters = []
        for X_scatter in X_scatters:
            Y_scatter = obj_fn(X_scatter)
            Y_scatters.append(Y_scatter.reshape([-1, 1]))
        Y_scatter = np.vstack(Y_scatters)
        Y_scatter = Y_scatter.flatten()
        X_scatter = np.vstack(X_scatters)
    # from scipy.stats.qmc import LatinHypercube
    # sampler = LatinHypercube(d = Ndim) #, optimization = "random-cd")
    # X_scatter = sampler.random(n=n_scatter_init)
    print("End LHS")

    sort_idx = np.argsort(Y_scatter.flatten())

    Y_scatter = Y_scatter[sort_idx]
    X_scatter = X_scatter[sort_idx,:]

    from sklearn.preprocessing import MinMaxScaler
    x_lim = np.vstack([lb, ub])
    xscale_obj = MinMaxScaler(feature_range=(0, 1)).fit(x_lim) #X_scatter)
    X_scatter_scaled = xscale_obj.transform(X_scatter)
    print("done scattering points")
    ###########################################################################
    """Select local minima"""
    print("Selecting local minima")
    # initial hypercube exclusion zone side length is 
    # 2 times (Volume/Npt)**1/Ndim 
    # (scales up volume by 2**Ndim)
    # (later try 4 times, scaling up volume by 4**Ndim)
    # Select top 10 local optima

    # exclusion_dist = 0.5 * Ndim * (1/n_scatter_init)**(1/Ndim) # half length
    exclusion_dist = Ndim * (1/n_scatter_init)**(1/Ndim) # manhattan exclusion distance (diag len is double this)
    # exclusion_dist = 2*Ndim * (1/n_scatter_init)**(1/Ndim) # double exclusion distance to allow some margin

    """only check N_check best optima"""
    # # scale down problem size (scalability)
    # N_check = 100
    # X_scatter_scaled = X_scatter_scaled[:Ncheck, :]

    local_optima_x = X_scatter_scaled[0,:].reshape([1,-1])
    local_optima_y = np.array([Y_scatter[0]])
    # local_optima_idx = np.array([0])

    exclusion_points = local_optima_x.copy()

    for ii in range(1, n_scatter_check): #n_scatter_init): #

        # check convergence
        if local_optima_y.size >= n_local_opts:
            break

        x_check = X_scatter_scaled[ii,:]

        manhattan_dists = np.sum(np.abs(x_check - exclusion_points), axis = 1)

        if np.all(manhattan_dists > exclusion_dist):
            local_optima_x = np.vstack([local_optima_x, x_check])
            local_optima_y = np.vstack([local_optima_y, Y_scatter[ii]])

        exclusion_points = np.vstack([exclusion_points, x_check])

    # local_optima_y = Y_scatter[local_optima_idx, :]
    local_optima_x_init = local_optima_x.copy()
    local_optima_y_init = local_optima_y.copy()

    # global Noptimizations
    # Noptimizations = local_optima_y.size

    """ check all Nscatter points, but discard more quickly """
    # Track local optimas and exclusion hyperspheres around them. Hypersphere 
    # radius is distance to furthest point discarded by hypershphere, plus some
    # exclusion distance (either n_scatter_init**(-1/Ndim) or double that). 
    # Each scatter point is checked against all local optima hyperspheres, and 
    # discarded by the nearest if it falls within any.

    """combine local optimas found using both scalable methods"""
    # use the "unique points" thing with a tighter tolerance (1e-9 or something)

    print("Done selecting local minima")

    ############################################################################
    """Perform optimization to improve local minima"""

    # # option 1: L-BFGS-B
    #
    # # bounds = np.hstack([lb.reshape(-1,1), ub.reshape(-1, 1)])
    # print("np.zeros([lb.size, 1]): ", np.zeros([lb.size, 1]))
    # print("np.ones([ub.size, 1]): ", np.ones([ub.size, 1]))
    # bounds = np.hstack([np.zeros([lb.size, 1]), np.ones([ub.size, 1])])
    # # obj_fn_scaled = lambda x : obj_fn(xscale_obj.inverse_transform(x))
    # obj_fn_scaled = lambda x : obj_fn(xscale_obj.inverse_transform(x.reshape([-1, Ndim])))
    #
    # for ii in range(local_optima_y.size):
    #     x0 = local_optima_x[ii,:]
    #     # x0 = x0.reshape((1, -1))
    #     print("x0: ", x0)
    #     y0 = local_optima_y[ii]
    #     res = minimize(obj_fn_scaled, x0 = x0, bounds = bounds, method="L-BFGS-B", options={"maxcor": 2*Ndim, "ftol": 1e-6, "eps": 1e-03, "maxfun": 15000, "maxiter": 15000}) # "ftol": 1e-6, "eps": 1e-06
    #     # res = minimize(obj_fn_scaled, x0 = x0, bounds = bounds, method="SLSQP", options={"ftol": 1e-7, "eps": 1e-3}) # "ftol": 1e-7, "eps": 1e-7
    #
    #     # update with the refined local optimum, but only if it is an improvement
    #     if res.fun < y0:
    #         local_optima_x[ii,:] = res.x
    #         local_optima_y[ii] = res.fun
    #         print("Optimum improved for local optimum "+str(ii+1)+" of "+str(local_optima_y.size))
    #     else:
    #         local_optima_x[ii,:] = x0
    #         local_optima_y[ii] = y0
    #         print("Optimation failed to improve for local optimum "+str(ii+1)+" of "+str(local_optima_y.size))


    # option 2: Gaussian scatter (must choose standard deviation) [based off of local sample density from last iter]
    # standard deviation = stacked spacing
    obj_fn_scaled = lambda x : obj_fn(xscale_obj.inverse_transform(x.reshape([-1, Ndim])))
    bounds = np.hstack([np.zeros([lb.size, 1]), np.ones([ub.size, 1])])
    step0 = n_local_opts**-(1/Ndim) # n_scatter_init**-(1/Ndim)
    for ii in range(local_optima_y.size):
        # print("local_optima_x: ", local_optima_x)
        # in case only a single local optima found, change to 2D
        if len(local_optima_x.shape)==1:
            local_optima_x = local_optima_x.reshape([1,-1])
        x0 = local_optima_x[ii,:]

        # gaussian scatter
        [x_opt, y_opt, total_parallel_acquisitions] = minimize_local(
            obj_fn_scaled, x0, step0, bounds, total_parallel_acquisitions, 
            Nscatter = n_scatter_gauss, ftol = 1e-6, xtol = 1e-4
        ) #, ftol = 1e-6, xtol = 1e-5)

        local_optima_x[ii,:] = x_opt
        local_optima_y[ii] = y_opt


    ############################################################################
    """Select final optimum"""

    x_opts = xscale_obj.inverse_transform(local_optima_x)
    y_opts = local_optima_y

    idx = np.argmin(y_opts)
    x_opt = x_opts[idx, :]
    y_opt = y_opts[idx]

    # de-scale
    exclusion_points = xscale_obj.inverse_transform(exclusion_points)
    local_optima_x_init = xscale_obj.inverse_transform(local_optima_x_init)

    return(x_opt, y_opt, x_opts, y_opts, exclusion_points, 
           X_scatter, Y_scatter, local_optima_x_init, local_optima_y_init, 
           total_parallel_acquisitions
    )

