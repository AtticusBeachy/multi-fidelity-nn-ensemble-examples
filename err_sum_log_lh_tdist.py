import numpy as np
from scipy.stats import t

def err_sum_log_lh_tdist(xs, t_dists):
    """
    Error metric, similar to NRMSE but for predicted Probability 
    Density Functions (PDFs) instead of point estimates. 
    In this case, the PDFs are t-distributions. 
    
    Arguments
    ---------

    xs (numpy array): observed data
    t_dists (class): predictions of observed data
        t_dists.mu (numpy array): means of predictive t-distributions
        t_dists.scale (numpy array): scales of predictive t-distributions
        t_dists.df (int): degrees of freedom (same for all t-distributions)

    Returns
    -------

    sum_log_lh (numpy.float64): the sum of the logs of the likelihoods
    of the observed data based on the predictive t-distributions. (Log is
    base e)
    """
    xs = xs.flatten()
    loc = t_dists.mu.flatten()
    scale = t_dists.sig.flatten()
    log_lh = t.logpdf(xs, t_dists.df, loc=loc, scale=scale)
    sum_log_lh = np.sum(log_lh)
    return(sum_log_lh)

