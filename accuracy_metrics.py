################################################################################
'''ACCURACY MEASURES'''

import numpy as np
from scipy.stats import norm
from scipy.stats import t


def R2(y_pred, y_true, p=0):
    '''
    R squared value
    '''
    y_pred = y_pred.flatten()
    y_true = y_true.flatten()
    y_mean = np.mean(y_true)
    Rsquared = 1-np.sum((y_pred-y_true)**2)/np.sum((y_mean-y_true)**2)
    return(Rsquared)

def NRMSE(y_pred, y_true):
    '''
    Additive normalized root mean squared error
    Use when modeling variables that have additive effects
    '''
    y_pred = y_pred.flatten()
    y_true = y_true.flatten()
    y_mean = np.mean(y_true)
    additive_NRMSE = np.sqrt(np.sum((y_pred-y_true)**2)/np.sum((y_mean-y_true)**2))
    return(additive_NRMSE)

def MNRMSE(y_pred, y_true):
    '''
    Multiplicative normalized root mean squared error
    Use when modeling variables that have multiplicative effects
    '''
    y_pred = y_pred.flatten()
    y_true = y_true.flatten()
    N = len(y_pred)
    multiplicative_NRMSE = np.sqrt(np.sum( ((y_pred-y_true)/y_true)**2 )/N)
    return(multiplicative_NRMSE)

def RMSE(y_pred, y_true,Nparam=0):
    '''
    Root mean squared error
    Use when error magnitudes are desired in the original units

    Nparam is the number of model parameters (only used if y_pred are predictions
    of a model constructed using y_true, and the RMSE is being used to
    estimate the model residuals)
    '''
    y_pred = y_pred.flatten()
    y_true = y_true.flatten()
    N = len(y_pred)
    rmse = np.sqrt(np.sum((y_pred - y_true)**2)/(N-Nparam))
    return(rmse)

def MSE(y_pred, y_true,Nparam=0):
    '''
    Mean squared error
    '''
    y_pred = y_pred.flatten()
    y_true = y_true.flatten()
    N = len(y_pred)
    mse = np.sum((y_pred - y_true)**2)/(N-Nparam)
    return(mse)

def sum_log_likelihood_norm(y_true, y_pred, scale):
    """
    returns the sum of log likelihoods for a normal distribution
    an error metric that takes model uncertainty or PDF into account
    """
    SLL = np.sum(norm.logpdf(y_true, loc=y_pred, scale=scale))
    return(SLL)

# class Tdists:
#     def __init__(self, mu, sig, df):
#         self.mu = mu
#         self.sig = sig
#         self.df = df

def sum_log_likelihood_tdist(xs, t_dists):
    """
    Error metric, similar to NRMSE but accounts for uncertainty
    Best way of measuring probability distribution accuracy
    """
    # Returns: the sum of log likelhoods of all the t-scores
    #          (base e)
    # xs: the data locations to check likelihood at
    # t_dists: the t-distributions of the predictions (custom class)
    SLL = 0
    for ii in range(len(xs)):
        SLL += t.logpdf(xs[ii], t_dists.df, loc=t_dists.mu[ii], scale=t_dists.sig[ii])
    # return(np.exp(SLL)) # log likelihood
    return(SLL)
