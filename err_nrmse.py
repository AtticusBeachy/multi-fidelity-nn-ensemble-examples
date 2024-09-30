import numpy as np

def err_nrmse(y_pred, y_true):
    '''
    Additive normalized root mean squared error
    Use when modeling variables that have additive effects
    '''
    y_pred = y_pred.flatten()
    y_true = y_true.flatten()
    y_mean = np.mean(y_true)
    nrmse = np.sqrt(np.sum((y_pred-y_true)**2)/np.sum((y_mean-y_true)**2))
    return(nrmse)

