import numpy as np

def get_rmse(y_pred, y_true, n_param=0):
    '''
    Root mean squared error
    Use when error magnitudes are desired in the original units

    n_param is the number of model parameters (only used if y_pred are
    predictions of a model constructed using y_true, and the RMSE is being used
    to estimate the model residuals)
    '''
    y_pred = y_pred.flatten()
    y_true = y_true.flatten()
    n_pt = len(y_pred)
    rmse = np.sqrt(np.sum((y_pred-y_true)**2)/(n_pt-n_param))
    return(rmse)

