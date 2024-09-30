import numpy as np

def get_mse(y_pred, y_true, n_param=0):
    '''
    Mean squared error of predictions
    '''
    y_pred = y_pred.flatten()
    y_true = y_true.flatten()
    n_pt = len(y_pred)
    mse = np.sum((y_pred - y_true)**2)/(n_pt-n_param)
    return(mse)

