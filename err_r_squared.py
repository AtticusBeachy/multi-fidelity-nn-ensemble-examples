import numpy as np

def get_r_squared(y_pred, y_true):
    '''
    R squared value
    '''
    y_pred = y_pred.flatten()
    y_true = y_true.flatten()
    y_mean = np.mean(y_true)
    Rsq= 1-np.sum((y_pred-y_true)**2)/np.sum((y_mean-y_true)**2)
    return(Rsq)

