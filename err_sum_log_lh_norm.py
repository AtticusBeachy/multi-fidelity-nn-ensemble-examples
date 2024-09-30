from scipy.stats import norm

def get_sum_log_lh_norm(y_true, y_pred, scale):
    """
    returns the sum of log likelihoods for a normal distribution
    (error metric that takes model uncertainty or PDF into account)
    """
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    sum_log_lh = np.sum(norm.logpdf(y_true, loc=y_pred, scale=scale)) 
    return(sum_log_lh)

