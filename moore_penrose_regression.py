import numpy as np
from scipy.linalg import svd

eps64 = np.finfo("float").eps
eps32 = np.finfo("float32").eps
eps = 1e-6 

def moore_penrose_regression(X, Y):
    # linear regression but more stable (singular value decomposition)
    k = eps32 #0 #eps64 #1e-5 #1e-11 #1e-9 #
    Nsamp = X.shape[0]
    Xreg = np.hstack([np.ones([Nsamp, 1]), X])
    U, Sig, Vt = svd(Xreg, full_matrices = True) #False) #
    V = Vt.T
    # numerically stabalize Sig
    # option 1: clamp
    sign = np.sign(Sig)
    sign[sign==0] = 1
    idx = np.abs(Sig) < k
    Sig[idx] = k * sign[idx]
    # # option 2: add everywhere
    # sign = np.sign(Sig)
    # sign[sign==0] = 1
    # Sig = Sig + sign*k
    # end numerically stabalize Sig
    Sig_pseudoinverse = np.zeros(Xreg.shape[::-1])
    np.fill_diagonal(Sig_pseudoinverse, 1/Sig)
    # Sig_pseudoinverse = np.linalg.pinv(Sig, rcond = k)
    Beta = V @ Sig_pseudoinverse @ U.T @ Y
    return(Beta)

