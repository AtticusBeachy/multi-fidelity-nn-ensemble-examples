import pickle
import math
import numpy as np
from pyDOE3 import lhs

def get_test_doe(N_TEST, N_DIM, LB, UB):
    """ Get test data locations X_test """

    filename_test = f"X_test_E2NN_{N_TEST}_{N_DIM}D_samples"
    
    try: 
        # load data
        with open(filename_test, "rb") as infile:
            X_test = pickle.load(infile)
    except: 
        # generate data
        if N_DIM == 1: 
            X_test = np.linspace(0, 1, N_TEST) # ignore 
            X_test = X_test.reshape(-1, 1) # make 2D
            
        elif N_DIM == 2:
            X1_test = np.linspace(0, 1, math.ceil(N_TEST**0.5)) #33)#65)#17) #
            X2_test = np.linspace(0, 1, math.ceil(N_TEST**0.5)) #33)#65)#17) #
            (X1_test, X2_test) = np.meshgrid(X1_test, X2_test)

            X1_test = X1_test.reshape(-1, 1) # 2D column
            X2_test = X2_test.reshape(-1, 1) # 2D column

            X_test = np.hstack([X1_test, X2_test]) # fuse

        else:
            # Get samples using LHS
            print("Generating test samples using LHS")
            X_test = lhs(N_DIM, samples=N_TEST, criterion="maximin", iterations=1_000)#20000)#20)
            print("DOE of test points complete")

        # save data
        with open(filename_test, "wb") as outfile:
            pickle.dump(X_test, outfile)

    x_test_raw = (UB-LB)*X_test + LB
    return(x_test_raw)
