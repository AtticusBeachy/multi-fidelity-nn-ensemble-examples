import pickle
import math
import numpy as np
from pyDOE2 import lhs, ff2n



def get_training_doe(Nsamp, Ndim, sample_corners = False):
    """ Get training data locations X_train """

    if sample_corners:
        filename_train = f"X_train_E2NN_{Nsamp}_{Ndim}D_samples"
    else:
        filename_train = f"X_train_E2NN_{Nsamp}_{Ndim}D_samples_no_FF"
    
    try: 
        # load data
        with open(filename_train, "rb") as infile:
            X_train = pickle.load(infile)
    except: 
        # generate data
        if Ndim == 1:
            X_train = np.linspace(0, 1, Nsamp) 
            X_train = X_train.reshape(-1, 1) # make 2D
        
        else: # need latin hypercube
            
            if sample_corners:
                
                print("Generating training samples using LHS")
                assert Nsamp>=2**Ndim, "The specified number of training samples is insufficient to sample all corners of the design space."
                X_lhs = lhs(Ndim, samples=Nsamp-2**Ndim, criterion="maximin", iterations=100_000)#20000)#20)
                X_ff = ff2n(Ndim)
                X_ff[X_ff<0] = 0
                X_train = np.concatenate((X_lhs, X_ff), axis=0)
                print("DOE of training points complete")

            else:
                print("Generating training samples using LHS")
                X_train = lhs(Ndim, samples=Nsamp, criterion="maximin", iterations=100_000)#20000)#20)
                print("DOE of training points complete")

        # save data
        with open(filename_train, "wb") as outfile:
            pickle.dump(X_train, outfile)

    return(X_train)


def get_test_doe(Ntest, Ndim):
    """ Get test data locations X_test """

    filename_test = f"X_test_E2NN_{Ntest}_{Ndim}D_samples"
    
    try: 
        # load data
        with open(filename_test, "rb") as infile:
            X_test = pickle.load(infile)
    except: 
        # generate data
        if Ndim == 1: 
            X_test = np.linspace(0, 1, Ntest) # ignore 
            X_test = X_test.reshape(-1, 1) # make 2D
            
        elif Ndim == 2:
            X1_test = np.linspace(0, 1, math.ceil(Ntest**0.5)) #33)#65)#17) #
            X2_test = np.linspace(0, 1, math.ceil(Ntest**0.5)) #33)#65)#17) #
            (X1_test, X2_test) = np.meshgrid(X1_test, X2_test)

            X1_test = X1_test.reshape(-1, 1) # 2D column
            X2_test = X2_test.reshape(-1, 1) # 2D column

            X_test = np.hstack([X1_test, X2_test]) # fuse

        else:
            # Get samples using LHS
            print("Generating test samples using LHS")
            X_test = lhs(Ndim, samples=Ntest, criterion="maximin", iterations=1_000)#20000)#20)
            print("DOE of test points complete")

        # save data
        with open(filename_test, "wb") as outfile:
            pickle.dump(X_test, outfile)

    return(X_test)