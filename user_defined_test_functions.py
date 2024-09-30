###############################################################################
"""USER-DEFINED HIGH-FIDELITY AND LOW-FIDELITY FUNCTIONS"""

import numpy as np
import random


def rosenbrock(X, c1=100.0, c2=1.0):
    """
    Takes in a 2D matrix of samples and outputs 2D column matrix of
    Rosenbrock function responses
    The function must be 2D or higher
    """
    Y = np.sum(c1*(X[:,1:] - X[:,:-1]**2.0)**2.0 +c2* (1 - X[:,:-1])**2.0, axis=1)
    # if random.random() < 0.05:
    #     Y = None
    return(Y)

def rosen_univariate(X, dim, N_DIM):
    """
    A Low Fidelity univariate version of the Rosenbrock function for use as an
    emulator
    X: A 2D numpy array (MUST BE 2D)
    dim: The dimension selected for the univariate function (0 to N_DIM-1)
    N_DIM: The total number of dimensions
    """
    c1 = 100.0* 1.5 * (dim+1) / N_DIM
    c2 = 1.0 + 2 * (dim+1) / N_DIM
    Xuni = np.zeros(X.shape) # cut through middle of design space (varies with bounds)
    Xuni[:, dim] = X[:,dim]
    Y = rosenbrock(Xuni, c1, c2)
    return(Y)#, Xuni)

def get_rosen_emulator(dim, N_DIM):
    """
    Returns a lambda function for use as an emulator
    """
    emulator = lambda X : rosen_univariate(X, dim = dim, N_DIM = N_DIM)
    return(emulator)

def sum_emulator(X, Fe):
    """
    Sums input functions
    """
    Y = 0
    for emulator in Fe:
        Y += emulator(X)
    return(Y)

def return_sum_emulator(Fe):
    """
    Returns a lambda function that is the sum of the input functions
    """
    emulator_sum = lambda X : sum_emulator(X, Fe)
    return(emulator_sum)

def scaled_sphere_rosenbrock(X):
    """
    sphere function with scaling appropriate to n-dim rosenbrock
    """
    Y = 100*np.sum(X**2, axis=1)
    return(Y)

def nonstationary_1d_hf(X):
    """
    A nonstationary 1dim HF function commonly used as as example
    X: A numpy array (I think it can be a 1D or 2D array, but not sure)
    """
    Y = (6*X-2)**2 * np.sin(12*X-4)
    # Y = (6*X-2)**2 + 10*X**5
    return(Y)

def nonstationary_1d_lf(X):
    """
    A low fidelity version of the nonstationary 1D function above
    X: A numpy array (I think it can be 1D or 2D, but not sure)
    """
    Y = 0.5*nonstationary_1d_hf(X) + 10*(X-0.5) - 5
    #Y = 0.1*nonstationary_1d_hf(X) + 10*(X-0.5)
    #Y = 0.1*nonstationary_1d_hf(X) + 10*(X-0.5) + 9
    return(Y)


def uninformative_1d(X):
    """
    An uninformative 1D function
    X: A numpy array 
    """
    # Y = np.zeros(np.shape(X))
    # return(Y)
    return(np.zeros(X.shape))


def nonstationary_2d_hf(X):
    """

    """
    x1 = X[:,0]
    x2 = X[:,1]
    # x1 = x1 + 0.05
    Y = np.sin(21*(x1-0.9)**4)*np.cos(2*(x1-0.9))+0.5*(x1-0.7)+2*x2**2*np.sin(x1*x2)
    return(Y)

def nonstationary_2d_lf(X):
    """

    """
    x1 = X[:,0]
    x2 = X[:,1]
    Yhf = nonstationary_2d_hf(X)
    Y = (Yhf-2+x1+x2)/(1+0.25*x1+0.5*x2)
    return(Y)

def uninformative_nd_lf(X):
    """
    An uninformative low fidelity function
    X: A numpy array (Must be 1D array for 1D functions, and 2D array otherwise)
    """
    Y = np.zeros(X.shape[0])
    return(Y)

def rastrigin_function(X):
    """ Optimum at X = [0,...,0], Y = 0 """
    N_DIM = X.shape[1]
    Y = 10*N_DIM + np.sum(X**2 - 10*np.cos(2*np.pi*X), axis = 1)
    return(Y)

def ackley_function(X):
    """ Optimum at f(0,0) = 0 """
    x1 = X[:,0]
    x2 = X[:,1]
    Y = -20*np.exp(-0.2*np.sqrt(0.5*(x1**2+x2**2))) - np.exp(0.5*(np.cos(2*np.pi*x1)+np.cos(2*np.pi*x2))) + np.e + 20
    return(Y)

###############################################################################
''' CALLING FUN3D '''

def fun3d_simulation_normx(x, out_name, subfolder):
    """ Modify input deck, run fun3d simulation, and read output deck to get results """
    
    import os
    import subprocess

    ############################# INPUT VARIABLES
    #            Mach, AoA, Altitude
    # 5
    ub = np.array([4.0,  8., 50_000.]) #6., 10., 25_000.
    lb = np.array([1.2, -5., 0.]) #-2.
    #ub = np.array([6.0,  10., 50_000.]) #6., 10., 25_000.
    #lb = np.array([1.2, -5., 0.]) #-2.

    x = x*(ub-lb) + lb

    ############################# FUNCTIONS FOR READING AND WRITING FILES

    from skaero.atmosphere import coesa
    from scipy import interpolate

    def edit_input(x):
        """ Edit fun3d.nml file """
        # unpack input x
        x = x.flatten() # only a single point
        x_mach = x[0]
        x_aoa = x[1]
        x_height = x[2]
        x_yaw = 0.0

        ###### extract atmospheric properties

        #[T, a, __, Rho] = atmoscoesa(x_height);
        h, T, p, rho = coesa.table(x_height)
        
        # Get speed of sound
        # Properties table: 
        # https://www.cambridge.org/us/files/9513/6697/5546/Appendix_E.pdf
        # Alternative calculation: https://www.grc.nasa.gov/www/BGH/sound.html
        t_table = np.array([200., 220., 240., 260., 280., 300., 320., 340.])
        cv_table = np.array([
            0.7153, 0.7155, 0.7158, 0.7162, 0.7168, 0.7177, 0.7188, 0.7202])
        cp_table = np.array([
            1.002, 1.003, 1.003, 1.003, 1.004, 1.005, 1.006, 1.007])
        gamma_table = cp_table/cv_table    
        
        interpolation_class = interpolate.interp1d(t_table, gamma_table, 
                                                   fill_value="extrapolate")
        
        gamma = interpolation_class(T)
        
        # # calorically perfect air:    
        # gamma_perfect = 1.4
        # a = np.sqrt(R*T*gamma_perfect)
        
        # calorically imperfect air:
        R = 286 # m^2/s^2/K
        a = np.sqrt(R*T*gamma)
        vel = a*x_mach
        
        # viscosity from Sutherland's Formula
        # (https://www.grc.nasa.gov/WWW/K-12/airplane/viscosity.html)
        S = 198.72/1.8 # R to K
        T0 = 518.7/1.8 # R to K
        mu0 = 3.62e-7 * 4.448222 * 1/0.3048**2 #lb-s/ft^2 to N-s/m^2 (est 1.716e-5)
        mu = mu0*(T/T0)**1.5*(T0+S)/(T+S);
        Len = 4.47 # Aircraft length (meters)
        Re = Len*vel*rho/mu
        

        # # Write variables
        # T       temperature = 221.65
        # rho     density = 0.039466     ! kg/m^3
        # Re      reynolds_number = 19932640.6964
        # x_mach  mach_number     = 0.95
        # vel     velocity = 283.5323    ! m/s
        # x_aoa   angle_of_attack = 10   ! degrees
        # none    angle_of_yaw = 0.0     ! degrees
        
        ###### Write atmospheric properties to file
        in_name = 'fun3d.txt'
        file1 = open(in_name, "r") # in_file
        lines = file1.readlines()
        file1.close()
        
        write_name = 'fun3d.nml';
        file2 = open(write_name, 'w') # out_file

        for line in lines:
            if 'mach_number' in line:
                file2.write('  mach_number     = '+str(x_mach)+'\n')
            elif 'angle_of_attack' in line:
                file2.write('  angle_of_attack = '+str(x_aoa)+'\n')
            elif 'angle_of_yaw' in line:
                file2.write('  angle_of_yaw = '+str(x_yaw)+'\n')
            elif 'density' in line:
                file2.write('  density = '+str(rho)+'\n') 
            elif 'temperature' in line and '_units' not in line:
                file2.write('  temperature = '+str(T)+'\n')
            elif 'velocity' in line:
                file2.write('  velocity = '+str(vel)+'\n') 
            elif 'reynolds_number' in line:
                file2.write('  reynolds_number = '+str(Re)+'\n') 
            else:
                file2.write(line)
        file2.close()
        return(None)


    import time
    from os.path import exists

    def read_output(out_name):
        # check output exists
        out_file = out_name+'_hist.dat'
        while True:
            output_exists = exists(out_file)
            if output_exists:
                print('------------- CFD simulation done ! ----------------')
                time.sleep(1)
                break
            else:
                print("No output deck found")
                time.sleep(0.1)


        # extract results
        file1 = open(out_file, "r")
        tline = file1.readline()
        tline = file1.readline()
        if 'R_6' in tline:
            cl_idx = 7
            cd_idx = 8
        else: # inviscid
            cl_idx = 6
            cd_idx = 7
       
        lines = file1.readlines()
        last_line = lines[-1]
        last_line = last_line.split(' ')
        # remove elements with only spaces
        last_line = [line for line in last_line if len(line.strip())>0] 
        last_line = list(map(float, last_line))
        CL = last_line[cl_idx]
        CD = last_line[cd_idx]
        return(CL, CD)


    ############################# MODIFY FILE

    # subfolder = "GHV_494k_v" #"GHV_34k_v" #
    os.chdir(subfolder)
    edit_input(x)


    ############################# RUN MODIFIED FILE

    # path = os.getcwd()
    # print("path: ", path)

    # run fun3d
    subprocess.run(["nodet_mpi"])


    ############################# EXTRACT RESULTS

    #out_name = "GHV02_494k"
    CL, CD = read_output(out_name)

    os.chdir("../")

    return(CL, CD)


def viscous_simulation_cl(xdat, out_name, subfolder):
    """ viscous coefficient of lift """
    ndat = xdat.shape[0]
    CL = np.zeros([ndat, 1])
    CD = np.zeros([ndat, 1])
    for ii in range(ndat):
        xii = xdat[ii,:]
        CL[ii], CD[ii] = fun3d_simulation_normx(xii, out_name, subfolder)
    return(CL)


def viscous_simulation_cd(xdat, out_name, subfolder):
    """  viscous coefficient of drag  """
    ndat = xdat.shape[0]
    CL = np.zeros([ndat, 1])
    CD = np.zeros([ndat, 1])
    for ii in range(ndat):
        xii = xdat[ii,:]
        CL[ii], CD[ii] = fun3d_simulation_normx(xii, out_name, subfolder)
    return(CD)


def viscous_simulation_cl_cd(xdat, out_name, subfolder):
    """ viscous lift to drag ratio """
    ndat = xdat.shape[0]
    CL = np.zeros([ndat, 1])
    CD = np.zeros([ndat, 1])
    for ii in range(ndat):
        xii = xdat[ii,:]
        CL[ii], CD[ii] = fun3d_simulation_normx(xii, out_name, subfolder)
    return(CL/CD)



