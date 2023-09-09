import math
import numpy as np
import tensorflow as tf
# from tensorflow.python.framework.ops import disable_eager_execution
# disable_eager_execution()
from tensorflow import keras
# physical_devices = tf.config.list_physical_devices("GPU")

eps64 = np.finfo("float").eps
eps32 = np.finfo("float32").eps
eps = 1e-6 #1e-3 #

def train_E2NN_RaNN_1L(X_train, Y_train, Emulator_train, Nemulator, dtype, Neurons="few", Activation="fourier"):

    ################################################################################
    """E2NN GRAPH STRUCTURE"""
    Nsamp, Ndim = X_train.shape
    x_input = keras.Input(shape=[Ndim], name="x", dtype = dtype)
    emulator_input = keras.Input(shape=[Nemulator], dtype = dtype, name="emulators")

    # begin first layer
    if "fourier" in Activation:
        b_init = keras.initializers.RandomUniform(minval=0, maxval=2*math.pi)
    elif Activation=="swish":
        b_init = keras.initializers.RandomUniform(minval=-4, maxval=4, seed=None)
    else:
        Error("Activation not 'fourier' varient or 'swish'")

    if Neurons=="few":
        N_nodes_L1 = 2*Nsamp
    elif Neurons=="many":
        N_nodes_L1 = 5000
    else:
        Error("Neurons not 'few' varient or 'many'")

    # reg = keras.regularizers.l2(1e-6)

    L1_ordinary = keras.layers.Dense(N_nodes_L1, input_shape = [Ndim],
                                     activation=Activation,
                                     dtype=dtype,
                                     kernel_initializer="glorot_normal",
                                     bias_initializer=b_init,
                                     kernel_regularizer=keras.regularizers.l2(1e-6),
                                     name="hidden1")(x_input) #x_concat) #
    # L1 = keras.layers.concatenate([L1_ordinary, emulator_input], axis=1)
    L1 = keras.layers.concatenate([L1_ordinary, x_input, emulator_input], axis=1) # CONNECT INPUT WITH OUTPUT
    # L1 = keras.layers.concatenate([L1_ordinary, x_concat], axis=1) # CONCATENATE INPUT AND EMULATORS WITH HIDDEN LAYER
    # end first layer


    # one layer only
    E2NN_output = keras.layers.Dense(1, name="prediction",
                                     input_shape=[N_nodes_L1+Ndim+Nemulator],
                                     activation = "linear",
                                     dtype = dtype)(L1)
    # end one layer only

    E2NN_model = keras.Model(
                             inputs=[x_input, emulator_input],
                             outputs = [E2NN_output]
                             )
    keras.utils.plot_model(E2NN_model, "E2NN_model.png", show_shapes=True)

    # Rapid Randomized Neural Network (R2NN)
    E2NN_random = keras.Model(
                              inputs = [x_input, emulator_input],
                              outputs = [L1] #[L2] #[L3] #
                              )

    # Retracing occurs for this prediction. Retracing is slow, but shouldn't
    # matter for a single query
    X_random = E2NN_random.predict([X_train, Emulator_train])

    # self-calculate linear regression but more stable (orthogonal decomposition)
    k = eps32 #0 #eps64 #1e-5 #1e-11 #1e-9 #
    from scipy.linalg import svd
    Xreg = np.hstack([np.ones([Nsamp, 1]), X_random]) # m=3 pt by n=501 dim
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
    Beta = V @ Sig_pseudoinverse @ U.T @ Y_train
    # Nreg = X_random.shape[1]+1
    # # Beta = np.linalg.inv(Xreg.T @ Xreg + k*np.eye(Nreg)) @ Xreg.T @ Y_train
    # Beta = np.linalg.lstsq(Xreg.T @ Xreg + k*np.eye(Nreg), Xreg.T)[0] @ Y_train
    biases = np.array([np.float64(Beta[0])]).flatten() #np.float32(Beta[0]) #
    weights = np.float64(Beta[1:]) #np.float32(Beta[1:]) #
    # end self-calculate stable

    max_weight = np.max(np.abs(weights.flatten()))

    weights = tf.convert_to_tensor(weights)
    biases = tf.convert_to_tensor(biases)

    # OVER-WRITE WEIGHTS USING LINEAR REGRESSION
    E2NN_model.get_layer("prediction").weights[0].assign(weights)
    E2NN_model.get_layer("prediction").weights[1].assign(biases)

    return(E2NN_model, max_weight)





def train_E2NN_RaNN_2L(X_train, Y_train, Emulator_train, Nemulator, dtype, Neurons="few", Activation="fourier"):

    ################################################################################
    """E2NN GRAPH STRUCTURE"""
    Nsamp, Ndim = X_train.shape
    x_input = keras.Input(shape=[Ndim], name="x", dtype = dtype)
    emulator_input = keras.Input(shape=[Nemulator], dtype = dtype, name="emulators")

    # begin first layer
    if "fourier" in Activation:
        b_init = keras.initializers.RandomUniform(minval=0, maxval=2*math.pi)
    elif Activation=="swish":
        b_init = keras.initializers.RandomUniform(minval=-4, maxval=4, seed=None)
    else:
        Error("Activation not 'fourier' varient or 'swish'")

    if Neurons=="few":
        N_nodes_L1 = 2*Nsamp
        N_nodes_L2 = 2*Nsamp
    elif Neurons=="many":
        N_nodes_L1 = 200 #2*Nsamp #
        N_nodes_L2 = 5000
    else:
        Error("Neurons not 'few' varient or 'many'")

    reg = keras.regularizers.l2(1e-6)

    # begin first layer
    L1_ordinary = keras.layers.Dense(N_nodes_L1, input_shape = [Ndim],
                                     activation=Activation,
                                     dtype = dtype,
                                     kernel_initializer="glorot_normal", #glorot_norm_nontrunc,
                                     bias_initializer=b_init,
                                     kernel_regularizer=reg,
                                     name="hidden1")(x_input) #x_concat) #
    # L1 = keras.layers.concatenate([L1_ordinary, emulator_input], axis=1)
    L1 = keras.layers.concatenate([L1_ordinary, x_input, emulator_input], axis=1) # CONNECT INPUT WITH OUTPUT
    # L1 = keras.layers.concatenate([L1_ordinary, x_concat], axis=1) # CONCATENATE INPUT AND EMULATORS WITH HIDDEN LAYER
    # end first layer


    # begin second layer
    L2_ordinary = keras.layers.Dense(N_nodes_L2, input_shape=[N_nodes_L1+Ndim+Nemulator],
                                     activation=Activation,
                                     dtype = dtype,
                                     kernel_initializer="glorot_normal", #glorot_norm_nontrunc,
                                     bias_initializer=b_init,
                                     kernel_regularizer=reg,
                                     name="hidden2")(L1)
    L2 = keras.layers.concatenate([L2_ordinary, x_input, emulator_input], axis=1)
    # end second layer


    # two layer only
    E2NN_output = keras.layers.Dense(1, name="prediction",
                                     input_shape=[N_nodes_L2+Ndim+Nemulator],
                                     activation = "linear",
                                     dtype = dtype)(L2)
    # end two layer only

    E2NN_model = keras.Model(
                             inputs=[x_input, emulator_input],
                             outputs = [E2NN_output]
                             )
    keras.utils.plot_model(E2NN_model, "E2NN_model.png", show_shapes=True)

    # Rapid Randomized Neural Network (R2NN)
    E2NN_random = keras.Model(
                              inputs = [x_input, emulator_input],
                              outputs = [L2]
                              )

    # Retracing occurs for this prediction. Retracing is slow, but shouldn't
    # matter for a single query
    X_random = E2NN_random.predict([X_train, Emulator_train])
    

    # self-calculate linear regression but more stable (orthogonal decomposition)
    k = eps32 #0 #eps64 #1e-5 #1e-11 #1e-9 #
    from scipy.linalg import svd
    Xreg = np.hstack([np.ones([Nsamp, 1]), X_random]) # m=3 pt by n=501 dim
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
    Beta = V @ Sig_pseudoinverse @ U.T @ Y_train
    # Nreg = X_random.shape[1]+1
    # # Beta = np.linalg.inv(Xreg.T @ Xreg + k*np.eye(Nreg)) @ Xreg.T @ Y_train
    # Beta = np.linalg.lstsq(Xreg.T @ Xreg + k*np.eye(Nreg), Xreg.T)[0] @ Y_train
    biases = np.array([np.float64(Beta[0])]).flatten() #np.float32(Beta[0]) #
    weights = np.float64(Beta[1:]) #np.float32(Beta[1:]) #
    # end self-calculate stable

    max_weight = np.max(np.abs(weights.flatten()))

    weights = tf.convert_to_tensor(weights)
    biases = tf.convert_to_tensor(biases)

    # OVER-WRITE WEIGHTS USING LINEAR REGRESSION
    E2NN_model.get_layer("prediction").weights[0].assign(weights)
    E2NN_model.get_layer("prediction").weights[1].assign(biases)

    return(E2NN_model, max_weight)