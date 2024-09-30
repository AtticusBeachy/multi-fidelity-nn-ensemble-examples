import math
import numpy as np
import tensorflow as tf
from tensorflow import keras

from moore_penrose_regression import moore_penrose_regression

def build_e2nn_two_layer(x_train, y_train, emulator_train, dtype, 
                         activation="fourier"):
    """
    builds an e2nn model with two hidden layers
    returns e2nn model and maximum weight from training
    """
    # setup parameters
    N_EMULATORS = emulator_train.shape[1]
    n_train, N_DIM = x_train.shape

    n_nodes_L1 = max(5*N_DIM, 200)
    n_nodes_L2 = max(2*n_train, 5000)

    # bias
    if "fourier" in activation:
        b_init = keras.initializers.RandomUniform(minval=0, maxval=2*math.pi)
    elif activation=="swish":
        b_init = keras.initializers.RandomUniform(minval=-4, maxval=4, seed=None)
    else:
        Error("Activation not 'fourier' varient or 'swish'")

    # Initialize e2nn model 
    x_input = keras.Input(shape=[N_DIM], name="x", dtype = dtype)
    emulator_input = keras.Input(shape=[N_EMULATORS], dtype = dtype, name="emulators")

    L1_ordinary = keras.layers.Dense(n_nodes_L1, input_shape = [N_DIM],
                                     activation=activation,
                                     dtype = dtype,
                                     kernel_initializer="glorot_normal", 
                                     bias_initializer=b_init,
                                     # kernel_regularizer=reg,
                                     name="hidden1"
                                     )(x_input) 
    L1 = keras.layers.concatenate([L1_ordinary, x_input, emulator_input], axis=1) 

    L2_ordinary = keras.layers.Dense(n_nodes_L2, 
                                     input_shape=[n_nodes_L1+N_DIM+N_EMULATORS],
                                     activation=activation,
                                     dtype = dtype,
                                     kernel_initializer="glorot_normal", 
                                     bias_initializer=b_init,
                                     # kernel_regularizer=reg,
                                     name="hidden2"
                                     )(L1)
    L2 = keras.layers.concatenate([L2_ordinary, x_input, emulator_input], axis=1)

    nn_output = keras.layers.Dense(1, name="prediction",
                                   input_shape=[n_nodes_L2+N_DIM+N_EMULATORS],
                                   activation = "linear",
                                   dtype = dtype
                                   )(L2)

    model = keras.Model(
                        inputs=[x_input, emulator_input],
                        outputs = [nn_output]
                        )
    keras.utils.plot_model(model, "model_two_layer.png", show_shapes=True)

    # Get hidden activations for training
    model_hidden = keras.Model(
                              inputs = [x_input, emulator_input],
                              outputs = [L2]
                              )
    # Retracing occurs for this prediction. Retracing is slow, but shouldn't
    # matter for a single query
    x_hidden = model_hidden.predict([x_train, emulator_train])
    
    # Rapid training
    Beta = moore_penrose_regression(x_hidden, y_train)

    biases = np.array([np.float64(Beta[0])]).flatten() #np.float32(Beta[0]) #
    weights = np.float64(Beta[1:]) #np.float32(Beta[1:]) #

    max_weight = np.max(np.abs(weights.flatten()))

    weights = tf.convert_to_tensor(weights)
    biases = tf.convert_to_tensor(biases)

    # over-write weights using linear regression
    model.get_layer("prediction").weights[0].assign(weights)
    model.get_layer("prediction").weights[1].assign(biases)

    return(model, max_weight)

