# multi-fidelity-nn-ensemble-examples

These codes can be used to replicate the numerical examples in the [Journal article](https://authors.elsevier.com/a/1hv-g,3rlesMFh) "Epistemic Modeling Uncertainty of Rapid Neural Network Ensembles for Adaptive Learning". The Journal article can be downloaded for free at the link until December 1, 2023. The preprint can be accessed [here](https://arxiv.org/abs/2309.06628). Results will differ somewhat based on random sample initializations, randomness in the neural network initializations, and randomness in the optimizers. 

Instructions:
main_e2nn_adaptive_sampling.py contains all of the example problems. Simply uncomment the desired problem formulation at the beginning of the script (under code section (1) USER SPECIFIED VARIABLES). Note that the Fun3D problem will not run, because it requires a Fun3D installation and a mesh file. However, the code is included for completeness.

Unfortunately, TensorFlow has a memory leak. Memory leakage occurs whenever a NN is created. This will eventually cause an out-of-memory crash when performing adaptive learning with an ensemble of NN models. 

This problem can be alleviated by running the main_relaunch.py script, which will call main_e2nn_adaptive_sampling.py repeatedly until it completes without error. Before running main_relaunch.py, set load_data=True at the start of main_e2nn_adaptive_sampling.py. This will ensure the script picks up where it left off when it is re-run after crashing. Also, delete or move the file state_to_load.pkl (if it exists) so that it is not loaded on the first run of main_e2nn_adaptive_sampling.py. 

(Update 10/18/2024)
The latest version of TensorFlow has a [bug](https://github.com/keras-team/keras/issues/20333) which causes a crash whenever custom activation functions are used. 

For example, when defining a custom activation function as described in the documentation [here](https://www.tensorflow.org/api_docs/python/tf/keras/utils/get_custom_objects)
```python
import tensorflow as tf
from tensorflow.keras.utils import get_custom_objects
fourier = lambda x : tf.sin(freq*x)
get_custom_objects()["fourier"] = Activation(fourier)
```
the code will crash if the custom activation function is ever used. 

The code has been updated to work around this issue. 


Package versions used include:
```
Package                       Version          
----------------------------- ---------------- 
tensorflow                    2.17.0  
keras                         3.5.0  
scikit-learn                  1.5.2  
numpy                         1.26.4  
scipy                         1.14.1  
matplotlib                    3.9.2  
pydot                         3.0.2
pyDOE3                        1.0.4  
```

