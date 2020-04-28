import numpy as np
import matplotlib.pyplot as plt
import functionlib as flib
""" Hyperparameters """
""" SGD, ADAM, CG, L_BFGS"""
iterations = 1000
learn_rate = 0.07
features = 2 # input dimension
batch_size = 128
N_train = 2048 # number of training observations
layer_width = [50, 1] # width of layers
N_layers = len(layer_width) # number of hidden layers
random_seed = 1337
initial = 1
""" Generate data """
np.random.seed(random_seed)
x = np.random.uniform(0,1,size=(features, N_train)) # each columns is one observations, the row represent features
y = flib.circles_classification(x)
""" Plot Result """
weights, biases, loss_list, time_list, gradient_list = flib.ADAM(x, y, layer_width, batch_size, learn_rate, iterations, initial, random_seed)
plt.figure()
flib.plot_results(weights, biases)
plt.show()