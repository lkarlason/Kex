import numpy as np
import matplotlib.pyplot as plt
import functionlib as flib
""" Hyperparameters """
""" SGD, ADAM, CG, L_BFGS"""
iterations = 6000
learn_rate = 0.1
features = 2 # input dimension
batch_size = 512
N_train = 2048 # number of training observations
layer_width = [20, 1] # width of layers
N_layers = len(layer_width) # number of hidden layers
random_seed = 1337
initial = 1
""" Generate data """
np.random.seed(random_seed)
x = np.random.uniform(0,1,size=(features, N_train)) # each columns is one observations, the row represent features
y = flib.circles_classification(x)
""" Plot Result """
# parameter to optimize
learn_rate = [1, 0.8, 0.6, 0.5]
plt.figure()
func = flib.SGD
loss_value = [None]*len(learn_rate)
for i in range(len(learn_rate)):
    weights, biases, loss_list, time_list, gradient_list = func(x, y, layer_width, batch_size, learn_rate[i], iterations, initial, random_seed)
    loss_value[i]= flib.calculate_loss(weights, biases)
    flib.plot_loss_epoch(loss_list, iterations//len(loss_list), str(learn_rate[i]))

print(loss_value)
plt.legend()
plt.show()
