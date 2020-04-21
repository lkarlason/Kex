import numpy as np
import matplotlib.pyplot as plt
import functionlib as flib
""" Hyperparameters """
""" SGD, ADAM, CG, L_BFGS"""
iterations = 500
learn_rate = 1
features = 2 # input dimension
batch_size = 64 
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
batch_size = [128, 256, 512]
parameter = batch_size
loss_value = [None]*len(parameter)
func = flib.CG
plt.figure()
for i in range(len(parameter)):
    
    for i in range(5):
    weights, biases, loss_list, time_list, gradient_list = func(x, y, layer_width, batch_size[i], learn_rate, iterations, initial, random_seed)
    loss_value[i]= flib.calculate_loss(weights, biases)
    flib.plot_loss_time(loss_list, time_list, str(parameter[i]))
    #flib.plot_loss_epoch(loss_list, iterations//len(loss_list), str(parameter[i]))

print(loss_value)
plt.legend()
plt.show()
