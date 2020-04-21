import numpy as np
import matplotlib.pyplot as plt
import functionlib as flib
""" Hyperparameters """
""" SGD, ADAM, CG, L_BFGS"""
iterations = 2000
learn_rate = 0.05
features = 2 # input dimension
batch_size = 2048 
N_train = 2048 # number of training observations
layer_width = [20, 1] # width of layers
N_layers = len(layer_width) # number of hidden l ayers
random_seed = 1337*10
initial = 1
""" Generate data """
np.random.seed(random_seed)
x = np.random.uniform(0,1,size=(features, N_train)) # each columns is one observations, the row represent features
y = flib.circles_classification(x)
""" Plot Result """
# parameter to optimize
batch_size = [512, 1024, 2048] #values to test
parameter = batch_size  #the parameter you want to change
loss_value = [0]*len(parameter)
func = flib.L_BFGS
plt.figure()
n = 5
for i in range(len(parameter)):
                                                                            # ad index to parameter you want to change
    weights, biases, loss_list, time_list, gradient_list = func(x, y, layer_width, batch_size[i], learn_rate, iterations, initial, random_seed)
    loss_value[i] += flib.calculate_loss(weights, biases)
    flib.plot_loss_time(loss_list, time_list, str(parameter[i]))
    #flib.plot_loss_epoch(loss_list, iterations//len(loss_list), str(parameter[i]))

print(loss_value)
plt.legend()
plt.show()
