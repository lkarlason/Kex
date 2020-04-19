import numpy as np
import matplotlib.pyplot as plt
import functionlib as flib
""" Hyperparameters """
""" SGD, ADAM, CG, L_BFGS"""
iterations = [4000, 2000, 600, 500]
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

loss_value = [None]*4

if False:
    f = flib.L_BFGS
    weights, biases, loss_list, time_list, gradient_list = f(x, y, layer_width, batch_size, learn_rate, iterations[3], initial, random_seed)
    plt.plot(loss_list)
    plt.show()
else:
    """ Plot Result """
    plt.figure()
    func = [flib.SGD, flib.ADAM, flib.CG, flib.L_BFGS]
    name = ['SGD', 'ADAM', 'CG', 'L-BFGS']
    for i in range(3):
        weights, biases, loss_list, time_list, gradient_list = func[i](x, y, layer_width, batch_size, learn_rate, iterations[i], initial, random_seed)
        loss_value[i]= flib.calculate_loss(weights, biases)
        #flib.plot_loss_time(loss_list, time_list, name[i])
        flib.plot_loss_epoch(loss_list, iterations[i]//len(loss_list), name[i])
        
    plt.legend()
    plt.show()
