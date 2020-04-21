import numpy as np
import matplotlib.pyplot as plt
import functionlib as flib
""" Hyperparameters """
""" SGD, ADAM, CG, L_BFGS"""
iterations = [700, 700, 700, 700]
learn_rate = [0.9, 0.07, 0.1, 0.05]
N_train = 2048 # number of training observations
batch_size = [64, 128, 256, N_train]
features = 2 # input dimension
layer_width = [20, 1] # width of layers
N_layers = len(layer_width) # number of hidden layers
N_seeds = 10
random_seed = 1337*10
initial = 1
""" Generate data """
np.random.seed(random_seed)
x = np.random.uniform(0,1,size=(features, N_train)) # each columns is one observations, the row represent features
y = flib.circles_classification(x)

""" Plot Result """
plt.figure()
loss_value = np.zeros(4)
reps = [1, 1, 3, 20]
func = [flib.SGD, flib.ADAM, flib.CG, flib.L_BFGS]
name = ['SGD', 'ADAM', 'CG', 'L-BFGS']
for i in range(4):
    print("method ",i)
    loss_mean = np.zeros((N_seeds, iterations[i]//reps[i]))
    time_mean = np.zeros((N_seeds, iterations[i]//reps[i]))
    for j in range(N_seeds):
        print("sample ",j)
        weights, biases, loss_mean[j,:], time_mean[j,:], gradient_list = func[i](x, y, layer_width, batch_size[i], learn_rate[i], iterations[i], initial, random_seed*(j+1))

    loss_value[i]= flib.calculate_loss(weights, biases)
    #flib.plot_loss_time(np.mean(loss_mean, axis=0), np.mean(time_mean, axis=0), name[i])
    flib.plot_loss_epoch(np.mean(loss_mean, axis=0), reps[i], name[i])
    
plt.legend()
plt.show()
