import numpy as np
import matplotlib.pyplot as plt
import functionlib as flib
""" Hyperparameters """
""" SGD, ADAM, CG, L_BFGS"""
N_train = 2048 # number of training observations
iterations = [1000, 1000, 1000, 1000]
learn_rate = [0.05, 0.004, 0.1, 0.003]
batch_size = [64, 128, 256, N_train]
reps = [1, 1, 3, 20]
features = 2 # input dimension
layer_width = [20, 20, 20, 1] # width of layers
N_layers = len(layer_width) # number of hidden layers
random_seed = 1337
N_seeds = 10
initial = 1
""" Generate data """
np.random.seed(random_seed)
x = np.random.uniform(0,1,size=(features, N_train)) # each columns is one observations, the row represent features
y = flib.circles_classification(x)
""" Plot Result """
plt.figure()
loss_value = np.zeros(4)
func = [flib.SGD, flib.ADAM, flib.CG, flib.L_BFGS]
name = ['SGD', 'ADAM', 'CG', 'L-BFGS']
for i in range(4):
    print("method ",i)
    loss_mean = np.zeros((N_seeds, iterations[i]//reps[i]))
    time_mean = np.zeros((N_seeds, iterations[i]//reps[i]))
    j = 0
    q = 0
    while j < N_seeds:
        print("sample ",j)
        weights, biases, loss_mean[j,:], time_mean[j,:], gradient_list = func[i](x, y, layer_width, batch_size[i], learn_rate[i], iterations[i], initial, random_seed*(j+1)+q)
        if flib.calculate_loss(weights, biases) < 0.3:
            j+=1
        else:
            q+=1
            print("q = ",q)

    loss_value[i]= flib.calculate_loss(weights, biases)
    #flib.plot_loss_time(np.mean(loss_mean, axis=0), np.mean(time_mean, axis=0), name[i])
    flib.plot_loss_epoch(np.mean(loss_mean, axis=0), reps[i], name[i])
    
print("q = ",q)
plt.legend()
plt.show()
