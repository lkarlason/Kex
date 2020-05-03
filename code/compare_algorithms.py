import numpy as np
import matplotlib.pyplot as plt
import functionlib as flib
""" Hyperparameters """
""" SGD, ADAM, CG, L_BFGS"""
N_train = 8192 # number of training observations
iterations = [3000, 3000, 3000, 3500]
learn_rate = [0.05, 0.004, 1, 0.003]
batch_size = [128, 256, 256, N_train]
reps = [1, 1, 3, 1] # how many times the algorithm iterates 
                    # over one batch before switching batch
features = 4 # input dimension
layer_width = [20, 20, 20, 1] # width of layers
N_layers = len(layer_width) # number of hidden layers -1
random_seed = 1337
N_seeds = 5 # number of samples used when calculating average
initial = 0.5 # the size of the initial parameters in network
""" Generate data """
np.random.seed(random_seed)
x = np.random.uniform(0,1,size=(features, N_train)) # each columns is one observations, the row represent features
y = flib.circles_classification(x)
""" Calculate Result """
plt.figure()
real_loss = np.zeros((N_seeds , 4))
func = [flib.SGD, flib.ADAM, flib.CG, flib.L_BFGS]
name = ['SGD', 'ADAM', 'CG', 'L-BFGS']
for i in range(4):
    print("method ",i)
    loss_mean = np.zeros((N_seeds, iterations[i]//reps[i]))
    time_mean = np.zeros((N_seeds, iterations[i]//reps[i]))
    gradient_mean = np.zeros((N_seeds, iterations[i]//reps[i]))
    j = 0
    q = 0
    while j < N_seeds:
        print("sample ",j)
        weights, biases, loss_mean[j,:], time_mean[j,:], gradient_mean[j,:] = func[i](x, y, layer_width, batch_size[i], learn_rate[i], iterations[i], initial, random_seed*(j+1)+q)
        if flib.calculate_loss(weights, biases) < 0.4:
            real_loss[j, i] = flib.calculate_loss(weights, biases)
            j+=1
        else:
            q+=1
            print("q = ", q)
    
    print("q = ",q,"/",q+N_seeds)
    #flib.plot_loss_epoch(np.mean(loss_mean, axis=0), reps[i], name[i])
    flib.plot_loss_time(np.mean(loss_mean, axis=0), np.mean(time_mean, axis=0), name[i])
""" Plot Result """
print(np.mean(real_loss, axis=0))
plt.legend()
plt.show()
