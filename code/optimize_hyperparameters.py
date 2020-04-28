import numpy as np
import matplotlib.pyplot as plt
import functionlib as flib
""" Hyperparameters """
""" SGD, ADAM, CG, L_BFGS"""
N_train = 2048 # number of training observations
iterations = [1000, 1000, 500, 1500]
learn_rate = [0.9, 0.07, 1, 0.05]
batch_size = [64, 128, 256, N_train]
reps = [1, 1, 3, 1]
features = 2 # input dimension
layer_width = [100, 1] # width of layers
N_layers = len(layer_width) # number of hidden layers
random_seed = 1337
N_seeds = 3
initial = 1
""" Generate data """
np.random.seed(random_seed)
x = np.random.uniform(0,1,size=(features, N_train)) # each columns is one observations, the row represent features
y = flib.circles_classification(x)
""" Plot Result """
# parameter to optimize
learn_rate = [0.1, 0.075, 0.05] #values to test
parameter = learn_rate  #the parameter you want to change
real_loss = np.zeros((N_seeds , len(parameter)))
func = [flib.SGD, flib.ADAM, flib.CG, flib.L_BFGS]
name = ['SGD', 'ADAM', 'CG', 'L-BFGS']
plt.figure()
k = 3 # choose method
for i in range(len(parameter)):
    print("parameter ", parameter[i])
    loss_mean = np.zeros((N_seeds, iterations[k]//reps[k]))
    time_mean = np.zeros((N_seeds, iterations[k]//reps[k]))
    j = 0
    q = 0
    while j < N_seeds:
        print("sample ",j)
        weights, biases, loss_mean[j,:], time_mean[j,:], gradient_list = func[k](x, y, layer_width, batch_size[k], learn_rate[i], iterations[k], initial, random_seed*(j+1)+q)
        real_loss[j, i] = flib.calculate_loss(weights, biases)
        if flib.calculate_loss(weights, biases) < 0.4:
            j+=1
        elif q > 5:
            j = N_seeds
        else:
            q+=1
            print("q = ",q)
    flib.plot_loss_time(np.mean(loss_mean, axis=0), np.mean(time_mean, axis=0), "Batch size " + str(parameter[i]))
    #flib.plot_loss_epoch(np.mean(loss_mean, axis=0), reps[k], str(parameter[i]))
    print("q = ",q,"/",q+N_seeds)

real_loss= np.mean(real_loss, axis=0)
print(real_loss)
plt.title(name[k])
plt.legend()
plt.show()
