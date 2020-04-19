import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.special import gamma
import math
import time

def circle_classification(x):
    """ Generates labels for points inside and outside a circle. """
    # Circle parameters (x-c_1)^2 + (y-c_2)^2 = r^2
    c = 0.5
    N = x.shape[1]
    features = x.shape[0]
    r = N_radius(features, 1)
    y = np.zeros((N,1))

    for i in range(N):
        rad_sum = 0
        for j in range(features):
            rad_sum += (x[j,i]-c)**2
        if rad_sum > r**2:
            y[i,0] = 1
    return y

def circles_classification(x):
    """ Generates labels for points inside and outside the circles. """
    N = x.shape[1]
    features = x.shape[0]
    circles = 2**features
    c = 0.25
    r = N_radius(features, circles)
    y = np.zeros((N,1))
    for i in range(N):
        rad = 0
        for j in range(features):
            if x[j,i] > 0.5:
                c = 0.75
            else: 
                c = 0.25
            rad += (x[j,i]-c)**2
        if rad > r**2:
            y[i,0] = 1
    return y

def N_radius(features, circles):
    """ Calculate the radius in n dimensional space """
    rad = (gamma([features/2+1])/(2*circles*np.power(np.pi,(features/2))))**(1/features)
    return rad[0]

def plot_training(x,y,circles):
    """ Plots training data. """
    plt.figure()
    if x.shape[0] != 2:
        return
    if circles == 4:
        plot_circles()
    else:
        plot_circle()
    plt.plot(x[0,y[:,0]==0],x[1,y[:,0]==0],'r*') # Plot all points with label 0 as red
    plt.plot(x[0,y[:,0]==1],x[1,y[:,0]==1],'bo') # Plot all points with label 1 as blue
    plt.ylim(0, 1)
    plt.xlim(0, 1)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show

def plot_circle():
    """ Plots the circle. """
    c = 0.5
    r = 0.39894
    theta = np.linspace(0,np.pi*2)
    x_c = c + r*np.cos(theta)
    y_c = c + r*np.sin(theta)
    plt.plot(x_c,y_c,'k')

def plot_circles():
    """ Plots the circles. """
    c = 0.25
    r = 0.19947
    theta = np.linspace(0,np.pi*2)
    x_c_1 = c + r*np.cos(theta)
    y_c_1 = c + r*np.sin(theta)
    x_c_2 = 1-c + r*np.cos(theta)
    y_c_2 = 1-c + r*np.sin(theta)
    x_c_3 = c + r*np.cos(theta)
    y_c_3 = 1-c + r*np.sin(theta)
    x_c_4 = 1-c + r*np.cos(theta)
    y_c_4 = c + r*np.sin(theta)
    plt.plot(x_c_1,y_c_1,'k')
    plt.plot(x_c_2,y_c_2,'k')
    plt.plot(x_c_3,y_c_3,'k')
    plt.plot(x_c_4,y_c_4,'k')

def plot_results(weights, biases):
    """ Only works if there are two features"""
    N = 10000
    if weights[0].shape[1] != 2: # if feature dim isn't 2
        return
    r = 0.39894
    x = np.random.uniform(0,1, size = (2,N))
    z, a = feedforward(weights, biases, x)
    activation = a[-1][0]
    plt.ylim(0, 1)
    plt.xlim(0, 1)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.plot(x[0,activation < 0.5],x[1,activation < 0.5],'ro')
    plt.plot(x[0,activation >= 0.5],x[1,activation >= 0.5],'bo')
    plot_circles()
    plt.show

def calculate_loss(weights, biases):
    feature =  weights[0].shape[1]
    N = 5000*feature
    x = np.random.uniform(0,1, size = (feature, N))
    z, a = feedforward(weights, biases, x)
    activation = a[-1][0]
    y = circles_classification(x)
    loss = 0
    for i in range(N):
        if (activation[i] > 0.5 and y[i] ==0) or (activation[i] < 0.5 and y[i]==1):
            loss +=1
    return loss/N

def plot_loss_epoch(loss_list):
    """ Plot loss over iterations """
    plt.title("Loss function")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(loss_list)

def plot_loss_epoch(loss_list, epoch, name):
    """ Plot loss over iterations """
    plt.title("Loss function")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot([x for x in range(0, len(loss_list)*epoch, epoch)],loss_list, label= name)

def param_vector(x, layer_width, param_size, random_seed):
    parameters = 0
    for i in range(len(layer_width)):
        if i==0:
            parameters += layer_width[i]*x.shape[0]
        else:
            parameters += layer_width[i]*layer_width[i-1]
        parameters += layer_width[i]
    np.random.seed(random_seed)
    return np.random.normal(0, param_size, size=parameters)

def vector_to_matrix(x, layer_width, vector):
    weights = []
    biases = []
    for i in range(len(layer_width)):
        if i==0:
            index = layer_width[i]*x.shape[0]
            weights.append(np.reshape(vector[0:index].copy(), (layer_width[i], x.shape[0])))
        else:
            prev_index = index
            index += layer_width[i]*layer_width[i-1]
            weights.append(np.reshape(vector[prev_index:index].copy(), (layer_width[i], layer_width[i-1])))
        prev_index = index
        index += layer_width[i]
        biases.append(np.reshape(vector[prev_index:index].copy(), (layer_width[i],1)))
    return weights, biases

def matrix_to_vector(vector, matrix_w, matrix_b):
    vector = vector.copy()
    for i in range(len(matrix_w)):
        if i==0:
            index = matrix_w[i].size
            vector[0:index] = (matrix_w[i].flatten()).copy()
        else:
            prev_index = index
            index += matrix_w[i].size
            vector[prev_index:index] = (matrix_w[i].flatten()).copy()
        prev_index = index
        index += matrix_b[i].size   
        vector[prev_index:index] = (matrix_b[i].flatten()).copy()
    return vector

def make_batch(x, y, batch_size):
    """ Creates minibatches from observations """
    N = x.shape[1]
    N_batches = N//batch_size
    x_batch = []
    y_batch = []
    for i in range(N_batches):
        x_batch.append(x[:,i*batch_size:(i+1)*batch_size])
        y_batch.append(y[i*batch_size:(i+1)*batch_size])
    return x_batch, y_batch, N_batches

def rectifier(x):
    y = np.copy(x)
    y[y < 0] = 0
    return y

def rectifier_prime(x):
    y = np.copy(x)
    y[y < 0] = 0
    y[y > 0] = 1
    return y

def sigmoid(x):
    """ Sigmoid function. """
    return 1/(1+np.exp(-x))

def sigmoid_prime(x):
    """ Derivative of the sigmoid function. """
    return sigmoid(x)*(1-sigmoid(x))

def tanh_prime(x):
    """ Derivative of the tanh function. """
    return 1-np.tanh(x)**2

def feedforward(W,b,a_0):
    """ Returns the output from the neural network and
        the input/ouput to/from each layer. """
    a = [0]*(len(W)+1)
    z = [0]*len(W)
    a[0] = a_0
    for i in range(len(W)):
        z[i] = np.matmul(W[i],a[i]) + b[i]
        if(i == len(W)-1): 
            a[i+1] = sigmoid(z[i])
        else:
            a[i+1] = np.tanh(z[i]) # rectifier(z[i])
    return z, a

def delta_out(a,y,z):
    """ Returns the error in the last layer. """
    return (-(y/(a+math.pow(10,-9))) + (1-y)/(1-a+math.pow(10,-9)))*sigmoid_prime(z)

def delta_n(W,d,z,l):
    """ Returns the error in the nth layer. """
    return np.matmul(W.T,d)*tanh_prime(z) #rectifier_prime(z) 
    
def backpropagate(w,z,a,y,l):
    """ Returns an array of the errors in all layers. """
    d = [0]*l

    d[l-1] = delta_out(a[-1],y.T,z[-1])
    for i in range(l-2,-1,-1):
        d[i] = delta_n(w[i+1],d[i+1],z[i],l)
    return d

def gradient(a,d,l,m):
    """ Returns the gradient w.r.t. the weights and
        the biases. """
    grad_W = [0]*l
    grad_b = [0]*l

    for i in range(l-1,-1,-1):
        grad_W[i] = np.matmul(d[i],a[i].T)/m
        grad_b[i] = np.sum(d[i], axis=1, keepdims = True)/m

    return grad_W, grad_b

def loss_function(y,a):
    """ Returns the value of the loss function. """
    return -np.sum(y.T*np.log(a+10**-10)+(1-y.T)*np.log(1-a+10**-10))/y.size

def SGD(x, y, layer_width, batch_size, learn_rate, iterations, initial, random_seed):
    """ Stocastic gradient decent """
    N_layers = len(layer_width)
    loss_list = [] # list for plotting loss over iterations
    time_list = []
    gradient_list = []
    batch_gradient_list = []
    t = 0 # iteration counter
    epoch = 0
    n = 0 # batch counter

    x_batch, y_batch, N_batches = make_batch(x, y, batch_size) # creates batches
    weight_vector = param_vector(x, layer_width, initial, random_seed)
    weights, biases = vector_to_matrix(x, layer_width, weight_vector)
    gradient_vector = param_vector(x, layer_width, 0, random_seed)

    start_time = time.time()
    while epoch < iterations:
        t += 1
        # calculate gradient
        z, a = feedforward(weights, biases, x_batch[n]) # node parameters for batch n
        d = backpropagate(weights, z, a, y_batch[n], N_layers) # deltas in network
        grad_W, grad_b = gradient(a, d, N_layers,batch_size) # generate gradient
        gradient_vector = matrix_to_vector(gradient_vector, grad_W, grad_b)
        # gradient decent
        weight_vector -= learn_rate * gradient_vector
        weights, biases = vector_to_matrix(x, layer_width, weight_vector)
        # measure time and next batch
        batch_gradient_list.append(linalg.norm(gradient_vector)**2)
        n +=1
        if n == N_batches: 
            z, a = feedforward(weights, biases, x) 
            loss_list.append(loss_function(y, a[-1]))
            time_list.append(time.time()-start_time)
            gradient_list.append(np.mean(batch_gradient_list))
            batch_gradient_list = []
            epoch += 1
            n = 0 # return to batch 1
            print("epoch = ", epoch, "/",iterations)
    return weights, biases, loss_list, time_list, gradient_list

def ADAM(x, y, layer_width, batch_size, learn_rate, iterations, initial, random_seed):
    """ Stocastic gradient decent with ADAM"""
    N_layers = len(layer_width)
    loss_list = [] # list for plotting loss over iterations
    time_list = []
    gradient_list = []
    batch_gradient_list = []
    t = 0 # iteration counter
    n = 0 # batch counter
    epoch = 0

    x_batch, y_batch, N_batches = make_batch(x, y, batch_size) # creates batches
    weight_vector = param_vector(x, layer_width, initial, random_seed)
    weights, biases = vector_to_matrix(x, layer_width, weight_vector)
    gradient_vector = param_vector(x, layer_width, 0, random_seed)
    s = param_vector(x, layer_width, 0, random_seed)    
    r = param_vector(x, layer_width, 0, random_seed)
    p1 = 0.9
    p2 = 0.999

    start_time = time.time()
    convergence_time = 0
    while epoch < iterations:
        """
        if loss_list[-1] < 0.2:
            end = time.time()
            convergence_time = end-start_time
        """
        t +=1
        # calculate gradient
        z, a = feedforward(weights, biases, x_batch[n]) # node parameters for batch n
        d = backpropagate(weights, z, a, y_batch[n], N_layers) # deltas in network
        grad_W, grad_b = gradient(a, d, N_layers, batch_size) # generate gradient
        gradient_vector = matrix_to_vector(gradient_vector, grad_W, grad_b)
        # ADAM
        s = p1*s + (1-p1)*gradient_vector
        r = p2*r + (1-p2)*gradient_vector**2
        s_hat = s/(1-p1**t)
        r_hat = r/(1-p2**t)
        weight_vector -= s_hat/(np.sqrt(r_hat)+10**(-10))*learn_rate
        weights, biases = vector_to_matrix(x, layer_width, weight_vector)
        # measure loss
        batch_gradient_list.append(linalg.norm(gradient_vector)**2)
        n +=1
        if n == N_batches: 
            z, a = feedforward(weights, biases, x) 
            loss_list.append(loss_function(y, a[-1]))
            time_list.append(time.time()-start_time)
            gradient_list.append(np.mean(batch_gradient_list))
            batch_gradient_list = []
            epoch += 1
            n = 0 # return to batch 1
            print("epoch = ", epoch, "/",iterations)

    if convergence_time == 0:
        convergence_time = time.time()-start_time
    return weights, biases, loss_list, time_list, gradient_list

def CG(x, y, layer_width, batch_size, learn_rate, iterations, initial, random_seed):
    """ Conjugate gradient decent """
    N_layers = len(layer_width)
    loss_list = [] # list for plotting loss over iterations
    time_list = []
    gradient_list = []
    batch_gradient_list = []
    t = 0 # iteration counter
    n = 0 # batch counter
    epoch = 0
    reps = 4
    
    x_batch, y_batch, N_batches = make_batch(x, y, batch_size) # creates batches
    weight_vector = param_vector(x, layer_width, initial, random_seed)
    weights, biases = vector_to_matrix(x, layer_width, weight_vector)
    start_time = time.time()
    while epoch < iterations:
        rho = param_vector(x, layer_width, 0, random_seed)
        gradient_vector = param_vector(x, layer_width, 0, random_seed)

        for j in range(reps):
            t +=1
            # calculate gradient
            z, a = feedforward(weights, biases, x_batch[n]) # node parameters for batch n
            d = backpropagate(weights, z, a, y_batch[n], N_layers) # deltas in network
            grad_W, grad_b = gradient(a, d, N_layers, batch_size) # generate gradient
            old_grad = gradient_vector.copy()
            gradient_vector = matrix_to_vector(gradient_vector, grad_W, grad_b)
            # Compute search direction
            beta = np.dot((gradient_vector-old_grad),gradient_vector)/(np.dot(old_grad,old_grad)+10**-10)
            rho = -gradient_vector+beta*rho
            step = [0.0] # Perform line search to find step size
            ret = minimize(fun= objective, x0= step, args=(weight_vector, rho, layer_width, x_batch[n], y_batch[n]), bounds = [(0, 50)])
            step = ret['x'] 
            # Update parameters
            weight_vector += rho*step[0]
            weights, biases = vector_to_matrix(x, layer_width, weight_vector)
            batch_gradient_list.append(linalg.norm(gradient_vector)**2)
        # measure loss
        n +=1
        if n == N_batches: 
            z, a = feedforward(weights, biases, x) 
            loss_list.append(loss_function(y, a[-1]))
            time_list.append(time.time()-start_time)
            gradient_list.append(np.mean(batch_gradient_list))
            batch_gradient_list = []
            epoch += reps
            n = 0 # return to batch 1
            print("epoch = ", epoch, "/",iterations)

    return weights, biases, loss_list, time_list, gradient_list

def objective(step, weight, rho, layer_width, x, y):  
    weight_vector = weight.copy()
    weight_vector += rho*step[0]
    weights, biases = vector_to_matrix(x, layer_width, weight_vector)
    z, a = feedforward(weights, biases, x)
    return loss_function(y, a[-1])

def update_list(history, ele, memory):
    history.insert(0, ele)
    if len(history) > memory:
                history.pop()
    return history

def L_BFGS(x, y, layer_width, batch_size, learn_rate, iterations, initial, random_seed):
    """ Limited-memory Broyden–Fletcher–Goldfarb–Shanno. """
    N_layers = len(layer_width)
    loss_list = [] # list for plotting loss over iterations
    time_list = []
    gradient_list = []
    batch_gradient_list = []
    t = 0 # iteration counter
    n = 0 # batch counter
    epoch = 0 # epoch counter
    reps = 1 
    m = 20 # number of updates kept
    
    x_batch, y_batch, N_batches = make_batch(x, y, batch_size) # creates batches
    weight_vector = param_vector(x, layer_width, initial, random_seed) # create weights
    weights, biases = vector_to_matrix(x, layer_width, weight_vector)
    gradient_vector = param_vector(x, layer_width, 0, random_seed) # create gradients

    start_time = time.time()
    Y = [] #y# gradient step sizes
    S = [learn_rate*weight_vector.copy()] #s# weights step sizes
    ro = []
    while epoch < iterations:  
        
        for k in range(1,reps+1):
            t +=1
            z, a = feedforward(weights, biases, x) 
            loss_list.append(loss_function(y, a[-1]))
            print(loss_list[-1])
            # calculate gradient
            z, a = feedforward(weights, biases, x_batch[n]) # node parameters for batch n
            d = backpropagate(weights, z, a, y_batch[n], N_layers) # deltas in network
            old_grad = gradient_vector.copy()
            grad_W, grad_b = gradient(a, d, N_layers,batch_size) # generate gradient
            gradient_vector = matrix_to_vector(gradient_vector, grad_W, grad_b)
            #L-BFGS
            Y = update_list(Y, gradient_vector-old_grad, m)
            ro = update_list(ro, 1/(np.dot(S[0], Y[0])+10**-10), m)
            q = gradient_vector.copy()
            alpha = np.zeros(len(S))
            beta = np.zeros(len(S))
            for i in range(len(S)):
                alpha[i] = np.dot(S[i], q)*ro[i]
                q -= alpha[i]*Y[i]
            gamma = np.dot(S[0],Y[0])/(np.dot(Y[0],Y[0])+10**-10)
            q = gamma*q
            for i in range(len(S)-1, -1, -1):
                beta[i] = np.dot(Y[i], q)*ro[i]
                q += S[i]*(alpha[i]-beta[i])
            weight_vector -= q*learn_rate
            weights, biases = vector_to_matrix(x, layer_width, weight_vector)
            S = update_list(S, -q*learn_rate, m)
            batch_gradient_list.append(linalg.norm(gradient_vector.copy())**2)

        n +=1
        if n == N_batches: 
            z, a = feedforward(weights, biases, x) 
            loss_list.append(loss_function(y, a[-1]))
            print(loss_list[-1])
            time_list.append(time.time()-start_time)
            gradient_list.append(np.mean(batch_gradient_list))
            batch_gradient_list = []
            epoch += reps
            n = 0 # return to batch 1
            print("epoch = ", epoch, "/",iterations)
    
    return weights, biases, loss_list, time_list, gradient_list