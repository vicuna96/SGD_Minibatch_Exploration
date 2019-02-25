import os
import numpy as np
from numpy import random
import scipy
import matplotlib
import mnist
import pickle
matplotlib.use('agg')
from matplotlib import pyplot

mnist_data_directory = os.path.join(os.path.dirname(__file__), "data")

# TODO add any additional imports and global variables


def load_MNIST_dataset():
    PICKLE_FILE = os.path.join(mnist_data_directory, "MNIST.pickle")
    try:
        dataset = pickle.load(open(PICKLE_FILE, 'rb'))
    except:
        # load the MNIST dataset
        mnist_data = mnist.MNIST(mnist_data_directory, return_type="numpy", gz=True)
        Xs_tr, Lbls_tr = mnist_data.load_training();
        Xs_tr = Xs_tr.transpose() / 255.0
        Ys_tr = np.zeros((10, 60000))
        for i in range(60000):
            Ys_tr[Lbls_tr[i], i] = 1.0  # one-hot encode each label
        Xs_tr = np.ascontiguousarray(Xs_tr)
        Ys_tr = np.ascontiguousarray(Ys_tr)
        Xs_te, Lbls_te = mnist_data.load_testing();
        Xs_te = Xs_te.transpose() / 255.0
        Ys_te = np.zeros((10, 10000))
        for i in range(10000):
            Ys_te[Lbls_te[i], i] = 1.0  # one-hot encode each label
        Xs_te = np.ascontiguousarray(Xs_te)
        Ys_te = np.ascontiguousarray(Ys_te)
        dataset = (Xs_tr, Ys_tr, Xs_te, Ys_te)
        pickle.dump(dataset, open(PICKLE_FILE, 'wb'))
    return dataset


# compute the gradient of the multinomial logistic regression objective, with regularization
#
# Xs        training examples (d * n)
# Ys        training labels   (c * n)
# ii        the list/vector of indexes of the training example to compute the gradient with respect to
# gamma     L2 regularization constant
# W         parameters        (c * d)
#
# returns   the aeverage gradient of the regularized loss of the examples in vector ii with respect to the model parameters
def multinomial_logreg_grad_i(Xs, Ys, ii, gamma, W):
    # TODO students should implement this
    Xs, Ys = Xs[:,ii], Ys[:,ii]
    ewx = np.exp(np.matmul(W,Xs))
    p = np.sum(ewx, axis=0)
    ans = 1 / Xs.shape[1] * np.matmul(-Ys+ 1/p * ewx, Xs.T) + gamma * W
    return ans


# compute the error of the classifier (SAME AS PROGRAMMING ASSIGNMENT 1)
#
# Xs        examples          (d * n)
# Ys        labels            (c * n)
# W         parameters        (c * d)
#
# returns   the model error as a percentage of incorrect labels
def multinomial_logreg_error(Xs, Ys, W):
    return np.mean(np.argmax(np.matmul(W, Xs), axis=0) != np.argmax(Ys, axis=0))


# ALGORITHM 1: run stochastic gradient descent on a multinomial logistic regression objective, with regularization
#
# Xs              training examples (d * n)
# Ys              training labels   (d * c)
# gamma           L2 regularization constant
# W0              the initial value of the parameters (c * d)
# alpha           step size/learning rate
# num_epochs      number of epochs (passes through the training set) to run
# monitor_period  how frequently, in terms of iterations (not epochs) to output the parameter vector
#
# returns         a list of model parameters, one every "monitor_period" iterations
def stochastic_gradient_descent(Xs, Ys, gamma, W0, alpha, num_epochs, monitor_period):
    # TODO students should implement this
    models = []
    for i in range(1,num_epochs*Xs.shape[1]+1):
        ii = [np.random.randint(Xs.shape[1])]
        W0 = W0 - alpha * multinomial_logreg_grad_i(Xs, Ys, ii, gamma, W0) - alpha * gamma * W0
        if i % monitor_period == 0:
            models.append(W0)
    return models


# ALGORITHM 2: run stochastic gradient descent with sequential sampling order
#
# Xs              training examples (d * n)
# Ys              training labels   (d * c)
# gamma           L2 regularization constant
# W0              the initial value of the parameters (c * d)
# alpha           step size/learning rate
# num_epochs      number of epochs (passes through the training set) to run
# monitor_period  how frequently, in terms of iterations (not epochs) to output the parameter vector
#
# returns         a list of model parameters, one every "monitor_period" iterations
def sgd_sequential_scan(Xs, Ys, gamma, W0, alpha, num_epochs, monitor_period):
    # TODO students should implement this
    n = Xs.shape[1]
    models = []
    for i in range(num_epochs):
        cur = i*n
        for j in range(n):
            W0 = W0 - alpha * multinomial_logreg_grad_i(Xs, Ys, [j], gamma, W0) - alpha * gamma * W0
            if (j+cur+1) % monitor_period == 0:
                models.append(W0)
    return models


# ALGORITHM 3: run stochastic gradient descent with minibatching
#
# Xs              training examples (d * n)
# Ys              training labels   (d * c)
# gamma           L2 regularization constant
# W0              the initial value of the parameters (c * d)
# alpha           step size/learning rate
# B               minibatch size
# num_epochs      number of epochs (passes through the training set) to run
# monitor_period  how frequently, in terms of batches (not epochs) to output the parameter vector
#
# returns         a list of model parameters, one every "monitor_period" batches
def sgd_minibatch(Xs, Ys, gamma, W0, alpha, B, num_epochs, monitor_period):
    # TODO students should implement this
    n = Xs.shape[1]
    models = []
    for i in range(1,num_epochs * n // B+1):
        ii = np.random.randint(n,size=B)
        W0 = W0 - alpha * multinomial_logreg_grad_i(Xs, Ys, ii, gamma, W0) - alpha * gamma * W0
        if i % monitor_period == 0:
            models.append(W0)
    return models


# ALGORITHM 4: run stochastic gradient descent with minibatching and sequential sampling order
#
# Xs              training examples (d * n)
# Ys              training labels   (d * c)
# gamma           L2 regularization constant
# W0              the initial value of the parameters (c * d)
# alpha           step size/learning rate
# B               minibatch size
# num_epochs      number of epochs (passes through the training set) to run
# monitor_period  how frequently, in terms of batches (not epochs) to output the parameter vector
#
# returns         a list of model parameters, one every "monitor_period" batches
def sgd_minibatch_sequential_scan(Xs, Ys, gamma, W0, alpha, B, num_epochs, monitor_period):
    # TODO students should implement this
    n = Xs.shape[1]
    models = []
    for i in range(num_epochs):
        cur = i*(n//B)
        for j in range(n // B):
            ii = np.arange(j*B,(j+1)*B)
            W0 = W0 - alpha * multinomial_logreg_grad_i(Xs, Ys, ii, gamma, W0) - alpha * gamma * W0
            if (j+cur+1) % monitor_period == 0:
                models.append(W0)
    return models
    

if __name__ == "__main__":
    (Xs_tr, Ys_tr, Xs_te, Ys_te) = load_MNIST_dataset()
    # TODO add code to produce figures
    import timeit
    gamma, alpha, alpha_m = 0.0001, 0.001, 0.05
    num_epochs, monitor_period, monitor_period_m = 10, 6000, 100
    batch_size = 60
    c,_ = Ys_tr.shape
    d,_ = Xs_tr.shape
    W0 = np.zeros((c,d))

    def get_error(Xs, Ys, models):
        return [multinomial_logreg_error(Xs, Ys, W) for W in models]

    sgd = lambda : stochastic_gradient_descent(Xs_tr, Ys_tr, gamma, W0, alpha, num_epochs, monitor_period)

    sgd_seq = lambda : sgd_sequential_scan(Xs_tr, Ys_tr, gamma, W0, alpha, num_epochs, monitor_period)

    sgd_mini = lambda :sgd_minibatch(Xs_tr, Ys_tr, gamma, W0, alpha_m, batch_size, num_epochs, monitor_period_m)

    sgd_mini_seq = lambda : sgd_minibatch_sequential_scan(Xs_tr, Ys_tr, gamma, W0, alpha_m, batch_size, num_epochs, monitor_period_m)

    algos = [sgd, sgd_seq, sgd_mini, sgd_mini_seq]
    names = ["SGD Random", "SGD Sequential", "SGD Minibatch", "SGD Minibatch Sequential"]
    models = []

    for algo, name in zip(algos, names):
        models.append(algo())
        print(name," done")

    # Get model errors
    model_error_tr = [get_error(Xs_tr, Ys_tr, model) for model in models]
    print("Errors for training set done")
    model_error_te = [get_error(Xs_te, Ys_te, model) for model in models]
    print("Errors for test set done")

    t = .1 * np.arange(len(models[0])) + .1

    ''' Plot the model error for the models, whose respective names are given by [names].
        Save the image by the name [title].png and inlude [title] in the figure title '''
    def plot_error(model_error, names, title):
        pyplot.figure(np.random.randint(1000))
        pyplot.xlabel('Epochs')
        pyplot.ylabel('Error')
        pyplot.title('MNIST '+title+' Error')
        pyplot.grid(True)
        for name, error in zip(names,model_error):
            pyplot.plot(t, error, label=name)
        pyplot.gca().legend()
        pyplot.savefig(title+'.png', bbox_inches='tight')

    # plot training error as a function of epochs
    plot_error(model_error_tr, names, "Training")
    # plot test error as a function of epochs
    plot_error(model_error_te, names, "Test")

    ''' Time the algorightms in [algos], whose respective names are given by [names],
        by averaging the runtime of the algorithm over 5 runs. 
        PreC :The algorithms must be lambdas that take no inputs '''
    def time_algos(names, algos):
        times = []
        for name, algo in zip(names,algos):
            time = 0
            for _ in range(5):
                time -= timeit.default_timer()
                _ = algo()
                time += timeit.default_timer()
            times.append(time/5)
            print(name,time/5,"s")
        return times

    # Make plots for the average runtimes
    times = time_algos(names, algos)
    x_positions = np.arange(len(names))

    # plot runtime for training as a bar graph
    pyplot.figure(3)
    pyplot.bar(x_positions, times, align='center', alpha=0.5)
    pyplot.xticks(x_positions, [name[4:] for name in names])
    pyplot.ylabel('Average runtime (per model)')
    pyplot.xlabel('Models')
    pyplot.title('Runtime of Training - SGD')
    for i, v in enumerate(times):
        pyplot.text(i-.25, v * (1.015), " " + str(round(v,2)), color='black', va='center', fontweight='bold')
    pyplot.savefig('train_time.png', bbox_inches='tight')