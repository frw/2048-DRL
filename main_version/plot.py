################################################
THEFILENAME = 'baseline_775_test.pkl.gz'########
################################################

import numpy as np
import pylab as pl
import cPickle as pickle
import gzip
import os

def plot_scores(scores, run_name):
    indices = np.arange(1, len(scores) + 1)
    moving_average = np.convolve(scores, np.repeat(1.0, 100) / 100, 'valid')
    print moving_average[-1]

    pl.plot(indices, scores, '-')
    pl.plot(indices[99:], moving_average, 'r--')
    pl.title("Game Scores Over Time: " + run_name)
    pl.ylabel("Score")
    pl.xlabel("Iteration")
    pl.show()

def scoring_statistics(scores):
    print "Mean score across last 1000 iterations:"
    print np.mean(scores[-1000:])
    print "Mean score across all iterations:"
    print np.mean(scores)
    print "Percent over 1000 in last 1000 iterations:"
    print sum([1 if score > 1000 else 0 for score in scores[-1000:]]) / 10.0


def plot_select_weights(weights, saving_multiple, weight_list, run_name):
    '''
    Plots weights over time for specified set of weights, each in a separate plot.
    Note: this function does not plot bias weights.
    '''
    indices = np.arange(1, len(weights) + 1) * saving_multiple

    #weight_list properties:
    #1st element = which layer, 2nd element = preceding layer node #, 3rd element = next layer node #

    for elem in weight_list:
        pl.figure()
        formatted_weights = []
        for i in range(len(weights)):
            formatted_weights.append(weights[i][elem[0]][elem[1],elem[2]])
        pl.plot(indices, formatted_weights)
        pl.title("Select Weights over Time: " + run_name)
        pl.ylabel("Weight Value")
        pl.xlabel("Iteration")
        pl.show()

def plot_select_bias_weights(weights, saving_multiple, weight_list, run_name):
    '''
    Plots weights over time for specified set of weights, each in a separate plot.
    Note: this function only plots bias weights.
    '''

    #weight_list properties:
    #1st element = which layer, 2nd element = next layer node #

    indices = np.arange(1, len(weights) + 1) * saving_multiple

    for elem in weight_list:
        pl.figure()
        formatted_weights = []
        for i in range(len(weights)):
            formatted_weights.append(weights[i][elem[0]][elem[1]])
        pl.plot(indices, formatted_weights)
        pl.title("Select Bias Weights over Time: " + run_name)
        pl.ylabel("Weight Value")
        pl.xlabel("Iteration")
        pl.show()

def plot_random_weights_true(weights, number_per_layer, saving_multiple, which_layer, run_name):
    '''
    Randomly draws weights according to number_per_layer and plots each over time, all in the same plot.
    Note: only draws from general inter-node weights, not bias weights.
    which_layer allows for all weights to be specific layer or in all layers
    '''
    pl.figure()
    indices = np.arange(1, len(weights) + 1) * saving_multiple
    num_weight_layers = len(weights[0]) / 2
    layer_list = []
    if which_layer:
        layer_list = [which_layer]
    else:
        layer_list = range(num_weight_layers)
    for layer in layer_list:
        dim1 = weights[0][layer*2].shape[0]
        dim2 = weights[0][layer*2].shape[1]
        weight_list = []
        dim1_samples = np.random.randint(0,dim1,size=number_per_layer)
        dim2_samples = np.random.randint(0,dim2,size=number_per_layer)
        for a in range(number_per_layer):
            formatted_weights = []
            for i in range(len(weights)):
                formatted_weights.append(weights[i][layer*2][dim1_samples[a],dim2_samples[a]])
            pl.plot(indices, formatted_weights)

    pl.title("Random Set of Weight Values Over Time: " + run_name)
    pl.ylabel("Weight Value")
    pl.xlabel("Iteration")
    pl.show()


def plot_random_weights_nice(weights, number_per_layer, saving_multiple, which_layer, run_name):
    '''
    Randomly draws weights according to number_per_layer and plots each over time, all in the same plot.
    Scales weight values to be between -1 and 1 on the y-axis, to make graph easier to read.
    Note: only draws from general inter-node weights, not bias weights.
    which_layer allows for all weights to be specific layer or in all layers
    '''
    pl.figure()
    indices = np.arange(1, len(weights) + 1) * saving_multiple
    num_weight_layers = len(weights[0]) / 2
    layer_list = []
    if which_layer:
        layer_list = [which_layer]
    else:
        layer_list = range(num_weight_layers)
    for layer in layer_list:
        dim1 = weights[0][layer*2].shape[0]
        dim2 = weights[0][layer*2].shape[1]
        weight_list = []
        dim1_samples = np.random.randint(0,dim1,size=number_per_layer)
        dim2_samples = np.random.randint(0,dim2,size=number_per_layer)
        for a in range(number_per_layer):
            formatted_weights = []
            for i in range(len(weights)):
                formatted_weights.append(weights[i][layer*2][dim1_samples[a],dim2_samples[a]])
            max_value = max(formatted_weights)
            min_value = min(formatted_weights)
            scaled_weights = formatted_weights / (max(abs(max_value),abs(min_value)))
            pl.plot(indices, scaled_weights)

    pl.title("Random Set of Scaled Weight Values Over Time: " + run_name)
    pl.ylabel("Relative Weight Value (Y-axis Differs Between Weights)")
    pl.xlabel("Iteration")
    pl.show()

def graph(learner, run_name):
    scoring_statistics(learner.scores)

    plot_scores(learner.scores, run_name)

    #How learner.weights is formatted: Top level is a list across iterations. Each element of this list is a list of numpy arrays.
    #The numpy arrays contain the individual weight values for different layers. The order of the numpy arrays is:
    #Layer #1 Inter-node weights, Layer #1 Bias weights, Layer #2 Inter-node weights, Layer #2 Bias weights, etc.

    #general_weight_list = [[0,0,0], [0,0,1], [0,10,0], [0,10,1], [0,16,0], [0,16,1], [0,17,0], [0,17,1], [2,0,0], [2,1,0]]
    #plot_select_weights(learner.weights, 1000, general_weight_list, run_name)

    #bias_weight_list = [[1,0], [1,1], [3,0]]
    #plot_select_bias_weights(learner.weights, 1000, bias_weight_list, run_name)

    plot_random_weights_true(learner.weights,10, 1000, 0, run_name)

    plot_random_weights_nice(learner.weights,10, 1000, 0, run_name)

    plot_random_weights_true(learner.weights,10, 1000, 1, run_name)

    plot_random_weights_nice(learner.weights,10, 1000, 1, run_name)

def get_results(filename):
    if os.path.isfile(filename):
        with gzip.open(filename, 'rb') as infile:
            learner = pickle.load(infile)

        graph(learner, filename[:(-7)])

    else:
        print('Cannot find file!')

get_results(THEFILENAME)
