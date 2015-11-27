import numpy as np
import pylab as pl
import cPickle as pickle
import gzip
import os


def plot_scores(scores, title):
    indices = np.arange(1, len(scores) + 1)
    moving_average = np.convolve(scores, np.repeat(1.0, 100) / 100, 'valid')
    print moving_average[-1]

    pl.plot(indices, scores, '-')
    pl.plot(indices[99:], moving_average, 'r--')
    pl.title(title)
    pl.ylabel("Score")
    pl.xlabel("Iteration")
    pl.show()

def scoring_statistics(scores):
    print "Mean score across all iterations:"
    print np.mean(scores[-1000:])


def plot_select_weights(weights, title, saving_multiple, weight_list):
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
        pl.title(title)
        pl.ylabel("Weight Value")
        pl.xlabel("Iteration")
        pl.show()

def plot_random_weights(weights, number_per_layer, saving_multiple):
    '''
    Randomly draws weights according to number_per_layer and plots each over time, all in the same plot.
    Note: only draws from general inter-node weights, not bias weights.
    '''

    pl.figure()
    indices = np.arange(1, len(weights) + 1) * saving_multiple
    num_weight_layers = len(weights[0]) / 2
    for layer in range(num_weight_layers):
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

    pl.title("Random Set of Weight Values Over Time")
    pl.ylabel("Weight Value")
    pl.xlabel("Iteration")
    pl.show()


def plot_random_weights_nice(weights, number_per_layer, saving_multiple):
    pl.figure()
    indices = np.arange(1, len(weights) + 1) * saving_multiple
    num_weight_layers = len(weights[0]) / 2
    for layer in range(num_weight_layers):
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

    pl.title("Random Set of Weight Values Over Time")
    pl.ylabel("Relative Weight Value (y-axis scaled per weight)")
    pl.xlabel("Iteration")
    pl.show()

def graph(learner):
    #plot_scores(learner.scores, learner.__class__.__name__ + " Scores")

    scoring_statistics(learner.scores)

    #weight_list = [[0,0,0], [0,0,1], [0,10,0], [0,10,1], [0,16,0], [0,16,1], [0,17,0], [0,17,1], [2,0,0], [2,1,0]]
    #plot_select_weights(learner.weights, learner.__class__.__name__ + " Single Weight", 100, weight_list)
    plot_random_weights(learner.weights,3, 100)

    plot_random_weights_nice(learner.weights,3, 100)

def get_results(filename):
    if os.path.isfile(filename):
        with gzip.open(filename, 'rb') as infile:
            learner = pickle.load(infile)

        graph(learner)

    else:
        print('Cannot find file!')

get_results('BasicDeepQLearner.pkl.gz')
