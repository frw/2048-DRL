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

def plot_select_general_weights(weights, title):
    indices = np.arange(1, len(weights) + 1) #* 100

    #1st element = which layer, 2nd element = preceding layer node #, 3rd element = next layer node #
    weight_list = [[0,0,0], [0,0,1], [0,10,0], [0,10,1], [0,16,0], [0,16,1], [0,17,0], [0,17,1], [2,0,0], [2,1,0]]

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


def graph(filename):
    if os.path.isfile(filename):
        with gzip.open(filename, 'rb') as infile:
            learner = pickle.load(infile)
        print np.mean(learner.scores[-1000:])
        plot_scores(learner.scores, learner.__class__.__name__ + " Scores")
        plot_select_general_weights(learner.weights, learner.__class__.__name__ + " Single Weight")
    else:
        print('Cannot find file!')

graph('DeepQLearner.pkl.gz')
