import numpy as np
import pylab as pl
import cPickle as pickle
import gzip
import os


def plot(scores, title):
    indices = np.arange(1, len(scores) + 1)
    moving_average = np.convolve(scores, np.repeat(1.0, 100) / 100, 'valid')
    print moving_average[-1]

    pl.plot(indices, scores, '-')
    pl.plot(indices[99:], moving_average, 'r--')
    pl.title(title)
    pl.ylabel("Score")
    pl.xlabel("Iteration")
    pl.show()


def graph(filename):
    if os.path.isfile(filename):
        with gzip.open(filename, 'rb') as infile:
            learner = pickle.load(infile)
        print np.mean(learner.scores[-1000:])
        plot(learner.scores, learner.__class__.__name__ + " Scores")
    else:
        print('Cannot find file!')

graph('DeepQLearner.pkl.gz')
