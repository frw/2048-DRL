import numpy as np
import numpy.random as npr
import cPickle as pickle
import gzip
import os

from tui.board import Board

ACTIONS = [Board.UP, Board.DOWN, Board.LEFT, Board.RIGHT]

def draw_greedy(values):
    m = np.max(values)
    best = []
    for i, v in enumerate(values):
        if v == m:
            best.append(i)
    return npr.choice(best)


class BoltzmannExplorer(object):
    def __init__(self, tau=10000., decay=0.9995):
        self.tau = tau
        self.decay = decay

    def decide_action(self, epoch, values):
        if self.tau == 0:
            return draw_greedy(values)
        else:
            try:
                temperature = values / self.tau

                diff = 20 - np.max(temperature)
                if np.isinf(diff):
                    return draw_greedy(values)

                # make sure we keep the exponential bounded (between +20 and -20)
                temperature += diff
                if np.min(temperature) < -20:
                    for i, v in enumerate(temperature):
                        if v < -20:
                            temperature[i] = -20
                probabilities = np.exp(temperature)
                probabilities = np.divide(probabilities, np.sum(probabilities))

                s = np.sum(probabilities)
                if not s < 1.00001 or not s > 0.99999:
                    print(values, self.tau, temperature, probabilities, 1 - s)
                    raise ValueError()
                r = npr.random()
                s = 0
                for i, p in enumerate(probabilities):
                    s += p
                    if s > r:
                        return i
                return npr.randint(len(probabilities))
            finally:
                self.tau *= self.decay


class EpsilonGreedyExplorer(object):
    def __init__(self, epsilon=0.3, decay=0.99999):
        self.epsilon = epsilon
        self.decay = decay

    def decide_action(self, epoch, values):
        try:
            if npr.random() < self.epsilon:
                return npr.randint(len(values))
            else:
                return draw_greedy(values)
        finally:
            self.epsilon *= self.decay


class BaseLearner(object):
    def __init__(self, explorer=BoltzmannExplorer()):
        self.filename = self.__class__.__name__ + '.pkl.gz'
        self.explorer = explorer
        self.epoch = 0
        self.last_state = None
        self.last_action = None
        self.last_reward = None
        self.scores = []

    def new_epoch(self):
        self.epoch += 1
        self.last_state = None
        self.last_action = None
        self.last_reward = None

    def end_epoch(self, score):
        self.scores.append(score)

    def action_callback(self, raw_state, possible_moves):
        new_state = self.process_state(raw_state)
        new_action = self.decide_action(new_state, possible_moves)

        self.last_state = new_state
        self.last_action = new_action

        return new_action

    def process_state(self, raw_state):
        state = []
        for row in raw_state:
            for col in row:
                if col == 0:
                    state.append(0)
                else:
                    state.append(np.log2(col))
        return tuple(state)

    def decide_action(self, new_state, possible_moves):
        return npr.choice(ACTIONS)

    def reward_callback(self, reward):
        self.last_reward = reward

    def load(self):
        if os.path.isfile(self.filename):
            with gzip.open(self.filename, 'rb') as infile:
                return pickle.load(infile)

        return self

    def save(self):
        with gzip.open(self.filename, 'wb') as outfile:
            pickle.dump(self, outfile, pickle.HIGHEST_PROTOCOL)


class QLearner(BaseLearner):
    def __init__(self):
        super(QLearner, self).__init__()
        self.learning_rate = 0.3
        self.discount_rate = 0.95
        self.Q = {}

    def process_state(self, raw_state):
        new_state = super(QLearner, self).process_state(raw_state)

        if new_state not in self.Q:
            self.Q[new_state] = np.zeros(4)

        if self.last_state is not None:
            q = self.Q[self.last_state][self.last_action]
            self.Q[self.last_state][self.last_action] =\
                q + self.learning_rate * (self.last_reward + self.discount_rate * np.max(self.Q[new_state]) - q)

        return new_state

    def decide_action(self, new_state, possible_moves):
        if possible_moves is None:
            return None
        return possible_moves[self.explorer.decide_action(self.epoch, self.Q[new_state][possible_moves])]
