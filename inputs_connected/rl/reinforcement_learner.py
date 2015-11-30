import cPickle as pickle
import gzip
import os

import numpy as np
import numpy.random as npr

from rl.neural_network import QNetwork
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

        if possible_moves is None:
            return None

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


class Greedy(BaseLearner):
    def process_state(self, raw_state):
        return raw_state

    def decide_action(self, new_state, possible_moves):
        if len(possible_moves) == 1:
            return possible_moves[0]

        best_move = None
        for move in possible_moves:
            score_inc = 0
            if move == Board.UP or move == Board.DOWN:
                for i in range(4):
                    prev = None
                    for j in range(4):
                        c = new_state[j][i]
                        if c != 0:
                            if prev is None:
                                prev = c
                            elif c == prev:
                                score_inc += c * 2
                                prev = None
            else:
                for i in range(4):
                    prev = None
                    for j in range(4):
                        c = new_state[i][j]
                        if c != 0:
                            if prev is None:
                                prev = c
                            elif c == prev:
                                score_inc += c * 2
                                prev = None

            if best_move is None or score_inc > best_move[1]:
                best_move = (move, score_inc)

        return best_move[0]


class StandardQLearner(BaseLearner):
    def __init__(self):
        super(StandardQLearner, self).__init__()
        self.learning_rate = 0.3
        self.discount_rate = 0.95
        self.Q = {}

    def process_state(self, raw_state):
        new_state = super(StandardQLearner, self).process_state(raw_state)

        if new_state not in self.Q:
            self.Q[new_state] = np.zeros(4)

        if self.last_state is not None:
            q = self.Q[self.last_state][self.last_action]
            self.Q[self.last_state][self.last_action] =\
                q + self.learning_rate * (self.last_reward + self.discount_rate * np.max(self.Q[new_state]) - q)

        return new_state

    def decide_action(self, new_state, possible_moves):
        return possible_moves[self.explorer.decide_action(self.epoch, self.Q[new_state][possible_moves])]


class DeepQLearner (BaseLearner):
    def __init__(self):
        super(DeepQLearner, self).__init__()
        self.discount_rate = 0.95
        self.network = QNetwork()
        self.weights = []

    def decide_action(self, new_state, possible_moves):
        if not self.last_state:
           return npr.choice(possible_moves)

        valid_rotations = self.get_rotated_boards(new_state,possible_moves)

        # compute Q-scores with forward-propagation
        list_Qscore = []
        for some_state in valid_rotations:
            list_Qscore.append(self.network.use_model(some_state))

        # get the best action & Q-score from current state
        best_Qscore = max(list_Qscore)
        # best_move = possible_moves[list_Qscore.index(best_Qscore)]

        # update weights with back-propagation
        self.network.update_model((self.get_rotated_boards(self.last_state, [self.last_action])[0]), float(self.last_reward + self.discount_rate * best_Qscore))

        return possible_moves[self.explorer.decide_action(self.epoch, np.asarray(list_Qscore))]

    def end_epoch(self, score):
        super(DeepQLearner, self).end_epoch(score)

        #save the network weights at this epoch
        if self.epoch % 1000 == 0:
            self.weights.append(self.network.get_all_weights())

    def get_rotated_boards(self, this_state, which_rotations):
        #for which_rotations:
        #1 gives 180 degree turn
        #2 gives 90 degree turn clockwise (possible_move left for self.last_action for original state!!!)
        #3 gives 90 degree turn CCW

        #returns list of processed tuples

        rotated_boards = []

        for move in which_rotations:
            if move == 0:
                rotated_boards.append(this_state)
            elif move == 1:
                new_board = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
                map_dictionary = [15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0]
                for i in range(16):
                    new_board[i] = this_state[map_dictionary[i]]
                rotated_boards.append(tuple(new_board))
            elif move == 2:
                new_board = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
                map_dictionary = [12,8,4,0,13,9,5,1,14,10,6,2,15,11,7,3]
                for i in range(16):
                    new_board[i] = this_state[map_dictionary[i]]
                rotated_boards.append(tuple(new_board))
            elif move == 3:
                new_board = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
                map_dictionary = [3,7,11,15,2,6,10,14,1,5,9,13,0,4,8,12]
                for i in range(16):
                    new_board[i] = this_state[map_dictionary[i]]
                rotated_boards.append(tuple(new_board))
        return rotated_boards








