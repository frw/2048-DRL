from collections import Counter
import numpy as np
import numpy.random as npr
import pylab as pl
import cPickle as pickle
import gzip
import os
# Uncomment to hide pygame window
# os.environ["SDL_VIDEODRIVER"] = "dummy"

from SwingyMonkey import SwingyMonkey


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

    def decide_action(self, values):
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

    def decide_action(self, values):
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
        self.last_score = None
        self.last_action = None
        self.last_reward = None
        self.scores = []

    def new_epoch(self):
        self.epoch += 1
        self.last_state = None
        self.last_score = None
        self.last_action = None
        self.last_reward = None

    def end_epoch(self):
        self.scores.append(self.last_score)

    def action_callback(self, raw_state):
        new_state = self.process_state(raw_state)
        new_action = self.decide_action(new_state)

        self.last_state = new_state
        self.last_score = raw_state['score']
        self.last_action = new_action

        return new_action

    def process_state(self, raw_state):
        def round_multiple(x, base):
            return int(base * round(float(x) / base))

        tree_state = raw_state['tree']
        monkey_state = raw_state['monkey']

        monkey_loc = (monkey_state['top'] + monkey_state['bot']) / 2

        # dist_from_center = monkey_loc - 200
        # if abs(dist_from_center) >= 98:
        #     dist_from_center = round_multiple(dist_from_center, 5)
        # else:
        #     dist_from_center = round_multiple(dist_from_center, 13)

        dist_from_gap = monkey_loc - (tree_state['top'] + tree_state['bot']) / 2
        if abs(dist_from_gap) <= 127:
            dist_from_gap = round_multiple(dist_from_gap, 5)
        elif dist_from_gap > 0:
            dist_from_gap = round_multiple(dist_from_gap - 132, 9) + 132
        else:
            dist_from_gap = round_multiple(dist_from_gap + 132, 9) - 132

        dist_from_tree = tree_state['dist']

        velocity = monkey_state['vel']
        velocity = round_multiple(velocity, 3)

        return dist_from_gap, dist_from_tree, velocity

    def decide_action(self, new_state):
        return npr.rand() < 0.1

    def reward_callback(self, reward):
        if reward < 0:
            self.last_reward = -1000
        else:
            self.last_reward = 1

    def load(self):
        if os.path.isfile(self.filename):
            with gzip.open(self.filename, 'rb') as infile:
                return pickle.load(infile)

        return self

    def save(self):
        with gzip.open(self.filename, 'wb') as outfile:
            pickle.dump(self, outfile, pickle.HIGHEST_PROTOCOL)


class ModelLearner(BaseLearner):
    def __init__(self):
        super(ModelLearner, self).__init__()
        self.state_action_count = {}
        self.state_action_reward = {}
        self.state_transition_count = {}

    def process_state(self, raw_state):
        new_state = super(ModelLearner, self).process_state(raw_state)

        if new_state not in self.state_action_count:
            self.state_action_count[new_state] = np.zeros(2)
            self.state_action_reward[new_state] = np.zeros(2)
            self.state_transition_count[new_state] = [Counter(), Counter()]

        if self.last_state is not None:
            self.state_action_count[self.last_state][self.last_action] += 1
            self.state_transition_count[self.last_state][self.last_action][new_state] += 1

        return new_state

    def reward_callback(self, reward):
        super(ModelLearner, self).reward_callback(reward)

        self.state_action_reward[self.last_state][self.last_action] += self.last_reward


class QLearner(BaseLearner):
    def __init__(self):
        super(QLearner, self).__init__()
        self.learning_rate = 0.3
        self.discount_rate = 0.95
        self.Q = {}

    def process_state(self, raw_state):
        new_state = super(QLearner, self).process_state(raw_state)

        if new_state not in self.Q:
            self.Q[new_state] = np.zeros(2)

        if self.last_state is not None:
            q = self.Q[self.last_state][self.last_action]
            self.Q[self.last_state][self.last_action] =\
                q + self.learning_rate * (self.last_reward + self.discount_rate * np.max(self.Q[new_state]) - q)

        return new_state

    def decide_action(self, new_state):
        return self.explorer.decide_action(self.Q[new_state])


class SARSALearner(BaseLearner):
    def __init__(self):
        super(SARSALearner, self).__init__()
        self.learning_rate = 0.3
        self.discount_rate = 0.95
        self.Q = {}

    def process_state(self, raw_state):
        new_state = super(SARSALearner, self).process_state(raw_state)

        if new_state not in self.Q:
            self.Q[new_state] = np.zeros(2)

        return new_state

    def decide_action(self, new_state):
        action = self.explorer.decide_action(self.Q[new_state])

        if self.last_state is not None:
            q = self.Q[self.last_state][self.last_action]
            self.Q[self.last_state][self.last_action] =\
                q + self.learning_rate * (self.last_reward + self.discount_rate * self.Q[new_state][action] - q)

        return action


class TDLearner(ModelLearner):
    def __init__(self):
        super(TDLearner, self).__init__()
        self.learning_rate = 0.3
        self.discount_rate = 0.95
        self.V = Counter()

    def process_state(self, raw_state):
        new_state = super(TDLearner, self).process_state(raw_state)

        if self.last_state is not None:
            v = self.V[self.last_state]
            self.V[self.last_state] =\
                v + self.learning_rate * (self.last_reward + self.discount_rate * self.V[new_state] - v)

        return new_state

    def decide_action(self, new_state):
        values = np.zeros(2)

        state_action_rewards = self.state_action_reward[new_state]
        state_transition_counts = self.state_transition_count[new_state]
        for i, state_action_count in enumerate(self.state_action_count[new_state]):
            if state_action_count != 0:
                values[i] = state_action_rewards[i] / state_action_count +\
                    sum([count / state_action_count * self.V[state]
                         for state, count in state_transition_counts[i].iteritems()])

        return self.explorer.decide_action(values)


class PolicyLearner(ModelLearner):
    def __init__(self):
        super(PolicyLearner, self).__init__()
        self.discount_rate = 0.95
        self.pi = {}
        self.V = {}

    def process_state(self, raw_state):
        new_state = super(PolicyLearner, self).process_state(raw_state)

        if new_state not in self.pi:
            self.pi[new_state] = npr.choice([False, True])
            self.V[new_state] = 0

        return new_state

    def get_values(self, state):
        total_rewards = self.state_action_reward[state]
        transition_counts = self.state_transition_count[state]
        return np.array([
            0 if visited_count == 0 else total_rewards[a] / visited_count + self.discount_rate *
            sum([c / visited_count * self.V[s_] for s_, c in transition_counts[a].iteritems()])
            for a, visited_count in enumerate(self.state_action_count[state])
        ])

    def decide_action(self, new_state):
        return self.explorer.decide_action(self.get_values(new_state))

    def end_epoch(self):
        while True:
            for i in range(20):
                for s, a in self.pi.iteritems():
                    visited_count = self.state_action_count[s][a]
                    self.V[s] = 0 if visited_count == 0\
                        else self.state_action_reward[s][a] / visited_count + self.discount_rate *\
                        sum([c / visited_count * self.V[s_] for s_, c in self.state_transition_count[s][a].iteritems()])

            changed = False
            for s, a_old in self.pi.iteritems():
                values = self.get_values(s)
                if max(values) != values[a_old]:
                    self.pi[s] = np.argmax(values)
                    changed = True
            if not changed:
                break


def plot(scores, title):
    indices = np.arange(1, len(scores) + 1)
    moving_average = np.convolve(scores, np.repeat(1.0, 100) / 100, 'valid')
    print moving_average[-1]

    pl.plot(indices, scores, '-')
    pl.plot(indices[99:], moving_average, 'r--')
    pl.title(title)
    pl.yscale('symlog', linthreshy=1)
    pl.ylabel("Score")
    pl.xlabel("Iteration")
    pl.show()

def run_learner(learner, iterations):
    learner = learner.load()

    saved = True
    for ii in xrange(iterations):
        # Reset the state of the learner.
        learner.new_epoch()

        # Make a new monkey object.
        swing = SwingyMonkey(sound=False,                      # Don't play sounds.
                             text="Epoch %d" % learner.epoch,  # Display the epoch on screen.
                             tick_length=0,                    # Make game ticks super fast.
                             action_callback=learner.action_callback,
                             reward_callback=learner.reward_callback)

        # Loop until you hit something.
        while swing.game_loop():
            pass

        print "Epoch %d: %d" % (learner.epoch, learner.last_score)

        # Save score
        learner.end_epoch()
        saved = False

        if learner.epoch % 1000 == 0:
            learner.save()
            saved = True

    if not saved:
        learner.save()

    plot(learner.scores, learner.__class__.__name__ + " Scores")

def graph(filename):
    if os.path.isfile(filename):
        with gzip.open(filename, 'rb') as infile:
            learner = pickle.load(infile)
        plot(learner.scores, learner.__class__.__name__ + " Scores")

run_learner(PolicyLearner(), 3000)
# graph('SARSALearner.pkl.gz')