import numpy as np

class ExperienceDatabase(object):
    def __init__(self, capacity=1000000):
        self.capacity = capacity
        self.history = []
        self.size = 0

    def add(self, last_state, last_action, last_reward, new_state, possible_moves):
        data = (last_state, last_action, last_reward, new_state, possible_moves)

        if self.size >= self.capacity:
            self.history[self.size % self.capacity] = data
        else:
            self.history.append(data)

        self.size += 1

    def sample(self, size=1):
        samples = np.random.randint(min(self.size, self.capacity), size=size)
        result = []
        for sample in samples:
            result.append(self.history[sample])
        return result


