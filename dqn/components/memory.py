import numpy as np
import random

import pdb
from collections import deque

class ExperienceReplayBuffer:   # stored as ( s, a, r, s'_ )

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity

    def remember(self, state, action, reward, next_state, done):
        state      = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)
        sample = (state, action, reward, next_state, done)
        self.buffer.append(sample)

    def get_batch(self, batch_size):
        num_samples = min(batch_size, len(self.buffer))
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, num_samples))
        return np.concatenate(state), action, reward, np.concatenate(next_state), done

    def has_more_space(self):
        ratio = len(self.buffer) / self.capacity
        if ratio > 0.1 and round(ratio, 4) % 0.125 == 0:
            # pdb.set_trace()
            print("Replay buffer now {:.1f}% filled".format(ratio*100))

        return len(self.buffer) < self.capacity

    '''
    def keras_remember(self, sample):
        self.buffer.append(sample)
    def keras_get_batch(self, batch_size):
        num_samples = min(batch_size, len(self.buffer))
        return random.sample(self.buffer, num_samples)
    '''
