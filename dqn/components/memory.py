import numpy as np
import random
from collections import deque

class ExperienceReplayBuffer:   # stored as ( s, a, r, s'_ )

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity
        self.verbose_countdown = 10

    def __len__(self):
        return len(self.buffer)

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
        if self.verbose_countdown == 0:
            ratio = len(self.buffer) / self.capacity
            if ratio > 50 and ratio < 70:
                print("{:.2f} capacity filled".format(ratio))
            self.verbose_countdown = 10
        else:
            self.verbose_countdown -= 1

        return len(self.buffer) < self.capacity
