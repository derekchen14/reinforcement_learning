import random

class ExperienceReplayBuffer:   # stored as ( s, a, r, s'_ )
    samples = []

    def __init__(self, capacity):
        self.capacity = capacity
        self.verbose_countdown = 10

    def remember(self, sample):
        self.samples.append(sample)

        if len(self.samples) > self.capacity:
            self.samples.pop(0)

    def get_batch(self, n):
        n = min(n, len(self.samples))
        return random.sample(self.samples, n)

    def has_more_space(self):
        if self.verbose_countdown == 0:
            ratio = len(self.samples) / self.capacity
            if ratio > 50 and ratio < 70:
                print("{:.2f} capacity filled".format(ratio))
            self.verbose_countdown = 10
        else:
            self.verbose_countdown -= 1

        return len(self.samples) < self.capacity
