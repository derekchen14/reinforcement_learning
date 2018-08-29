import pdb
import random
import numpy as np

from collections import deque
from numpy import concatenate as concat

class ExperienceReplayBuffer(object):
  def __init__(self, gamma):
    self.gamma = gamma
    self.reset_experience()

  def reset_experience(self):
    self.done_history = []
    self.reward_pool = []
    self.log_probs = []
    self.frame_count = 0

  def remember(self, done, reward, log_prob):
    self.done_history.append(done)
    self.reward_pool.append(reward)
    self.log_probs.append(log_prob)

  def get_batch(self):
    self._discount_rewards()
    self._normalize_rewards()
    return self.log_probs, self.reward_pool

  def _discount_rewards(self):
    running_add = 0
    for i in reversed(range(self.frame_count)):
      if self.done_history[i]:
        running_add = 0
      else:
        running_add = running_add * self.gamma + self.reward_pool[i]
        self.reward_pool[i] = running_add

  def _normalize_rewards(self):
    rewards = np.array(self.reward_pool)
    rewards -= rewards.mean()
    rewards /= rewards.std() + np.finfo(np.float32).eps
    self.reward_pool = rewards


class PrioritizedReplayBuffer(object):
  def __init__(self, capacity, per_alpha=0.6):
    self.per_max    = 1.0
    self.per_alpha  = per_alpha
    self.per_beta   = 1.0 - per_alpha

    self.pos        = 0    # position pointer
    self.capacity   = capacity
    self.buffer     = deque(maxlen=capacity)
    self.priorities = np.ones((capacity,), dtype=np.float32)

  def remember(self, state, action, reward, next_state, done):
    state      = np.expand_dims(state, 0)
    next_state = np.expand_dims(next_state, 0)
    sample = (state, action, reward, next_state, done)

    self.buffer[self.pos] = sample
    self.priorities[self.pos] = self.per_max
    self.pos = (self.pos + 1) % self.capacity

  def get_batch(self, batch_size):
    probs  = self.priorities ** self.per_alpha
    probs /= probs.sum()

    self.indices = np.random.choice(self.capacity, batch_size, p=probs)
    samples  = [self.buffer[idx] for idx in self.indices]
    # Importance Sampling reweighting
    weights  = (len(self.buffer) * probs[self.indices]) ** (-self.per_beta)
    weights /= weights.max()
    self.weights = np.array(weights, dtype=np.float32)

    state, action, reward, next_state, done = zip(*samples)
    return concat(state), action, reward, concat(next_state), done

  def update_priorities(self, batch_priorities):
    self.per_max = max(self.per_max, np.max(batch_priorities))
    for idx, priority in zip(self.indices, batch_priorities):
      self.priorities[idx] = priority

  def update_beta(self, frame_idx):
    b = 1.0 - self.per_alpha
    self.per_beta = min(1.0, b + (1.0 - b) * (frame_idx / 1000))

  def __len__(self):
    return len(self.buffer)
  @property
  def ordering(self):
      return 'prioritized'


