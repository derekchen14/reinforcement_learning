import pdb
import random
import numpy as np

from collections import deque
from numpy import concatenate as concat

class ExperienceReplayBuffer(object):   # stored as ( s, a, r, s'_ )
  def __init__(self, capacity):
    self.buffer   = deque(maxlen=capacity)
    self.capacity = capacity
    self.ordering = 'regular'

  def remember(self, state, action, reward, next_state, done):
    state      = np.expand_dims(state, 0)
    next_state = np.expand_dims(next_state, 0)
    sample = (state, action, reward, next_state, done)
    self.buffer.append(sample)

  def get_batch(self, batch_size):
    samples = random.sample(self.buffer, batch_size)
    state, action, reward, next_state, done = zip(*samples)
    return concat(state), action, reward, concat(next_state), done

  def has_more_space(self):
    ratio = len(self.buffer) / self.capacity
    if ratio > 0.1 and round(ratio, 4) % 0.125 == 0:
      print("Replay buffer now {:.1f}% filled".format(ratio*100))
    return len(self.buffer) < self.capacity

  def __len__(self):
    return len(self.buffer)

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


class SumTreePrioritizedBuffer(object):
  '''
  Naive implementation has constant O(1) inserts, but incurs linear O(n)
  sampling since we might have to look through all experiences.  To improve
  upon this, we build a binary tree which has O(log n) inserts and pulls.
  '''
  def __init__(self, capacity, per_alpha=0.6):
    self.per_max    = 1.0
    self.per_alpha  = per_alpha
    self.per_beta   = 1.0 - per_alpha

    self.pos        = 0    # position pointer
    self.buffer     = np.zeros(capacity, dtype=object)
    self.num_leaf_nodes = capacity
    self.num_parent_nodes = capacity - 1
    self.initialize_priority_tree()

  def remember(self, state, action, reward, next_state, done):
    state      = np.expand_dims(state, 0)
    next_state = np.expand_dims(next_state, 0)
    sample = (state, action, reward, next_state, done)

    self.buffer[self.pos] = sample
    self.update_one_priority( self.pos+self.num_parent_nodes, self.per_max )
    self.pos = (self.pos + 1) % self.num_leaf_nodes

  def initialize_priority_tree(self):
    self.priority_tree = np.zeros(self.num_parent_nodes + self.num_leaf_nodes)
    for node in range(self.num_leaf_nodes):
      node_idx = node + self.num_parent_nodes
      self.update_one_priority(node_idx, self.per_max)

  def update_priorities(self, batch_priorities):
    self.per_max = max(self.per_max, np.max(batch_priorities))
    for idx, priority in zip(self.indices, batch_priorities):
      self.update_one_priority(idx, priority)

  def update_one_priority(self, idx, priority):
    change = priority - self.priority_tree[idx]
    # update the score stored at the leaf of the priority tree
    self.priority_tree[idx] = priority
    # propogate this change up the tree
    while idx != 0:
      idx = (idx - 1) // 2
      self.priority_tree[idx] += change

  def get_batch(self, batch_size):
    probs  = self.priority_tree[-self.num_leaf_nodes:] ** self.per_alpha
    probs /= probs.sum()

    self.indices = np.zeros((batch_size,), dtype=np.int32)
    priority_segment = self.total_priority / batch_size
    for i in range(batch_size):
      seg_start = priority_segment * i
      seg_end = priority_segment * (i + 1)
      # Sampling is linear in batch size, but logarithmic in buffer size
      tree_idx = self.extract_index(np.random.uniform(seg_start, seg_end))
      # convert tree_idx to buffer_idx which equals number of leaf nodes
      self.indices[i] = tree_idx - self.num_parent_nodes

    samples  = [self.buffer[idx] for idx in self.indices]
    # Importance Sampling reweighting
    weights  = (len(self.buffer) * probs[self.indices]) ** (-self.per_beta)
    weights /= weights.max()
    self.weights = np.array(weights, dtype=np.float32)

    state, action, reward, next_state, done = zip(*samples)
    return concat(state), action, reward, concat(next_state), done

  def extract_index(self, point, tree_idx=0):
    '''
    Given the point within the segment, return the index of episode.
    Recall, the priority_tree is unshuffled, so uniformly sampling from
      32 segments is equivalent to randomly sampling 32 points in total,
      and since the episode "size" is proportional to its priority,
      high priority episodes will be sampled more often
    '''
    left_child = 2 * tree_idx + 1
    right_child = left_child + 1
    # terminal stopping condition: if we reached leaves, end the search
    if left_child >= len(self.priority_tree):
      return tree_idx
    # Recurse down the tree to find the leaf nodes
    if point <= self.priority_tree[left_child]:
      return self.extract_index(point, left_child)
    else:
      point -= self.priority_tree[left_child]
      return self.extract_index(point, right_child)

  def update_beta(self, frame_idx):
    b = 1.0 - self.per_alpha
    self.per_beta = min(1.0, b + (1.0 - b) * (frame_idx / 1000))

  @property
  def total_priority(self):
      return self.priority_tree[0] # Returns the root node
  @property
  def ordering(self):
      return 'prioritized'


'''
def keras_remember(self, sample):
  self.buffer.append(sample)
def keras_get_batch(self, batch_size):
  num_samples = min(batch_size, len(self.buffer))
  return random.sample(self.buffer, num_samples)
'''
