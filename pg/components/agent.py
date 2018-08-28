import pdb
import math, random
import numpy as np
from torch import Tensor, LongTensor
from torch.distributions import Bernoulli, Categorical

from components.memory import ExperienceReplayBuffer
from components.brain import Brain

GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.01        # 0.1
EPSILON_DECAY = 700       # speed of decay, larger means slower decay
UPDATE_TARGET_FREQUENCY = 100

BATCH_SIZE = 5
HIDDEN_DIM = 36

class Agent:
  def __init__(self, num_states, num_actions, config):
    self.name = "student"
    self.steps = 0.0    # global frame counter
    self.num_states = num_states
    self.num_actions = num_actions
    self.epsilon = EPSILON_START
    self.train_every = BATCH_SIZE
    self.model_type = config['model_type']
    self.learning_rate = config['learning_rate']

    self.brain = Brain(num_states, num_actions, config)
    self.memory = ExperienceReplayBuffer(GAMMA)

  @staticmethod
  def configure_model(args):
    return {
      'learning_rate': args.learning_rate,
      'model_type': args.model,
      'optimizer': args.optimizer,
      'hidden_dim': HIDDEN_DIM,
      'buffer_size': args.buffer_size,
    }

  def act(self, state):
    current_state = Tensor(state).unsqueeze(0)
    prob_per_action = self.brain.policy(current_state)
    m = Bernoulli(prob_per_action)  # Categorical(prob_per_action)
    sampled_action = m.sample()
    log_prob = m.log_prob(Tensor(sampled_action))

    return int(sampled_action.item()), log_prob

  def observe(self, *episode):
    self.memory.remember(*episode)  #(s, a, r, log_prob)
    # if self.steps % UPDATE_TARGET_FREQUENCY == 0:
    #     self.brain.update_target_network()
    self.steps += 1.0   # anneal epsilon
    self.memory.frame_count += 1
    # self.epsilon = EPSILON_END + (EPSILON_START - EPSILON_END) * math.exp(-self.steps / EPSILON_DECAY)

  def learn(self):
    self.memory.prepare_batch()
    self.brain.optimizer.zero_grad()

    for i in range(self.memory.frame_count):
      current_reward   = self.memory.reward_pool[i]
      current_log_prob = self.memory.log_probabilities[i]
      loss = -current_log_prob * current_reward
      loss.backward()

    self.brain.optimizer.step()
    self.memory.reset_experience()


class RandomActor:
  def __init__(self, num_actions, config):
    self.name = "random"
    self.num_actions = num_actions
    self.memory = ExperienceReplayBuffer(config['buffer_size'])

  def act(self, s):
    return random.randint(0, self.num_actions-1)

  def observe(self, *episode):
    self.memory.remember(*episode)

  def learn(self):
    pass  # since this agent will always act randomly

