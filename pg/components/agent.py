import pdb
import math
import numpy as np
import torch
from torch import Tensor, LongTensor
from torch.distributions import Bernoulli, Categorical

from components.memory import ExperienceReplayBuffer
from components.brain import Brain

GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.01        # 0.1
EPSILON_DECAY = 700       # speed of decay, larger means slower decay
MAX_GRAD_NORM = 0.5       # for gradient clipping
VAL_LOSS_COEF = 0.9      # for determining strength of DQN impact on loss
UPDATE_TARGET_FREQUENCY = 100

BATCH_SIZE = 5
HIDDEN_DIM = 36  #64

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
    self.val_loss_coef = config['val_loss_coef']

    self.brain = Brain(num_states, num_actions, config)
    self.memory = ExperienceReplayBuffer(GAMMA)

  @staticmethod
  def configure_model(args):
    return {
      'learning_rate': args.learning_rate,
      'model_type': args.model,
      'optimizer': args.optimizer,
      'hidden_dim': HIDDEN_DIM,
      'max_grad_norm': MAX_GRAD_NORM,
      'buffer_size': args.buffer_size,
      'val_loss_coef': VAL_LOSS_COEF,
    }

  def act(self, state):
    current_state = Tensor(state).unsqueeze(0)
    prob_per_action, values_per_action = self.brain.model(current_state)
    m = Bernoulli(prob_per_action)  # Categorical(prob_per_action)
    sampled_action = m.sample()
    log_prob = m.log_prob(Tensor(sampled_action))  # sampled_action.float()
    distribution_entropy = m.entropy().mean()
    action = int(sampled_action.item())

    return action, log_prob, values_per_action

  def observe(self, *episode):
    self.memory.remember(*episode)  #(s, a, r, log_prob, value)
    # if self.steps % UPDATE_TARGET_FREQUENCY == 0:
    #     self.brain.update_target_network()
    self.steps += 1.0   # anneal epsilon
    self.memory.frame_count += 1
    # self.epsilon = EPSILON_END + (EPSILON_START - EPSILON_END) * math.exp(-self.steps / EPSILON_DECAY)

  def learn(self):
    # log_probs is a list of Tensors, reward_pool is an array
    log_probs, reward_pool = self.memory.get_batch()
    if self.model_type == 'a2c':
      action_log_probs = torch.cat(log_probs).squeeze(1)
      value_loss = self.memory.bar * self.val_loss_coef
      action_loss = self.memory.foo.detach() * action_log_probs
      loss = (value_loss - action_loss).sum()
    else:
      loss = [-lp * reward for lp, reward in zip(log_probs, reward_pool)]
      loss = torch.cat(loss).sum()

    self.brain.train(loss)
    self.memory.reset_experience()

