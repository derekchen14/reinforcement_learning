import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor, LongTensor

def huber_loss(y_true, y_pred):
  huber_loss_delta = 1.0
  err = y_true - y_pred
  cond = K.abs(err) < huber_loss_delta

  L2 = 0.5 * K.square(err)
  L1 = huber_loss_delta * (K.abs(err) - 0.5 * huber_loss_delta)
  loss = tf.where(cond, L2, L1)   # Keras does not cover where function in TF
  return K.mean(loss)

class Brain:
  def __init__(self, num_states, num_actions, config):
    self.num_states = num_states
    self.num_actions = num_actions
    self.hidden_dim = config['hidden_dim']
    self.prioritized = config['prioritized']

    self.main_network = self._create_model(config['model_type'])
    self.target_network = self._create_model(config['model_type'])
    self.optimizer = self._construct_optimizer(config)

  def _create_model(self, model_type):
    if model_type == 'dqn' or model_type == 'prioritized':
      net = DQN(self.num_states, self.num_actions, self.hidden_dim)
    elif model_type == 'dueling':
      net = Dueling(self.num_states, self.num_actions, self.hidden_dim)
    elif model_type == 'noisy':
      net = NoisyNet(self.num_states, self.num_actions, self.hidden_dim)
    elif model_type == 'rainbow':
      net = Rainbow(self.num_states, self.num_actions, self.hidden_dim)
    return net.cuda() if torch.cuda.is_available() else net

  def _construct_optimizer(self, config):
    params = self.main_network.parameters()
    if config['optimizer'] == 'adam':
      return optim.Adam(params, config['learning_rate'])    # 0.001
    elif config['optimizer'] == 'rms':
      return optim.RMSprop(params, config['learning_rate']) # 0.01

  def train_with_per(self, pred, target, weights):
    loss  = (pred - target).pow(2) * weights
    prioritized_loss = loss + 0.001  # small epsilon to avoid zero probability
    loss  = loss.mean()

    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()

    return prioritized_loss

  def train(self, pred, target):
    loss_func = nn.SmoothL1Loss()  # huber_loss
    # loss_func = nn.MSELoss()
    loss = loss_func(pred, target)

    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()

  def update_target_network(self):
    learned_weights = self.main_network.state_dict()
    self.target_network.load_state_dict(learned_weights)

  def reset_noise(self):
    self.main_network.noisy1.reset_noise()
    self.main_network.noisy2.reset_noise()
    self.target_network.noisy1.reset_noise()
    self.target_network.noisy2.reset_noise()


class DQN(nn.Module):
  def __init__(self, num_states, num_actions, hidden_dim):
    super(DQN, self).__init__()
    self.fc1 = nn.Linear(num_states, hidden_dim)
    self.fc2 = nn.Linear(hidden_dim, hidden_dim)
    self.fc3 = nn.Linear(hidden_dim, num_actions)

  def forward(self, x):
    x = torch.relu(self.fc1(x))
    x = torch.relu(self.fc2(x))
    x = self.fc3(x)
    return x


class NoisyNet(nn.Module):
  def __init__(self, num_states, num_actions, hidden_dim):
    super(NoisyNet, self).__init__()
    self.linear = nn.Linear(num_states, hidden_dim)
    self.noisy1 = NoisyLayer(hidden_dim, hidden_dim)
    self.noisy2 = NoisyLayer(hidden_dim, num_actions)

  def forward(self, x):
    x = torch.relu(self.linear(x))
    x = torch.relu(self.noisy1(x))
    x = self.noisy2(x)
    return x


class Dueling(nn.Module):
  def __init__(self, num_states, num_actions, hidden_dim):
    super(Dueling, self).__init__()
    self.affine = nn.Sequential( nn.Linear(num_states, 128), nn.ReLU() )
    self.advantage = nn.Sequential(
          nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, num_actions)
    )
    self.value = nn.Sequential(
          nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 1)
    )

  def forward(self, x):
    x = self.affine(x)
    advantage = self.advantage(x)
    value     = self.value(x)
    return value + advantage  - advantage.mean()


class NoisyLayer(nn.Module):
  def __init__(self, in_dim, out_dim, std_init=0.4):
    super(NoisyLayer, self).__init__()
    self.in_dim = in_dim
    self.out_dim = out_dim
    self.std_init = std_init

    self.weight_mu    = nn.Parameter(Tensor(out_dim, in_dim))
    self.weight_sigma = nn.Parameter(Tensor(out_dim, in_dim))
    self.register_buffer('weight_epsilon', Tensor(out_dim, in_dim))

    self.bias_mu    = nn.Parameter(Tensor(out_dim))
    self.bias_sigma = nn.Parameter(Tensor(out_dim))
    self.register_buffer('bias_epsilon', Tensor(out_dim))

    self.reset_parameters()
    self.reset_noise()

  def forward(self, x):
    if self.training:
      weight = self.weight_mu + self.weight_sigma.mul(self.weight_epsilon)
      bias   = self.bias_mu   + self.bias_sigma.mul(self.bias_epsilon)
    else:
      weight = self.weight_mu
      bias   = self.bias_mu

    return nn.functional.linear(x, weight, bias)

  def reset_parameters(self):
      mu_range = 1 / math.sqrt(self.weight_mu.size(1))

      self.weight_mu.data.uniform_(-mu_range, mu_range)
      self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.weight_sigma.size(1)))

      self.bias_mu.data.uniform_(-mu_range, mu_range)
      self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.bias_sigma.size(0)))

  def reset_noise(self):
      epsilon_in  = self._scale_noise(self.in_dim)
      epsilon_out = self._scale_noise(self.out_dim)
      # using factorized Gaussian noise (rather than independent) with
      # torch.ger() which performs a non-broadcasting outer product
      self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
      self.bias_epsilon.copy_(self._scale_noise(self.out_dim))

  def _scale_noise(self, size):
      x = torch.randn(size)
      x = x.sign().mul(x.abs().sqrt())
      return x



'''
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *
from keras import backend as K

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

class Keras_Brain:
  def __init__(self, num_states, num_actions, config):
    self.num_states = num_states
    self.num_actions = num_actions
    self.learning_rate = config['learning_rate']
    self.main_network = self._create_model()
    self.target_network = self._create_model()
  def _create_model(self):
    model = Sequential()
    model.add(Dense(units=64, activation='relu', input_dim=self.num_states))
    model.add(Dense(units=self.num_actions, activation='linear'))
    opt = RMSprop(lr=self.learning_rate)
    loss_func = huber_loss # 'mse'
    model.compile(loss=loss_func, optimizer=opt)
    return model
  def train(self, x, y, epoch=1, verbose=0):
    self.main_network.fit(x, y, batch_size=64, epochs=epoch, verbose=verbose)
  def predict(self, s, target=False):
    if target:
      return self.target_network.predict(s)
    else:
      return self.main_network.predict(s)
  def predict_one(self, state):
    flat_state = state.reshape(1, self.num_states)
    return self.predict(flat_state).flatten()
  def update_target_network(self):
    learned_weights = self.main_network.get_weights()
    self.target_network.set_weights(learned_weights)
'''

