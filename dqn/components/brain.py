import pdb
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

    self.main_network = self._create_model()
    self.target_network = self._create_model()
    self.optimizer = self._construct_optimizer(config)

  def _create_model(self):
    net = Q_Network(self.num_states, self.num_actions, self.hidden_dim)
    return net.cuda() if torch.cuda.is_available() else net

  def _construct_optimizer(self, config):
    params = self.main_network.parameters()
    if config['optimizer'] == 'adam':
      return optim.Adam(params, config['learning_rate'])    # 0.001
    elif config['optimizer'] == 'rms':
      return optim.RMSprop(params, config['learning_rate']) # 0.01

  def train(self, pred, target):
    loss = nn.SmoothL1Loss()  # huber_loss
    # loss = nn.MSELoss()
    output = loss(pred, target)

    self.optimizer.zero_grad()
    output.backward()
    self.optimizer.step()

  def update_target_network(self):
    learned_weights = self.main_network.state_dict()
    self.target_network.load_state_dict(learned_weights)

class Q_Network(nn.Module):
  def __init__(self, num_states, num_actions, hidden_dim):
    super(Q_Network, self).__init__()
    self.fc1 = nn.Linear(num_states, hidden_dim)
    self.fc2 = nn.Linear(hidden_dim, hidden_dim)
    self.fc3 = nn.Linear(hidden_dim, num_actions)

  def forward(self, x):
    x = torch.relu(self.fc1(x))
    x = torch.relu(self.fc2(x))
    x = self.fc3(x)
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

