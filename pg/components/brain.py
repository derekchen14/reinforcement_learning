import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_ as clip_grad
from torch import Tensor, LongTensor

torch.manual_seed(1)
np.random.seed(1)

class Brain:
  def __init__(self, num_states, num_actions, config):
    self.num_states = num_states
    self.num_actions = num_actions
    self.hidden_dim = config['hidden_dim']
    self.max_grad_norm = config['max_grad_norm']

    self.model = self._create_model(config['model_type'])
    self.optimizer = self._construct_optimizer(config)

  def _create_model(self, model_type):
    if model_type == 'pg':
      net = PolicyNet(self.num_states, self.num_actions, self.hidden_dim)
    elif model_type == 'a2c':
      net = A2C(self.num_states, self.num_actions, self.hidden_dim)
    elif model_type == 'ppo':
      net = PPO(self.num_states, self.num_actions, self.hidden_dim)
    return net.cuda() if torch.cuda.is_available() else net

  def _construct_optimizer(self, config):
    params = self.model.parameters()
    if config['optimizer'] == 'adam':
      return optim.Adam(params, config['learning_rate'])    # 0.001
    elif config['optimizer'] == 'rms':
      return optim.RMSprop(params, config['learning_rate']) # 0.01
      # return optim.RMSprop(params, lr=7e-4, eps=1e-5, alpha=0.99)

  def train(self, loss):
    self.optimizer.zero_grad()
    loss.backward()
    params = self.model.parameters()
    clip_grad(params, self.max_grad_norm)
    self.optimizer.step()

def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module
def init_normc_(weight, gain=1):
    weight.normal_(0, 1)
    weight *= gain / torch.sqrt(weight.pow(2).sum(1, keepdim=True))


class A2C(nn.Module):
  def __init__(self, num_states, num_actions, hidden_dim):
    super(A2C, self).__init__()
    # Policy Network
    self.actor = nn.Sequential(
        nn.Linear(num_states, hidden_dim), nn.Tanh(),
        nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
        nn.Linear(hidden_dim, 1), nn.Sigmoid()
    )
    # Advantage Function
    self.critic = nn.Sequential(
        nn.Linear(num_states, hidden_dim), nn.Tanh(),
        nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
        nn.Linear(hidden_dim, num_actions)
    )
  def forward(self, x):
    return self.actor(x), self.critic(x)

class PolicyNet(nn.Module):
  def __init__(self, num_states, num_actions, hidden_dim):
    super(PolicyNet, self).__init__()
    self.fc1 = nn.Linear(num_states, hidden_dim)
    self.fc2 = nn.Linear(hidden_dim, hidden_dim)
    self.fc3 = nn.Linear(hidden_dim, 1)  # num_actions
    self.value = None  # placeholder so arg_count lines up with A2C

  def forward(self, x):
    x = torch.relu(self.fc1(x))
    x = torch.relu(self.fc2(x))
    x = torch.sigmoid(self.fc3(x)) # .softmax(self.fc3(x), dim=1)
    return x, self.value