import torch
import torch.nn as nn
import torch.nn.functional as F
from distributions import Categorical
from utils import init, init_normc_
import pdb

class Brain(nn.Module):
    def __init__(self, num_states, num_actions):
        super(Brain, self).__init__()

        self.model = A2C(num_states)
        self.dist = Categorical(self.model.output_size, num_actions)

    def forward(self, inputs):
        raise NotImplementedError

    def act(self, inputs, deterministic=False):
        value, actor_features = self.model(inputs)
        dist = self.dist(actor_features)
        action = dist.sample()
        action_log_probs = dist.log_probs(action)

        return value, action, action_log_probs

    def get_value(self, inputs):
        value, _ = self.model(inputs)
        return value

    def evaluate_actions(self, inputs, action):
        value, actor_features = self.model(inputs)
        dist = self.dist(actor_features)
        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy

class A2C(nn.Module):
    def __init__(self, num_states, hidden_size=64):
        super(A2C, self).__init__()
        self.output_size = hidden_size

        init_ = lambda m: init(m,
            init_normc_,
            lambda x: nn.init.constant_(x, 0))

        # pdb.set_trace()

        self.actor = nn.Sequential(
            init_(nn.Linear(num_states, hidden_size)),
            nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh()
        )

        self.critic = nn.Sequential(
            init_(nn.Linear(num_states, hidden_size)),
            nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh()
        )

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs):
        x = inputs
        hidden_critic = self.critic(x)
        hidden_actor = self.actor(x)

        return self.critic_linear(hidden_critic), hidden_actor