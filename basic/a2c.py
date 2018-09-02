import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_ as clip_grad
import sys, pdb

class Agent():
  def __init__(self, actor_critic, value_loss_coef, entropy_coef,
                lr=None, eps=None, alpha=None, max_grad_norm=None):

    self.actor_critic = actor_critic
    self.value_loss_coef = value_loss_coef
    self.entropy_coef = entropy_coef

    self.max_grad_norm = max_grad_norm
    self.optimizer = optim.RMSprop(
      actor_critic.parameters(), lr, eps=eps, alpha=alpha)

  def learn(self, rollouts):
    num_states = 4
    num_actions = 1
    batch_size = 5
    num_processes = 16

    values, action_log_probs, dist_entropy = self.actor_critic.evaluate_actions(
      rollouts.obs[:-1].view(-1, num_states),
      rollouts.actions.view(-1, num_actions))
    # (80,1)  (80,1)  (scalar)

    values = values.view(batch_size, num_processes, 1)
    action_log_probs = action_log_probs.view(batch_size, num_processes, 1)
    advantages = rollouts.returns[:-1] - values

    self.optimizer.zero_grad()

    value_loss = advantages.pow(2).mean() * self.value_loss_coef
    action_loss = (advantages.detach() * action_log_probs).mean()
    entropy_loss = dist_entropy * self.entropy_coef
    (value_loss - action_loss - entropy_loss).backward()

    params = self.actor_critic.parameters()
    clip_grad(params, self.max_grad_norm)
    self.optimizer.step()

    return value_loss.item(), action_loss.item(), dist_entropy.item()
