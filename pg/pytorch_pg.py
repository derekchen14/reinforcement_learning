import argparse
import gym
import numpy as np
from itertools import count

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical


parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
args = parser.parse_args()

env = gym.make('CartPole-v0')
env.seed(args.seed)
torch.manual_seed(args.seed)

class PolicyNet(nn.Module):
    def __init__(self):
        super(PolicyNet, self).__init__()

        self.fc1 = nn.Linear(4, 24)
        self.fc2 = nn.Linear(24, 36)
        self.fc3 = nn.Linear(36, 1)  # Prob of Left
        self.reset_histories()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        return x

    def reset_histories(self):
        self.log_prob_history = []
        self.reward_history = []

def select_action(state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs = policy(state)
    m = Categorical(probs)
    action = m.sample()
    policy.log_prob_history.append(m.log_prob(action))
    return action.item()



def rollout_episode():
    for t in range(10000):  # Don't infinite loop while learning
        action = select_action(state)
        state, reward, done, _ = env.step(action)
        if args.render:
            env.render()
        policy.reward_history.append(reward)
        if done:
            break
    running_reward = running_reward * 0.99 + t * 0.01

def discount_rewards():
    R = 0
    policy_loss = []
    rewards = []
    for r in policy.reward_history[::-1]:
        R = r + args.gamma * R
        rewards.insert(0, R)

    #normalize rewards
    rewards = torch.tensor(rewards)
    rewards -= rewards.mean()
    rewards /= rewards.std() + eps

def update_loss():
    for log_prob, reward in zip(policy.log_prob_history, rewards):
        policy_loss.append(-log_prob * reward)
    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward()
    optimizer.step()

    policy.reset_histories()

def display_progress():
    if episode_idx % args.log_interval == 0:
        print('Episode {}\tLast length: {:5d}\tAverage length: {:.2f}'.format(
            episode_idx, t, running_reward))
    if running_reward > env.spec.reward_threshold:
        print("Solved! Running reward is now {} and "
              "the last episode runs to {} time steps!".format(running_reward, t))
        break


if __name__ == '__main__':
    running_reward = 10

    policy = PolicyNet()
    optimizer = optim.Adam(policy.parameters(), lr=1e-2)
    eps = np.finfo(np.float32).eps.item()

    for episode_idx in count(1):
        state = env.reset()

        rollout_episode()
        discount_rewards()
        update_loss()

        display_progress()