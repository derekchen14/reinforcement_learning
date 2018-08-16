# OpenGym CartPole-v0
# -------------------
# This code demonstrates use of a basic Q-network (without target network)
# to solve OpenGym CartPole-v0 problem.

import gym
import pdb
import argparse
import numpy as np
import math, random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F

from collections import deque
from common.wrappers import make_atari, wrap_deepmind, wrap_pytorch

from components.agent import Agent, RandomActor
from components.world import World
from components.plotting import Artist

parser = argparse.ArgumentParser()
parser.add_argument('-p','--problem', default='CartPole-v1', type=str,
    choices=['CartPole-v1', 'FrozenLake-v0', 'FrozenLake8x8-v0', 'Taxi-v2'])
parser.add_argument('-m', '--model', default='dqn', type=str,
    choices=['basic', 'dqn', 'a3c', 'random'], help='model type to choose from')
parser.add_argument('--double', default=False, action='store_true')
parser.add_argument('--dueling', default=False, action='store_true')
parser.add_argument('--prioritized', default=False, action='store_true')
parser.add_argument('--noisy', default=False, action='store_true')
parser.add_argument('--categorical', default=False, action='store_true')
parser.add_argument('--rainbow', default=False, action='store_true')

parser.add_argument('-l','--learning-rate', default=0.00025, type=float)
parser.add_argument('-e','--num-episodes', default=50, type=int)
parser.add_argument('-r','--num-runs', default=10, type=int)
parser.add_argument('-s','--save-weights', default=False, action='store_true')

args = parser.parse_args()

if __name__ == "__main__":
  universal_rewards = np.zeros(args.num_episodes)
  universal_artist = Artist('dqn_multiple_runs', save=True)
  for run in range(args.num_runs):
    world = World(args.problem)
    num_states  = world.environment.observation_space.shape[0]
    num_actions = world.environment.action_space.n

    agent = Agent(num_states, num_actions, args.learning_rate, args.model)
    random_actor = RandomActor(num_actions) # if args.model == 'dqn' else None
    artist = Artist('dqn_{}'.format(run+1), color='c', save=True)

    if random_actor is not None:
      while random_actor.memory.has_more_space():
        random_actor = world.run_episode(random_actor)
      agent.memory.samples = random_actor.memory.samples
      random_actor = None
      world.all_rewards = []  # reset since we only care about rewards for agent

    for episode in range(args.num_episodes):
      agent = world.run_episode(agent)

      ar = world.all_rewards
      average = sum(ar) / float(len(ar))
      if episode%120 == 0: #  and (agent.name != "random"):
        print("Ep {0}) reward: {1}, average: {2:.2f}".format(episode, ar[-1], average))
      if average > 100.0:
        print("Early stop due to average at {}".format(average))
        remaining = args.num_episodes - len(world.all_rewards)
        world.all_rewards.extend([1] * remaining)
        break

    print("Final average: {}".format(sum(world.all_rewards) / float(len(world.all_rewards))))
    artist.draw(world.all_rewards)
    universal_rewards += np.array(world.all_rewards)

    if args.save_weights:
      agent.brain.main_model.save("cartpole-{}.h5".format(args.model))

  universal_rewards /= args.num_runs
  universal_artist.draw(universal_rewards)