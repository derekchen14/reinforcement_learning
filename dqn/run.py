# coding: utf-8
import gym
import pdb
import argparse
import numpy as np
import random
from collections import deque
# from common.wrappers import make_atari, wrap_deepmind, wrap_pytorch

# from config import *
from components.agent import Agent, RandomActor
from components.world import World
from components.plotting import Artist

parser = argparse.ArgumentParser()
parser.add_argument('-p','--problem', default='CartPole-v0', type=str,
    choices=['CartPole-v0', 'CartPole-v1', 'FrozenLake-v0', 'Taxi-v2'])
parser.add_argument('-m', '--model', default='dqn', type=str,
    choices=['basic', 'dqn', 'a3c', 'random'], help='model type to choose from')
parser.add_argument('--double', default=False, action='store_true')
parser.add_argument('--dueling', default=False, action='store_true')
parser.add_argument('--prioritized', default=False, action='store_true')
parser.add_argument('--noisy', default=False, action='store_true')
parser.add_argument('--categorical', default=False, action='store_true')
parser.add_argument('--rainbow', default=False, action='store_true')

parser.add_argument('-l','--learning-rate', default=0.00025, type=float)
parser.add_argument('-o','--optimizer', default="Adam", type=str)
parser.add_argument('-e','--num-episodes', default=50, type=int)
parser.add_argument('-r','--num-runs', default=3, type=int)
parser.add_argument('-b','--buffer-size', default=100000, type=int)
parser.add_argument('-s','--save-weights', default=False, action='store_true')

args = parser.parse_args()

if __name__ == "__main__":
  random.seed(14)

  if args.num_runs > 1:
    universal_rewards = np.zeros(args.num_episodes)
    universal_artist = Artist('dqn_single_run', save=True)

  for run in range(args.num_runs):
    world = World(args.problem)
    num_states  = world.environment.observation_space.shape[0]
    num_actions = world.environment.action_space.n
    config = Agent.configure_model(args)

    agent = Agent(num_states, num_actions, config)
    random_actor = RandomActor(num_actions, config)
    artist = Artist('dqn_{}'.format(run+1), color='c', save=True)

    while random_actor.memory.has_more_space():
      random_actor = world.run_episode(random_actor)
    agent.memory.buffer = random_actor.memory.buffer
    world.all_rewards = []  # reset since we only care about rewards for agent

    for episode in range(args.num_episodes + 1):
      agent = world.run_episode(agent)
      average = world.calculate_average()

      if episode > 0 and episode%100 == 0:
        print("Ep {0}) reward: {1}, average: {2:.2f}".format(episode, \
            world.all_rewards[-1], average))
      if average > 100.0:
        print("Early stop due to average at {}".format(average))
        remaining = args.num_episodes - len(world.all_rewards)
        world.all_rewards.extend([1] * remaining)
        break

    # print("Final average: {:.3f}".format(sum(world.all_rewards) / float(len(world.all_rewards))))
    artist.draw(world.all_rewards)
    if args.num_runs > 1:
      universal_rewards += np.array(world.all_rewards[1:])

    if args.save_weights:
      agent.brain.main_model.save("{}-{}.h5".format(args.problem, args.model))

  if args.num_runs > 1:
    universal_rewards /= args.num_runs
    universal_artist.draw(universal_rewards)