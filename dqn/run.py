# coding: utf-8
import gym
import pdb
import argparse
import numpy as np
import random
from collections import deque

from components.agent import Agent, RandomActor
from components.world import World
from components.plotting import Artist

parser = argparse.ArgumentParser()
parser.add_argument('-p','--problem', default='CartPole-v1', type=str,
    choices=['CartPole-v1', 'CartPole-v0', 'Acrobot-v1', 'Taxi-v2'])
parser.add_argument('-m', '--model', default='dqn', type=str,
    choices=['dqn', 'dueling', 'prioritized', 'noisy', 'rainbow', 'a3c'])

parser.add_argument('-t','--print-every', default=20, type=int)
parser.add_argument('-l','--learning-rate', default=0.001, type=float)
parser.add_argument('-o','--optimizer', default="adam", type=str)
parser.add_argument('-e','--num-episodes', default=140, type=int)
parser.add_argument('-r','--num-runs', default=1, type=int)
parser.add_argument('-b','--buffer-size', default=14000, type=int)
parser.add_argument('-s','--save-weights', default=False, action='store_true')

args = parser.parse_args()

if __name__ == "__main__":
  world = World(args.problem)
  try:
    num_states = world.environment.observation_space.n
  except(AttributeError):
    num_states = world.environment.observation_space.shape[0]
  num_actions = world.environment.action_space.n
  config = Agent.configure_model(args)

  for run in range(args.num_runs):
    # artist = Artist('dqn_{}'.format(run+1), color='c', save=args.save_weights)
    random_actor = RandomActor(num_actions, config)
    while random_actor.memory.has_more_space():
      random_actor = world.run_episode(random_actor)

    agent = Agent(num_states, num_actions, config)
    agent.memory.buffer = random_actor.memory.buffer
    world.all_rewards = []  # reset since we only care about rewards for agent

    for episode in range(args.num_episodes + 1):
      agent = world.run_episode(agent)
      average = world.calculate_average()

      if episode > 0 and episode%args.print_every == 0:
        print("Ep {0}) reward: {1}, average: {2:.2f}".format(episode, \
            world.all_rewards[-1], average))
      if average > 100.0:
        print("Early stop at episode {}".format(episode))
        break

    print("Final average: {:.3f}".format(float(sum(world.all_rewards)) / len(world.all_rewards)))
    # artist.draw(world.all_rewards)
    if args.save_weights:
      agent.brain.main_model.save("{}-{}.h5".format(args.problem, args.model))