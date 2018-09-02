import copy
import glob
import os, pdb
import time

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor, LongTensor

from arguments import get_args
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.vec_env.vec_normalize import VecNormalize
from baselines import bench

from model import Brain
from rollouts import ExperienceReplayBuffer
from a2c import Agent


def make_env(env_id, seed, rank, log_dir):
  def _thunk():
    env = gym.make(env_id)
    env.seed(seed + rank)
    env = bench.Monitor(env, os.path.join(log_dir, str(rank)))
    return env

  return _thunk

args = get_args()
num_updates = int(args.num_frames) // args.num_steps // args.num_processes
torch.manual_seed(args.seed)

if __name__ == "__main__":
  envs = [make_env(args.env_name, args.seed, i, args.log_dir)
        for i in range(args.num_processes)]
  envs = SubprocVecEnv(envs)
  device = "cpu"

  num_states = envs.observation_space.shape[0]
  num_actions = envs.action_space.n

  actor_critic = Brain(num_states, num_actions)
  actor_critic.to(device)

  agent = Agent(actor_critic, args.value_loss_coef, args.entropy_coef, lr=args.lr,
               eps=args.eps, alpha=args.alpha, max_grad_norm=args.max_grad_norm)

  rollouts = ExperienceReplayBuffer(args.num_steps, args.num_processes,
    num_states, envs.action_space)

  obs = envs.reset()
  rollouts.obs[0].copy_(Tensor(obs))

  # These variables are used to compute average rewards for all processes.
  episode_rewards = torch.zeros([args.num_processes, 1])
  final_rewards = torch.zeros([args.num_processes, 1])

  start = time.time()
  for j in range(num_updates):
    for step in range(args.num_steps):
      # Sample actions
      with torch.no_grad():
        value, action, action_log_prob = actor_critic.act(rollouts.obs[step])
      cpu_actions = action.squeeze(1).cpu().numpy()
      # Observe reward and next obs
      obs, reward, done, info = envs.step(cpu_actions)
      reward = torch.from_numpy(np.expand_dims(np.stack(reward), 1)).float()
      episode_rewards += reward
      # If done then clean the history of observations.
      masks = Tensor([[0.0] if done_ else [1.0] for done_ in done])
      final_rewards *= masks
      final_rewards += (1 - masks) * episode_rewards
      episode_rewards *= masks
      masks = masks.to(device)

      rollouts.insert(Tensor(obs), action, action_log_prob, value, reward, masks)

    with torch.no_grad():
      next_value = actor_critic.get_value(rollouts.obs[-1]).detach()

    rollouts.compute_returns(next_value, args.gamma)
    value_loss, action_loss, dist_entropy = agent.learn(rollouts)

    rollouts.after_update()

    if j % args.log_interval == 0:
      end = time.time()
      total_num_steps = (j + 1) * args.num_processes * args.num_steps
      mean_reward = final_rewards.mean()
      print("Updates {}, mean: {:.1f}, min/max reward {:.1f}/{:.1f}, entropy {:.5f}, value loss {:.5f}".
        format(j, mean_reward, final_rewards.min(),
          final_rewards.max(), dist_entropy, value_loss))
      if mean_reward > 100:
        print("Reached convergence!")
        break
