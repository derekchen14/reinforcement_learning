import gym
gym.logger.set_level(40)

class World:
  def __init__(self, problem):
    self.problem = problem
    self.environment = gym.make(problem)
    self.all_rewards = []
    self.max_timesteps = 2000

  def calculate_average(self):
    trailing_rewards = self.all_rewards[-100:]
    average = sum(trailing_rewards) / float(len(trailing_rewards))
    return average

  def run_episode(self, agent):
    current_state = self.environment.reset()
    done = False
    episode_reward = 0

    while not done and (episode_reward <= self.max_timesteps):
      action = agent.act(current_state)
      next_state, reward, done, _ = self.environment.step(action)

      agent.observe(current_state, action, reward, next_state, done)
      agent.learn()

      current_state = next_state
      episode_reward += reward

    self.all_rewards.append(episode_reward)

    return agent

# self.environment.render()
# next_state = None if done else next_state
# agent.observe((current_state, action, reward, next_state))