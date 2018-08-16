import random
import math
import numpy as np
import pdb

from components.mem import ExperienceReplayBuffer
from components.brain import Q_Network

GAMMA = 0.99
MAX_EPSILON = 1
MIN_EPSILON = 0.01   # 0.1
LAMBDA = 0.001      # speed of decay
MEMORY_CAPACITY = 100000
BATCH_SIZE = 64

UPDATE_TARGET_FREQUENCY = 1000

class Agent:
  def __init__(self, num_states, num_actions, learning_rate, model_type):
    self.name = "student"
    self.steps = 0
    self.num_states = num_states
    self.num_actions = num_actions
    self.epsilon = MAX_EPSILON

    self.learning_rate = learning_rate
    self.use_target = True if model_type == "dqn" else False

    self.brain = Q_Network(num_states, num_actions, learning_rate, self.use_target)
    self.memory = ExperienceReplayBuffer(MEMORY_CAPACITY)

  def act(self, state):
    if random.random() < self.epsilon:
      random_action = random.randint(0, self.num_actions-1)
      return random_action
    else:
      actions = self.brain.predict_one(state)
      top_action = np.argmax(actions)
      return top_action

  def observe(self, sample):  # in (s, a, r, s'_) format
    self.memory.remember(sample)

    if self.use_target and (self.steps % UPDATE_TARGET_FREQUENCY == 0):
        self.brain.update_target_network()
    # debug the Q function in poin S
    # if self.steps % 100 == 0:
    #     S = numpy.array([-0.01335408, -0.04600273, -0.00677248, 0.01517507])
    #     pred = agent.brain.predict_one(S)
    #     print(pred[0])
    #     sys.stdout.flush()

    # slowly decrease Epsilon based on our eperience
    self.steps += 1
    self.epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * math.exp(-LAMBDA * self.steps)

    '''
    Initialize your targets with the scores per action as predicted by
      the original model
    Suppose your action choices were [left, stay, or right]
      and your original Q-network predicts scores of 0.4, 0.5, and 0.9
      for taking those respective actions.
    Next, it turns out that in this example, the action you actually
      sampled was the "left" action.  Then, you actually have a more
      accurate estimate of that particular action because you get a
      genuine reward signal R (plus the discounted future Q-score)
    So good, go ahead and update the target score for that particular action
    Since of course the Brain predicts the same action scores for itself
      when done twice, no gradients occur for those [middle, right] actions
      since the loss is zero.  However, for the "left" action, a gradient
      *will* occur because the loss is non-zero.
    You now have a set of target scores per action, given the current state
      which serve as your Y and X respectively.

    Q(s, a) = reward + GAMMA * argmax[ Q(s', a) ]
    Note how the formula requires Q-scores for current and next state

    In a neural network, each Q-score is actually a vector of numbers
    where the dimension = (1 x num_actions)
    '''

  def gather_targets(self, batch):
    batch_length = len(batch)
    # Q(s, a) predictions per action given the current state
    current_states = np.array([ o[0] for o in batch ])
    predicted_Q_score = self.brain.predict(current_states)
    # Q(s', a) predictions per action given the next state
    empty = np.zeros(self.num_states)
    next_states = np.array([ (empty if o[3] is None else o[3]) for o in batch ])
    # double_Q_score = self.brain.predict(next_states, target=False)
    future_Q_score = self.brain.predict(next_states, target=self.use_target)

    # the inputs into our neural network, which are the set of states
    x = np.zeros((batch_length, self.num_states))
    # the target labels we want to predict, which are the set of actions
    y = np.zeros((batch_length, self.num_actions))
    # Errors used for calculating priority experience replay
    errors = np.zeros(batch_length)

    for idx, episode in enumerate(batch):
      current_state, action, reward, next_state = episode

      # by default we are correct, so initialize our target to be our prediction
      targets = predicted_Q_score[idx]
      old_value = targets[action]
      # we have an actual reward signal for one particular action
      # so update the target Q-score for that specific action
      if next_state is None:
        # then game_over, so no future reward
        targets[action] = reward
      elif self.name == 'double_dqn':
        selected_action = np.argmax(double_Q_score[i])
        targets[action] = reward + GAMMA * future_Q_score[i][ selected_action ]
      else:
        # current reward plus discounted future reward
        targets[action] = reward + GAMMA * np.amax(future_Q_score[idx])


      x[idx] = current_state
      y[idx] = targets  # Q-scores per action
      errors[idx] = abs(targets[action] - old_value)

    return (x, y, errors)

def learn(self):
  batch = self.memory.get_batch(BATCH_SIZE)
  x, y, errors = self.gather_targets(batch)
  #update errors
  for i in range(batch_length):
    j = batch[i][0]
    self.memory.update(j, errors[i])

  self.brain.train(x, y)


class RandomActor:
  def __init__(self, num_actions):
    self.name = "random"
    self.num_actions = num_actions
    self.memory = ExperienceReplayBuffer(MEMORY_CAPACITY)

  def act(self, s):
    return random.randint(0, self.num_actions-1)

  def observe(self, sample):  # in (s, a, r, s_) format
    self.memory.remember(sample)

  def learn(self):
    pass  # since this agent will always act randomly