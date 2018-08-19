import pdb
import math, random
import numpy as np
from torch import Tensor, LongTensor

from components.memory import ExperienceReplayBuffer
from components.brain import Brain, Keras_Brain

GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.01        # 0.1
EPSILON_DECAY = 3000      # speed of decay, larger means slower decay
MEMORY_CAPACITY = 100000
UPDATE_TARGET_FREQUENCY = 1000

BATCH_SIZE = 64
HIDDEN_DIM = 64

class Agent:
  def __init__(self, num_states, num_actions, config):
    self.name = "student"
    self.steps = 0.0
    self.num_states = num_states
    self.num_actions = num_actions
    self.epsilon = EPSILON_START
    self.use_target = True
    self.learning_rate = config['learning_rate']

    self.brain = Keras_Brain(num_states, num_actions, config)
    self.memory = ExperienceReplayBuffer(MEMORY_CAPACITY)

  @staticmethod
  def configure_model(args):
    return {
      'learning_rate': args.learning_rate,
      'model_type': args.model,
      'optimizer': args.optimizer,
      'hidden_dim': HIDDEN_DIM,
    }

  def act(self, state):
    if random.random() < self.epsilon:
      random_action = random.randrange(self.num_actions)
      return random_action
    else:
      q_val_per_action = self.brain.predict_one(state)
      return np.argmax(q_val_per_action)
    '''
    else:
      current_state = Tensor(state).unsqueeze(0)
      q_val_per_action = self.brain.main_network(current_state)
      # recall that max() returns tuple of (max_value, max_index)
      top_action = q_val_per_action.max(1)[1].numpy()[0]
      return top_action

  def observe(self, *episode):
    self.memory.remember(*episode)  #(s, a, r, s') + done
    '''

  def observe(self, episode):
    self.memory.keras_remember(episode)  #(s, a, r, s') + done

    if self.steps % UPDATE_TARGET_FREQUENCY == 0:
        self.brain.update_target_network()

    self.steps += 1.0   # anneal epsilon
    self.epsilon = EPSILON_END + (EPSILON_START - EPSILON_END) * math.exp(-self.steps / EPSILON_DECAY)

  # Q(s, a) = reward + GAMMA * argmax[ Q(s', a) ]
  # Note how the formula requires Q-scores for current and next state

  # In a neural network, each Q-score is actually a vector of numbers
  # where the dimension = (1 x num_actions)

  def keras_learn(self):
    batch = self.memory.keras_get_batch(BATCH_SIZE)

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
        selected_action = np.argmax(double_Q_score[idx])
        targets[action] = reward + GAMMA * future_Q_score[idx][ selected_action ]
      else:
        # current reward plus discounted future reward
        targets[action] = reward + GAMMA * np.amax(future_Q_score[idx])

      x[idx] = current_state
      y[idx] = targets  # Q-scores per action
      errors[idx] = abs(targets[action] - old_value)

    self.brain.train(x, y)

    return (x, y, errors)
    '''
    for idx, episode in enumerate(batch):
      current_state, action, reward, next_state = episode
      # we have an actual reward signal for one particular action
      # so update the target Q-score for that specific action
      if next_state is None:
        # then game_over, so no future reward
        target = reward
      elif self.name == 'double_dqn':
        selected_action = np.argmax(double_Q_score[idx])
        target = reward + GAMMA * future_Q_score[idx][ selected_action ]
      else:
        # current reward plus discounted future reward
        target = reward + GAMMA * np.amax(future_Q_score[idx])

      x[idx] = current_state
      # the label is the same as your predicted Q-scores per action
      y[idx] = predicted_Q_score[idx]
      # except that for the particular action you took, you can have a more accurate
      #   measure of the expected reward because you experienced that outcome
      y[idx][action] = target
      errors[idx] = abs(predicted_Q_score[idx][action] - target)
    '''

  def learn(self):
    # Each item is a vector/array with length equal to batch size
    state, action, reward, next_state, done = self.memory.get_batch(BATCH_SIZE)

    current_state  = Tensor(np.float32(state))      # (batch_size, num_state)
    next_state     = Tensor(np.float32(next_state)) # (batch_size, num_state)
    action         = LongTensor(action)             # (batch_size, ) int
    current_reward = Tensor(reward)                 # (batch_size, ) float
    done           = Tensor(done)                   # (batch_size, ) float

    # Q_scores are matrices of shape (batch_size, num_action)
    predicted_Q_score = self.brain.main_network(current_state)
    future_Q_score    = self.brain.target_network(next_state)

    # gather grabs the Q-values associated with just the action we took, it
    #     requires that the dim of the input (Q_score) matches the dim of the
    #     indexer, so we unsqueeze a dim into action to meet the requirement
    pred_q_val = predicted_Q_score.gather(1, action.unsqueeze(1)).squeeze(1)
    # when "done" is true, next_q_val is eliminated
    discounted_future_reward = GAMMA * future_Q_score.max(1)[0] * (1 - done)
    # our best prediction of Q(s', a'), which serves as part of the target label
    target_q_val = current_reward + discounted_future_reward

    loss = (pred_q_val - target_q_val).pow(2).mean()
    self.brain.train(loss)


class RandomActor:
  def __init__(self, num_actions):
    self.name = "random"
    self.num_actions = num_actions
    self.memory = ExperienceReplayBuffer(MEMORY_CAPACITY)

  def act(self, s):
    return random.randint(0, self.num_actions-1)

  # def observe(self, *episode):  # in (s, a, r, s_) format
  #   self.memory.remember(*episode)
  def observe(self, sample):
    self.memory.keras_remember(sample)

  def learn(self):
    pass  # since this agent will always act randomly

