# OpenGym CartPole-v0
# -------------------
#
# This code demonstrates use a full DQN implementation
# to solve OpenGym CartPole-v0 problem.
#
# Made as part of blog series Let's make a DQN, available at: 
# https://jaromiru.com/2016/09/27/lets-make-a-dqn-theory/
# 
# author: Jaromir Janisch, 2016

import random, numpy, math, gym, sys
from keras import backend as K

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from components.mem import ExperienceReplayBuffer
from components.brain import Q_Network

# #----------
# HUBER_LOSS_DELTA = 1.0
# LEARNING_RATE = 0.00025

# #----------
# def huber_loss(y_true, y_pred):
#     err = y_true - y_pred

#     cond = K.abs(err) < HUBER_LOSS_DELTA
#     L2 = 0.5 * K.square(err)
#     L1 = HUBER_LOSS_DELTA * (K.abs(err) - 0.5 * HUBER_LOSS_DELTA)

#     loss = tf.where(cond, L2, L1)   # Keras does not cover where function in tensorflow :-(

#     return K.mean(loss)

# #-------------------- BRAIN ---------------------------
# from keras.models import Sequential
# from keras.layers import *
# from keras.optimizers import *

# class Brain:
#     def __init__(self, stateCnt, actionCnt):
#         self.stateCnt = stateCnt
#         self.actionCnt = actionCnt

#         self.model = self._createModel()
#         self.model_ = self._createModel()

#     def _createModel(self):
#         model = Sequential()

#         model.add(Dense(units=64, activation='relu', input_dim=stateCnt))
#         model.add(Dense(units=actionCnt, activation='linear'))

#         opt = RMSprop(lr=LEARNING_RATE)
#         model.compile(loss=huber_loss, optimizer=opt)

#         return model

#     def train(self, x, y, epochs=1, verbose=0):
#         self.model.fit(x, y, batch_size=64, epochs=epochs, verbose=verbose)

#     def predict(self, s, target=False):
#         if target:
#             return self.model_.predict(s)
#         else:
#             return self.model.predict(s)

#     def predictOne(self, s, target=False):
#         return self.predict(s.reshape(1, self.stateCnt), target=target).flatten()

#     def updateTargetModel(self):
#         self.model_.set_weights(self.model.get_weights())


#-------------------- AGENT ---------------------------
MEMORY_CAPACITY = 100000
BATCH_SIZE = 64

GAMMA = 0.99

MAX_EPSILON = 1
MIN_EPSILON = 0.01
LAMBDA = 0.001      # speed of decay

UPDATE_TARGET_FREQUENCY = 1000

class Agent:
    steps = 0
    epsilon = MAX_EPSILON

    def __init__(self, stateCnt, actionCnt):
        self.stateCnt = stateCnt
        self.actionCnt = actionCnt

        self.brain = Q_Network(stateCnt, actionCnt, 0.00025, True)
        self.memory = ExperienceReplayBuffer(MEMORY_CAPACITY)
        
    def act(self, s):
        if random.random() < self.epsilon:
            return random.randint(0, self.actionCnt-1)
        else:
            return numpy.argmax(self.brain.predict_one(s))

    def observe(self, sample):  # in (s, a, r, s_) format
        self.memory.remember(sample)        

        if self.steps % UPDATE_TARGET_FREQUENCY == 0:
            self.brain.update_target_network()

        # debug the Q function in poin S
        if self.steps % 100 == 0:
            S = numpy.array([-0.01335408, -0.04600273, -0.00677248, 0.01517507])
            pred = agent.brain.predict_one(S)
            # print(pred[0])
            # sys.stdout.flush()

        # slowly decrease Epsilon based on our eperience
        self.steps += 1
        self.epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * math.exp(-LAMBDA * self.steps)

    def learn(self):    
        batch = self.memory.get_batch(BATCH_SIZE)
        batchLen = len(batch)

        no_state = numpy.zeros(self.stateCnt)

        states = numpy.array([ o[0] for o in batch ])
        states_ = numpy.array([ (no_state if o[3] is None else o[3]) for o in batch ])

        p = self.brain.predict(states)
        p_ = self.brain.predict(states_, target=True)

        x = numpy.zeros((batchLen, self.stateCnt))
        y = numpy.zeros((batchLen, self.actionCnt))
        
        for i in range(batchLen):
            o = batch[i]
            s = o[0]; a = o[1]; r = o[2]; s_ = o[3]
            
            t = p[i]
            if s_ is None:
                t[a] = r
            else:
                t[a] = r + GAMMA * numpy.amax(p_[i])

            x[i] = s
            y[i] = t

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

#-------------------- ENVIRONMENT ---------------------
class Environment:
    def __init__(self, problem):
        self.problem = problem
        self.env = gym.make(problem)

    def run(self, agent):
        s = self.env.reset()
        R = 0 

        while True:            
            # self.env.render()

            a = agent.act(s)

            s_, r, done, info = self.env.step(a)

            if done: # terminal state
                s_ = None

            agent.observe( (s, a, r, s_) )
            agent.learn()            

            s = s_
            R += r

            if done:
                break

        return R

#-------------------- MAIN ----------------------------
PROBLEM = 'CartPole-v0'
env = Environment(PROBLEM)

stateCnt  = env.env.observation_space.shape[0]
actionCnt = env.env.action_space.n

agent = Agent(stateCnt, actionCnt)
randomAgent = RandomActor(actionCnt)

while randomAgent.memory.has_more_space():
    env.run(randomAgent)

agent.memory.samples = randomAgent.memory.samples
randomAgent = None

counter = 0.0
total_reward = 0
while True:
    counter += 1

    episode_reward = env.run(agent)
    total_reward += episode_reward
    average_reward = total_reward / counter

    if counter%100 == 0:
        print("Total reward: {}, with average {:.2f}".format(episode_reward, average_reward))

    if counter > 100 and average_reward > 50.0:
        print("Average: {}".format(average_reward))
        break

agent.brain.model.save("cartpole-dqn.h5")
