from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *
from keras import backend as K

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

def huber_loss(y_true, y_pred):
    huber_loss_delta = 1.0
    err = y_true - y_pred
    cond = K.abs(err) < huber_loss_delta

    L2 = 0.5 * K.square(err)
    L1 = huber_loss_delta * (K.abs(err) - 0.5 * huber_loss_delta)
    loss = tf.where(cond, L2, L1)   # Keras does not cover where function in TF
    return K.mean(loss)


class Q_Network:
  def __init__(self, num_states, num_actions, learning_rate, use_target=False):
    self.num_states = num_states
    self.num_actions = num_actions
    self.learning_rate = learning_rate

    self.use_target = use_target
    self.main_model = self._create_model()
    if use_target:
      self.target_network = self._create_model()
    # self.model.load_weights("cartpole-basic.h5")

  def _create_model(self):
    model = Sequential()
    model.add(Dense(units=64, activation='relu', input_dim=self.num_states))
    model.add(Dense(units=self.num_actions, activation='linear'))

    opt = RMSprop(lr=self.learning_rate)
    loss_func = huber_loss if self.use_target else 'mse'
    model.compile(loss=loss_func, optimizer=opt)

    return model

  def train(self, x, y, epoch=1, verbose=0):
    self.main_model.fit(x, y, batch_size=64, epochs=epoch, verbose=verbose)

  def predict(self, s, target=False):
    if target:
      return self.target_network.predict(s)
    else:
      return self.main_model.predict(s)

  def predict_one(self, state):
    flat_state = state.reshape(1, self.num_states)
    return self.predict(flat_state).flatten()

  def update_target_network(self):
    learned_weights = self.main_model.get_weights()
    self.target_network.set_weights(learned_weights)
