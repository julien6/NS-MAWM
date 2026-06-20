import json
import os
import sys
from collections import namedtuple

import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exp_config import ACTION_SIZE, NUM_MIXTURE, RNN_SIZE, Z_SIZE

LOGSTD_MIN = -6.0
LOGSTD_MAX = 2.0

HyperParams = namedtuple('HyperParams', [
  'z_size',
  'action_size',
  'rnn_size',
  'num_mixture',
  'batch_size',
  'max_seq_len',
  'learning_rate',
  'is_training',
])


def default_hps():
  return HyperParams(
    z_size=Z_SIZE,
    action_size=ACTION_SIZE,
    rnn_size=RNN_SIZE,
    num_mixture=NUM_MIXTURE,
    batch_size=16,
    max_seq_len=32,
    learning_rate=1e-3,
    is_training=1,
  )


hps_model = default_hps()
hps_sample = hps_model._replace(batch_size=1, max_seq_len=1, is_training=0)


def one_hot_action(actions):
  actions = np.asarray(actions, dtype=np.int64)
  return np.eye(ACTION_SIZE, dtype=np.float32)[np.clip(actions, 0, ACTION_SIZE - 1)]


class GridcraftRNN:
  def __init__(self, hps=None, z_size=Z_SIZE, action_size=ACTION_SIZE, rnn_size=RNN_SIZE, num_mixture=NUM_MIXTURE, learning_rate=1e-3):
    if hps is not None:
      z_size = hps.z_size
      action_size = hps.action_size
      rnn_size = hps.rnn_size
      num_mixture = hps.num_mixture
      learning_rate = hps.learning_rate
    self.hps = hps or default_hps()
    self.z_size = z_size
    self.action_size = action_size
    self.rnn_size = rnn_size
    self.num_mixture = num_mixture
    self.input_size = z_size + action_size
    self.output_size = z_size * num_mixture * 3 + 2
    self.cell = tf.keras.layers.LSTMCell(rnn_size, name="rnn_cell")
    self.out = tf.keras.layers.Dense(self.output_size, name="rnn_out")
    self.optimizer = tf.keras.optimizers.Adam(learning_rate)
    self._build()

  def _build(self):
    x = tf.zeros((1, self.input_size), dtype=tf.float32)
    state = self.zero_state(batch_size=1)
    h, _ = self.cell(x, states=state)
    self.out(h)

  def zero_state(self, batch_size=1):
    return [
      tf.zeros((batch_size, self.rnn_size), dtype=tf.float32),
      tf.zeros((batch_size, self.rnn_size), dtype=tf.float32),
    ]

  def _split_output(self, raw):
    mdn = raw[:, :self.z_size * self.num_mixture * 3]
    reward = raw[:, -2]
    done_logit = raw[:, -1]
    mdn = tf.reshape(mdn, (-1, self.z_size, self.num_mixture * 3))
    logmix, mean, logstd = tf.split(mdn, 3, axis=2)
    logmix = tf.nn.log_softmax(logmix, axis=2)
    logstd = tf.clip_by_value(logstd, LOGSTD_MIN, LOGSTD_MAX)
    return logmix, mean, logstd, reward, done_logit

  def step(self, z, action, state):
    z = np.asarray(z, dtype=np.float32).reshape(1, self.z_size)
    action_oh = one_hot_action([int(action)]).reshape(1, self.action_size)
    x = tf.convert_to_tensor(np.concatenate([z, action_oh], axis=1), dtype=tf.float32)
    h, next_state = self.cell(x, states=state)
    raw = self.out(h)
    logmix, mean, logstd, reward, done_logit = self._split_output(raw)
    return (
      logmix.numpy()[0],
      mean.numpy()[0],
      logstd.numpy()[0],
      reward.numpy()[0],
      done_logit.numpy()[0],
      next_state,
    )

  def train_batch(self, z, action, reward, done):
    z = tf.convert_to_tensor(z, dtype=tf.float32)
    action_oh = tf.convert_to_tensor(one_hot_action(action), dtype=tf.float32)
    reward = tf.convert_to_tensor(reward, dtype=tf.float32)
    done = tf.convert_to_tensor(done.astype(np.float32), dtype=tf.float32)
    batch_size = tf.shape(z)[0]
    seq_len = z.shape[1] - 1
    inputs = tf.concat([z[:, :-1, :], action_oh[:, :-1, :]], axis=2)
    target_z = z[:, 1:, :]
    target_reward = reward[:, :-1]
    target_done = done[:, :-1]

    with tf.GradientTape() as tape:
      state = self.zero_state(batch_size=batch_size)
      z_losses = []
      reward_losses = []
      done_losses = []
      for t in range(seq_len):
        h, state = self.cell(inputs[:, t, :], states=state, training=True)
        raw = self.out(h, training=True)
        logmix, mean, logstd, pred_reward, done_logit = self._split_output(raw)
        target = tf.expand_dims(target_z[:, t, :], axis=2)
        log_prob = logmix - 0.5 * tf.square((target - mean) / tf.exp(logstd)) - logstd
        z_loss = -tf.reduce_mean(tf.reduce_logsumexp(log_prob, axis=2))
        reward_loss = tf.reduce_mean(tf.square(target_reward[:, t] - pred_reward))
        done_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
          labels=target_done[:, t], logits=done_logit))
        z_losses.append(z_loss)
        reward_losses.append(reward_loss)
        done_losses.append(done_loss)
      z_loss = tf.add_n(z_losses) / float(seq_len)
      reward_loss = tf.add_n(reward_losses) / float(seq_len)
      done_loss = tf.add_n(done_losses) / float(seq_len)
      loss = z_loss + reward_loss + done_loss
    variables = self.cell.trainable_variables + self.out.trainable_variables
    grads = tape.gradient(loss, variables)
    self.optimizer.apply_gradients(zip(grads, variables))
    return float(loss.numpy()), float(z_loss.numpy()), float(reward_loss.numpy()), float(done_loss.numpy())

  def _weights(self):
    return self.cell.get_weights() + self.out.get_weights()

  def save_json(self, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
      json.dump([w.tolist() for w in self._weights()], f)

  def load_json(self, path):
    with open(path) as f:
      params = [np.asarray(w, dtype=np.float32) for w in json.load(f)]
    self.cell.set_weights(params[:3])
    self.out.set_weights(params[3:])


def rnn_init_state(rnn):
  return rnn.zero_state(batch_size=1)


def rnn_next_state(rnn, z, action, state):
  _, _, _, _, _, next_state = rnn.step(z, action, state)
  return next_state


def rnn_output(state, z):
  hidden = state[0].numpy()[0]
  return np.concatenate([z, hidden], axis=0)


def rnn_output_size():
  return Z_SIZE + RNN_SIZE
