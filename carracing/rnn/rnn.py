import json
import os
import sys
from collections import namedtuple

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from worldmodels_compat import LegacyLSTMState, legacy_lstm_step, softmax_rows


MODE_ZCH = 0
MODE_ZC = 1
MODE_Z = 2
MODE_Z_HIDDEN = 3
MODE_ZH = 4

HyperParams = namedtuple('HyperParams', ['num_steps',
                                         'max_seq_len',
                                         'input_seq_width',
                                         'output_seq_width',
                                         'rnn_size',
                                         'batch_size',
                                         'grad_clip',
                                         'num_mixture',
                                         'learning_rate',
                                         'decay_rate',
                                         'min_learning_rate',
                                         'use_layer_norm',
                                         'use_recurrent_dropout',
                                         'recurrent_dropout_prob',
                                         'use_input_dropout',
                                         'input_dropout_prob',
                                         'use_output_dropout',
                                         'output_dropout_prob',
                                         'is_training',
                                        ])


def default_hps():
  return HyperParams(num_steps=2000,
                     max_seq_len=1000,
                     input_seq_width=35,
                     output_seq_width=32,
                     rnn_size=256,
                     batch_size=100,
                     grad_clip=1.0,
                     num_mixture=5,
                     learning_rate=0.001,
                     decay_rate=1.0,
                     min_learning_rate=0.00001,
                     use_layer_norm=0,
                     use_recurrent_dropout=0,
                     recurrent_dropout_prob=0.90,
                     use_input_dropout=0,
                     input_dropout_prob=0.90,
                     use_output_dropout=0,
                     output_dropout_prob=0.90,
                     is_training=1)


hps_model = default_hps()
hps_sample = hps_model._replace(batch_size=1, max_seq_len=1, use_recurrent_dropout=0, is_training=0)


class MDNRNN():
  def __init__(self, hps, gpu_mode=True, reuse=False):
    self.hps = hps
    self.num_mixture = hps.num_mixture
    self.output_w = np.zeros((hps.rnn_size, hps.output_seq_width * hps.num_mixture * 3), dtype=np.float32)
    self.output_b = np.zeros((hps.output_seq_width * hps.num_mixture * 3,), dtype=np.float32)
    self.lstm_kernel = np.zeros((hps.input_seq_width + hps.rnn_size, hps.rnn_size * 4), dtype=np.float32)
    self.lstm_bias = np.zeros((hps.rnn_size * 4,), dtype=np.float32)

  def zero_state(self, batch_size=None):
    batch = batch_size or self.hps.batch_size
    return LegacyLSTMState(np.zeros((batch, self.hps.rnn_size), dtype=np.float32),
                           np.zeros((batch, self.hps.rnn_size), dtype=np.float32))

  def close_sess(self):
    pass

  def step(self, input_x, prev_state):
    x = np.asarray(input_x, dtype=np.float32).reshape(-1, self.hps.input_seq_width)
    h, next_state = legacy_lstm_step(x, prev_state, self.lstm_kernel, self.lstm_bias)
    output = np.matmul(h, self.output_w) + self.output_b
    output = output.reshape(-1, self.hps.output_seq_width, self.hps.num_mixture * 3)
    output = output[0].reshape(-1, self.hps.num_mixture * 3)
    logmix, mean, logstd = np.split(output, 3, axis=1)
    logmix = np.log(softmax_rows(logmix) + 1e-8)
    return logmix, mean, logstd, next_state

  def get_model_params(self):
    params = [self.output_w, self.output_b, self.lstm_kernel, self.lstm_bias]
    names = ["RNN/output_w:0", "RNN/output_b:0", "RNN/lstm/kernel:0", "RNN/lstm/bias:0"]
    return [np.round(p * 10000).astype(int).tolist() for p in params], [p.shape for p in params], names

  def get_random_model_params(self, stdev=0.5):
    _, mshape, _ = self.get_model_params()
    return [np.random.standard_cauchy(s) * stdev for s in mshape]

  def set_random_params(self, stdev=0.5):
    self.set_model_params(self.get_random_model_params(stdev))

  def set_model_params(self, params):
    arrays = [np.asarray(p, dtype=np.float32) / 10000.0 for p in params]
    expected = [self.output_w.shape, self.output_b.shape, self.lstm_kernel.shape, self.lstm_bias.shape]
    for arr, shape in zip(arrays, expected):
      assert arr.shape == shape, "inconsistent shape"
    self.output_w, self.output_b, self.lstm_kernel, self.lstm_bias = arrays

  def load_json(self, jsonfile='rnn.json'):
    with open(jsonfile, 'r') as f:
      params = json.load(f)
    self.set_model_params(params)

  def save_json(self, jsonfile='rnn.json'):
    model_params, _, _ = self.get_model_params()
    with open(jsonfile, 'wt') as outfile:
      json.dump(model_params, outfile, sort_keys=True, indent=0, separators=(',', ': '))


def get_pi_idx(x, pdf):
  accumulate = 0
  for i in range(0, pdf.size):
    accumulate += pdf[i]
    if accumulate >= x:
      return i
  return pdf.size - 1


def sample_sequence(sess, s_model, hps, init_z, actions, temperature=1.0, seq_len=1000):
  prev_x = np.zeros((1, 1, hps.output_seq_width), dtype=np.float32)
  prev_x[0][0] = init_z
  prev_state = rnn_init_state(s_model)
  strokes = np.zeros((seq_len, hps.output_seq_width), dtype=np.float32)
  for i in range(seq_len):
    input_x = np.concatenate((prev_x, actions[i].reshape((1, 1, 3))), axis=2)
    logmix, mean, logstd, next_state = s_model.step(input_x, prev_state)
    logmix2 = np.copy(logmix) / temperature
    logmix2 -= logmix2.max()
    logmix2 = np.exp(logmix2)
    logmix2 /= logmix2.sum(axis=1).reshape(hps.output_seq_width, 1)
    chosen_mean = np.zeros(hps.output_seq_width)
    chosen_logstd = np.zeros(hps.output_seq_width)
    for j in range(hps.output_seq_width):
      idx = get_pi_idx(np.random.rand(), logmix2[j])
      chosen_mean[j] = mean[j][idx]
      chosen_logstd[j] = logstd[j][idx]
    next_x = chosen_mean + np.exp(chosen_logstd) * np.random.randn(hps.output_seq_width) * np.sqrt(temperature)
    strokes[i, :] = next_x
    prev_x[0][0] = next_x
    prev_state = next_state
  return strokes


def rnn_init_state(rnn):
  return rnn.zero_state(batch_size=1)


def rnn_next_state(rnn, z, a, prev_state):
  input_x = np.concatenate((z.reshape((1, 1, 32)), a.reshape((1, 1, 3))), axis=2)
  _, _, _, next_state = rnn.step(input_x, prev_state)
  return next_state


def rnn_output_size(mode):
  if mode == MODE_ZCH:
    return 32 + 256 + 256
  if (mode == MODE_ZC) or (mode == MODE_ZH):
    return 32 + 256
  return 32


def rnn_output(state, z, mode):
  if mode == MODE_ZCH:
    return np.concatenate([z, np.concatenate((state.c, state.h), axis=1)[0]])
  if mode == MODE_ZC:
    return np.concatenate([z, state.c[0]])
  if mode == MODE_ZH:
    return np.concatenate([z, state.h[0]])
  return z
