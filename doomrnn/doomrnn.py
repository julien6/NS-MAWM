import json
import os
import sys
from collections import namedtuple

import gymnasium as gym
import numpy as np
import tensorflow as tf
from gymnasium import spaces

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from worldmodels_compat import LegacyLSTMState, SimpleImageViewer, legacy_lstm_step, resize_image, softmax_rows

model_path_name = 'tf_models'
model_rnn_size = 512
model_num_mixture = 5
model_restart_factor = 10.
model_state_space = 2

TEMPERATURE = 1.25

HyperParams = namedtuple('HyperParams', ['max_seq_len',
                                         'seq_width',
                                         'rnn_size',
                                         'batch_size',
                                         'grad_clip',
                                         'num_mixture',
                                         'restart_factor',
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
  return HyperParams(max_seq_len=1000,
                     seq_width=64,
                     rnn_size=model_rnn_size,
                     batch_size=100,
                     grad_clip=1.0,
                     num_mixture=model_num_mixture,
                     restart_factor=model_restart_factor,
                     learning_rate=0.001,
                     decay_rate=0.99999,
                     min_learning_rate=0.00001,
                     use_layer_norm=0,
                     use_recurrent_dropout=0,
                     recurrent_dropout_prob=0.90,
                     use_input_dropout=0,
                     input_dropout_prob=0.90,
                     use_output_dropout=0,
                     output_dropout_prob=0.90,
                     is_training=0)


hps_model = default_hps()
hps_sample = hps_model._replace(batch_size=1, max_seq_len=2, use_recurrent_dropout=0, is_training=0)


def reset_graph():
  tf.keras.backend.clear_session()


class ConvVAE(object):
  def __init__(self, z_size=64, batch_size=100, learning_rate=0.0001, kl_tolerance=0.5, is_training=True, reuse=False, gpu_mode=True):
    self.z_size = z_size
    self.batch_size = batch_size
    self.learning_rate = learning_rate
    self.is_training = is_training
    self.kl_tolerance = kl_tolerance
    self.reuse = reuse
    self._build_model()

  def _build_model(self):
    inputs = tf.keras.Input(shape=(64, 64, 3), name="x")
    h = tf.keras.layers.Conv2D(32, 4, strides=2, activation="relu", padding="valid", name="enc_conv1")(inputs)
    h = tf.keras.layers.Conv2D(64, 4, strides=2, activation="relu", padding="valid", name="enc_conv2")(h)
    h = tf.keras.layers.Conv2D(128, 4, strides=2, activation="relu", padding="valid", name="enc_conv3")(h)
    h = tf.keras.layers.Conv2D(256, 4, strides=2, activation="relu", padding="valid", name="enc_conv4")(h)
    h = tf.keras.layers.Reshape((2 * 2 * 256,), name="enc_flatten")(h)
    mu = tf.keras.layers.Dense(self.z_size, name="enc_fc_mu")(h)
    logvar = tf.keras.layers.Dense(self.z_size, name="enc_fc_log_var")(h)
    self.encoder = tf.keras.Model(inputs, [mu, logvar], name="conv_vae_encoder")

    z_inputs = tf.keras.Input(shape=(self.z_size,), name="z")
    h = tf.keras.layers.Dense(4 * 256, name="dec_fc")(z_inputs)
    h = tf.keras.layers.Reshape((1, 1, 4 * 256), name="dec_reshape")(h)
    h = tf.keras.layers.Conv2DTranspose(128, 5, strides=2, activation="relu", padding="valid", name="dec_deconv1")(h)
    h = tf.keras.layers.Conv2DTranspose(64, 5, strides=2, activation="relu", padding="valid", name="dec_deconv2")(h)
    h = tf.keras.layers.Conv2DTranspose(32, 6, strides=2, activation="relu", padding="valid", name="dec_deconv3")(h)
    y = tf.keras.layers.Conv2DTranspose(3, 6, strides=2, activation="sigmoid", padding="valid", name="dec_deconv4")(h)
    self.decoder = tf.keras.Model(z_inputs, y, name="conv_vae_decoder")
    self.model = tf.keras.Model(inputs, self.decoder(mu), name="conv_vae")

  @property
  def _ordered_layers(self):
    return [
      self.encoder.get_layer("enc_conv1"),
      self.encoder.get_layer("enc_conv2"),
      self.encoder.get_layer("enc_conv3"),
      self.encoder.get_layer("enc_conv4"),
      self.encoder.get_layer("enc_fc_mu"),
      self.encoder.get_layer("enc_fc_log_var"),
      self.decoder.get_layer("dec_fc"),
      self.decoder.get_layer("dec_deconv1"),
      self.decoder.get_layer("dec_deconv2"),
      self.decoder.get_layer("dec_deconv3"),
      self.decoder.get_layer("dec_deconv4"),
    ]

  def close_sess(self):
    pass

  def encode(self, x):
    mu, logvar = self.encode_mu_logvar(x)
    return mu + np.exp(logvar / 2.0) * np.random.randn(*logvar.shape)

  def encode_mu_logvar(self, x):
    mu, logvar = self.encoder(np.asarray(x, dtype=np.float32), training=False)
    return mu.numpy(), logvar.numpy()

  def decode(self, z):
    return self.decoder(np.asarray(z, dtype=np.float32), training=False).numpy()

  def get_model_params(self):
    model_params = []
    model_shapes = []
    model_names = []
    for layer in self._ordered_layers:
      weights = layer.get_weights()
      if len(weights) != 2:
        continue
      for name, value in [(layer.name + "/kernel:0", weights[0]), (layer.name + "/bias:0", weights[1])]:
        model_names.append(name)
        model_params.append(np.round(value * 10000).astype(int).tolist())
        model_shapes.append(value.shape)
    return model_params, model_shapes, model_names

  def get_random_model_params(self, stdev=0.5):
    _, mshape, _ = self.get_model_params()
    return [np.random.standard_cauchy(s) * stdev for s in mshape]

  def set_model_params(self, params):
    idx = 0
    for layer in self._ordered_layers:
      weights = layer.get_weights()
      if len(weights) != 2:
        continue
      kernel = np.asarray(params[idx], dtype=np.float32) / 10000.0
      bias = np.asarray(params[idx + 1], dtype=np.float32) / 10000.0
      assert kernel.shape == weights[0].shape, "inconsistent kernel shape"
      assert bias.shape == weights[1].shape, "inconsistent bias shape"
      layer.set_weights([kernel, bias])
      idx += 2

  def load_json(self, jsonfile='vae.json'):
    with open(jsonfile, 'r') as f:
      self.set_model_params(json.load(f))

  def save_json(self, jsonfile='vae.json'):
    model_params, _, _ = self.get_model_params()
    with open(jsonfile, 'wt') as outfile:
      json.dump(model_params, outfile, sort_keys=True, indent=0, separators=(',', ': '))

  def set_random_params(self, stdev=0.5):
    self.set_model_params(self.get_random_model_params(stdev))


class Model():
  def __init__(self, hps, gpu_mode=True, reuse=False):
    self.hps = hps
    self.num_mixture = hps.num_mixture
    self.lstm_kernel = np.zeros((hps.seq_width + 2 + hps.rnn_size, hps.rnn_size * 4), dtype=np.float32)
    self.lstm_bias = np.zeros((hps.rnn_size * 4,), dtype=np.float32)
    self.output_w = np.zeros((hps.rnn_size, hps.seq_width * hps.num_mixture * 3 + 1), dtype=np.float32)
    self.output_b = np.zeros((hps.seq_width * hps.num_mixture * 3 + 1,), dtype=np.float32)

  def zero_state(self, batch_size=None):
    batch = batch_size or self.hps.batch_size
    return LegacyLSTMState(np.zeros((batch, self.hps.rnn_size), dtype=np.float32),
                           np.zeros((batch, self.hps.rnn_size), dtype=np.float32))

  def close_sess(self):
    pass

  def step(self, z, action, restart, prev_state):
    z = np.asarray(z, dtype=np.float32).reshape(-1, self.hps.seq_width)
    action = np.asarray(action, dtype=np.float32).reshape(-1, 1)
    restart = np.asarray(restart, dtype=np.float32).reshape(-1, 1)
    c = np.where(restart > 0.5, np.zeros_like(prev_state.c), prev_state.c)
    h = np.where(restart > 0.5, np.zeros_like(prev_state.h), prev_state.h)
    x = np.concatenate([z, action, restart], axis=1)
    h_out, next_state = legacy_lstm_step(x, LegacyLSTMState(c, h), self.lstm_kernel, self.lstm_bias)
    output = np.matmul(h_out, self.output_w) + self.output_b
    restart_logits = output[:, 0]
    mdn = output[:, 1:].reshape(-1, self.hps.seq_width, self.hps.num_mixture * 3)[0]
    logmix, mean, logstd = np.split(mdn, 3, axis=1)
    logmix = np.log(softmax_rows(logmix) + 1e-8)
    return logmix, mean, logstd, restart_logits, next_state

  def get_model_params(self):
    params = [self.lstm_kernel, self.lstm_bias, self.output_w, self.output_b]
    names = ["RNN/lstm/kernel:0", "RNN/lstm/bias:0", "RNN/output_w:0", "RNN/output_b:0"]
    return [np.round(p * 10000).astype(int).tolist() for p in params], [p.shape for p in params], names

  def set_model_params(self, params):
    arrays = [np.asarray(p, dtype=np.float32) / 10000.0 for p in params]
    # Historical Doom JSON order is LSTM kernel/bias then output projection.
    if arrays[0].shape == self.output_w.shape:
      arrays = [arrays[2], arrays[3], arrays[0], arrays[1]]
    expected = [self.lstm_kernel.shape, self.lstm_bias.shape, self.output_w.shape, self.output_b.shape]
    for arr, shape in zip(arrays, expected):
      assert arr.shape == shape, "inconsistent shape"
    self.lstm_kernel, self.lstm_bias, self.output_w, self.output_b = arrays

  def get_random_model_params(self, stdev=0.5):
    _, mshape, _ = self.get_model_params()
    return [np.random.standard_cauchy(s) * stdev for s in mshape]

  def set_random_params(self, stdev=0.5):
    self.set_model_params(self.get_random_model_params(stdev))

  def load_json(self, jsonfile='rnn.json'):
    with open(jsonfile, 'r') as f:
      self.set_model_params(json.load(f))

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


def sigmoid(x):
  return 1 / (1 + np.exp(-x))


class DoomCoverRNNEnv(gym.Env):
  metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 50}

  def __init__(self, render_mode=False, load_model=True):
    self.render_mode = render_mode
    with open(os.path.join(model_path_name, 'initial_z.json'), 'r') as f:
      [initial_mu, initial_logvar] = json.load(f)
    self.initial_mu_logvar = [list(elem) for elem in zip(initial_mu, initial_logvar)]

    reset_graph()
    self.vae = ConvVAE(batch_size=1, gpu_mode=False, is_training=False, reuse=True)
    self.rnn = Model(hps_sample, gpu_mode=False)
    if load_model:
      self.vae.load_json(os.path.join(model_path_name, 'vae.json'))
      self.rnn.load_json(os.path.join(model_path_name, 'rnn.json'))

    self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
    self.outwidth = self.rnn.hps.seq_width
    self.obs_size = self.outwidth + model_rnn_size * model_state_space
    self.observation_space = spaces.Box(low=-50., high=50., shape=(self.obs_size,), dtype=np.float32)
    self.zero_state = self.rnn.zero_state(batch_size=1)
    self.np_random = np.random.default_rng()
    self.viewer = None
    self.max_frame = 2100
    self.reset()

  def seed(self, seed=None):
    self.np_random = np.random.default_rng(seed)
    tf.random.set_seed(seed or 0)
    return [seed]

  def _sample_init_z(self):
    idx = self.np_random.integers(0, len(self.initial_mu_logvar))
    init_mu, init_logvar = self.initial_mu_logvar[idx]
    init_mu = np.array(init_mu) / 10000.0
    init_logvar = np.array(init_logvar) / 10000.0
    return init_mu + np.exp(init_logvar / 2.0) * self.np_random.standard_normal(init_logvar.shape)

  def _current_state(self):
    if model_state_space == 2:
      return np.concatenate([self.z, self.rnn_state.c.flatten(), self.rnn_state.h.flatten()], axis=0)
    return np.concatenate([self.z, self.rnn_state.h.flatten()], axis=0)

  def reset(self, *, seed=None, options=None):
    if seed is not None:
      self.seed(seed)
    self.temperature = TEMPERATURE
    self.rnn_state = self.zero_state
    self.z = self._sample_init_z()
    self.restart = 1
    self.frame_count = 0
    return self._current_state()

  def step(self, action):
    self.frame_count += 1
    action_value = float(np.asarray(action).reshape(-1)[0])
    logmix, mean, logstd, logrestart, next_state = self.rnn.step(
      self.z.reshape(1, self.outwidth),
      np.array([[action_value]], dtype=np.float32),
      np.array([[self.restart]], dtype=np.float32),
      self.rnn_state,
    )

    logmix2 = np.copy(logmix) / self.temperature
    logmix2 -= logmix2.max()
    logmix2 = np.exp(logmix2)
    logmix2 /= logmix2.sum(axis=1).reshape(self.outwidth, 1)

    chosen_mean = np.zeros(self.outwidth)
    chosen_logstd = np.zeros(self.outwidth)
    for j in range(self.outwidth):
      idx = get_pi_idx(self.np_random.random(), logmix2[j])
      chosen_mean[j] = mean[j][idx]
      chosen_logstd[j] = logstd[j][idx]
    rand_gaussian = self.np_random.standard_normal(self.outwidth) * np.sqrt(self.temperature)
    self.z = chosen_mean + np.exp(chosen_logstd) * rand_gaussian
    self.restart = 1 if logrestart[0] > 0 else 0
    self.rnn_state = next_state
    done = bool(self.restart or self.frame_count >= self.max_frame)
    return self._current_state(), 1, done, {}

  def _get_image(self, upsize=False):
    img = self.vae.decode(self.z.reshape(1, 64)) * 255.0
    img = np.round(img).astype(np.uint8).reshape(64, 64, 3)
    if upsize:
      img = resize_image(img, (640, 640))
    return img

  def render(self, mode='human'):
    if not self.render_mode and mode != 'rgb_array':
      return None
    img = self._get_image(upsize=True)
    if mode == 'rgb_array':
      return img
    if self.viewer is None:
      self.viewer = SimpleImageViewer()
    self.viewer.imshow(img)

  def close(self):
    if self.viewer is not None:
      self.viewer.close()
      self.viewer = None


if __name__ == "__main__":
  env = DoomCoverRNNEnv(render_mode=True)
  for i in range(3):
    env.reset()
    total_reward = 0
    for t in range(200):
      _, reward, done, _ = env.step(np.array([0.0], dtype=np.float32))
      total_reward += reward
      env.render()
      if done:
        break
    print("episode", i, "reward", total_reward, "timesteps", t)
