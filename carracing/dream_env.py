import json
import os
import sys

import gymnasium as gym
import numpy as np
from gymnasium import spaces

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from worldmodels_compat import SimpleImageViewer, resize_image

SCREEN_X = 64
SCREEN_Y = 64
FACTOR = 8

with open(os.path.join('initial_z', 'initial_z.json'), 'r') as f:
  [initial_mu, initial_logvar] = json.load(f)

initial_mu_logvar = [list(elem) for elem in zip(initial_mu, initial_logvar)]


def get_pi_idx(x, pdf):
  accumulate = 0
  for i in range(0, pdf.size):
    accumulate += pdf[i]
    if accumulate >= x:
      return i
  return pdf.size - 1


class CarRacingDream(gym.Env):
  metadata = {
      'render_modes': ['human', 'rgb_array'],
      'render_fps': 60
  }

  def __init__(self, agent):
    self.observation_space = spaces.Box(low=-50., high=50., shape=(32,), dtype=np.float32)
    self.action_space = spaces.Box(low=np.array([-1.0, 0.0, 0.0], dtype=np.float32),
                                   high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
                                   dtype=np.float32)
    self.np_random = np.random.default_rng()
    self.agent = agent
    self.vae = agent.vae
    self.rnn = agent.rnn
    self.z_size = self.rnn.hps.output_seq_width
    self.viewer = None
    self.frame_count = 0
    self.z = None
    self.temperature = 0.7
    self.reset()

  def seed(self, seed=None):
    self.np_random = np.random.default_rng(seed)
    return [seed]

  def _sample_z(self, mu, logvar):
    return mu + np.exp(logvar / 2.0) * self.np_random.standard_normal(logvar.shape)

  def reset(self, *, seed=None, options=None):
    if seed is not None:
      self.seed(seed)
    idx = self.np_random.integers(0, len(initial_mu_logvar))
    init_mu, init_logvar = initial_mu_logvar[idx]
    init_mu = np.array(init_mu) / 10000.0
    init_logvar = np.array(init_logvar) / 10000.0
    self.z = self._sample_z(init_mu, init_logvar)
    self.frame_count = 0
    return self.z

  def _sample_next_z(self, action):
    outwidth = self.rnn.hps.output_seq_width
    prev_x = np.zeros((1, 1, outwidth), dtype=np.float32)
    prev_x[0][0] = self.z
    input_x = np.concatenate((prev_x, np.asarray(action).reshape(1, 1, 3)), axis=2)
    logmix, mean, logstd, self.agent.state = self.rnn.step(input_x, self.agent.state)

    logmix2 = np.copy(logmix) / self.temperature
    logmix2 -= logmix2.max()
    logmix2 = np.exp(logmix2)
    logmix2 /= logmix2.sum(axis=1).reshape(outwidth, 1)

    chosen_mean = np.zeros(outwidth)
    chosen_logstd = np.zeros(outwidth)
    for j in range(outwidth):
      idx = get_pi_idx(self.np_random.random(), logmix2[j])
      chosen_mean[j] = mean[j][idx]
      chosen_logstd[j] = logstd[j][idx]

    rand_gaussian = self.np_random.standard_normal(outwidth) * np.sqrt(self.temperature)
    return chosen_mean + np.exp(chosen_logstd) * rand_gaussian

  def step(self, action):
    self.frame_count += 1
    self.z = self._sample_next_z(action)
    done = self.frame_count > 1200
    return self.z, 0, done, {}

  def decode_obs(self, z):
    img = self.vae.decode(z.reshape(1, self.z_size)) * 255.0
    return np.round(img).astype(np.uint8).reshape(64, 64, 3)

  def render(self, mode='human'):
    img = self.decode_obs(self.z)
    img = resize_image(img, (int(np.round(SCREEN_Y * FACTOR)), int(np.round(SCREEN_X * FACTOR))))
    if mode == 'rgb_array':
      return img
    if self.viewer is None:
      self.viewer = SimpleImageViewer()
    self.viewer.imshow(img)

  def close(self):
    if self.viewer is not None:
      self.viewer.close()
      self.viewer = None


def make_env(env_name, agent, seed=-1, render_mode=False):
  env = CarRacingDream(agent)
  if seed < 0:
    seed = np.random.randint(2**31 - 1)
  env.seed(seed)
  return env
