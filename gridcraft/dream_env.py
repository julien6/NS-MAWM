import json
import os

import numpy as np

from exp_config import ACTION_SIZE, GRIDCRAFT_MAX_STEPS, Z_SIZE
from rnn.rnn import GridcraftRNN, rnn_init_state


def get_pi_idx(x, pdf):
  acc = 0.0
  for i in range(pdf.size):
    acc += pdf[i]
    if acc >= x:
      return i
  return pdf.size - 1


class GridcraftDreamEnv:
  def __init__(self, seed=None, max_steps=GRIDCRAFT_MAX_STEPS, rnn_path="rnn/rnn.json", initial_z_path="initial_z/initial_z.json"):
    self.max_steps = max_steps
    self.rnn = GridcraftRNN()
    self.rnn.load_json(rnn_path)
    with open(initial_z_path) as f:
      self.initial_z = np.asarray(json.load(f), dtype=np.float32)
    self.rng = np.random.default_rng(seed)
    self.action_size = ACTION_SIZE
    self.obs_size = Z_SIZE
    self.seed(seed)

  def seed(self, seed=None):
    self.rng = np.random.default_rng(seed)
    return [seed]

  def reset(self, seed=None):
    if seed is not None:
      self.seed(seed)
    idx = int(self.rng.integers(0, len(self.initial_z)))
    self.z = np.asarray(self.initial_z[idx], dtype=np.float32)
    self.rnn_state = rnn_init_state(self.rnn)
    self.frame_count = 0
    return self.z

  def step(self, action):
    self.frame_count += 1
    logmix, mean, logstd, reward, done_logit, self.rnn_state = self.rnn.step(self.z, int(action), self.rnn_state)
    mix = np.exp(logmix - np.max(logmix, axis=1, keepdims=True))
    mix = mix / np.sum(mix, axis=1, keepdims=True)
    next_z = np.zeros((Z_SIZE,), dtype=np.float32)
    for j in range(Z_SIZE):
      idx = get_pi_idx(self.rng.random(), mix[j])
      next_z[j] = mean[j][idx] + np.exp(logstd[j][idx]) * self.rng.standard_normal()
    self.z = next_z
    done = bool(done_logit > 0 or self.frame_count >= self.max_steps)
    return self.z, float(reward), done, {}

  def render(self):
    return None

  def close(self):
    pass


def make_env(seed=-1, render_mode=False, max_steps=GRIDCRAFT_MAX_STEPS):
  env = GridcraftDreamEnv(seed=seed if seed >= 0 else None, max_steps=max_steps)
  return env
