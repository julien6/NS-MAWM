import json
import os

import numpy as np
import tensorflow as tf

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


class GridcraftBatchDreamEnv:
  def __init__(self, seed=None, max_steps=GRIDCRAFT_MAX_STEPS, rnn_path="rnn/rnn.json", initial_z_path="initial_z/initial_z.json", deterministic_z=True):
    self.max_steps = int(max_steps)
    self.rnn = GridcraftRNN()
    self.rnn.load_json(rnn_path)
    with open(initial_z_path) as f:
      self.initial_z = np.asarray(json.load(f), dtype=np.float32)
    self.deterministic_z = deterministic_z
    self.action_size = ACTION_SIZE
    self.obs_size = Z_SIZE
    self.seed(seed)

  def seed(self, seed=None):
    self.rng = np.random.default_rng(seed)
    return [seed]

  def reset(self, batch_size, seed=None):
    if seed is not None:
      self.seed(seed)
    self.batch_size = int(batch_size)
    idx = self.rng.integers(0, len(self.initial_z), size=self.batch_size)
    self.z = np.asarray(self.initial_z[idx], dtype=np.float32)
    self.rnn_state = rnn_init_state(self.rnn, batch_size=self.batch_size)
    self.frame_count = np.zeros((self.batch_size,), dtype=np.int32)
    self.done = np.zeros((self.batch_size,), dtype=np.bool_)
    return self.z

  def set_state(self, z, rnn_state, frame_count=None, done=None):
    self.z = np.asarray(z, dtype=np.float32).reshape(-1, Z_SIZE)
    self.batch_size = self.z.shape[0]
    self.rnn_state = [
      tf.convert_to_tensor(rnn_state[0], dtype=tf.float32),
      tf.convert_to_tensor(rnn_state[1], dtype=tf.float32),
    ]
    self.frame_count = np.zeros((self.batch_size,), dtype=np.int32) if frame_count is None else np.asarray(frame_count, dtype=np.int32)
    self.done = np.zeros((self.batch_size,), dtype=np.bool_) if done is None else np.asarray(done, dtype=np.bool_)

  def get_state(self):
    return (
      np.asarray(self.z, dtype=np.float32).copy(),
      [tf.identity(self.rnn_state[0]), tf.identity(self.rnn_state[1])],
      np.asarray(self.frame_count, dtype=np.int32).copy(),
      np.asarray(self.done, dtype=np.bool_).copy(),
    )

  def step(self, action_batch):
    action_batch = np.asarray(action_batch, dtype=np.int32).reshape(self.batch_size)
    next_z, reward, done_logit, self.rnn_state, _ = self.rnn.step_batch(
      self.z,
      action_batch,
      self.rnn_state,
      deterministic=self.deterministic_z,
      rng=self.rng,
    )
    self.z = next_z.numpy().astype(np.float32)
    self.frame_count += (~self.done).astype(np.int32)
    step_done = np.logical_or(done_logit.numpy() > 0, self.frame_count >= self.max_steps)
    self.done = np.logical_or(self.done, step_done)
    return self.z, reward.numpy().astype(np.float32), self.done.copy(), {}

  def close(self):
    pass


def make_env(seed=-1, render_mode=False, max_steps=GRIDCRAFT_MAX_STEPS):
  env = GridcraftDreamEnv(seed=seed if seed >= 0 else None, max_steps=max_steps)
  return env
