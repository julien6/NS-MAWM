import os
import sys

import gymnasium as gym
import numpy as np
import tensorflow as tf
from gymnasium import spaces

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from worldmodels_compat import SimpleImageViewer, process_frame, resize_image

import vizdoom as vzd

from doomrnn import ConvVAE, Model, hps_sample, model_path_name, model_rnn_size, model_state_space, reset_graph

SCREEN_Y = 64
SCREEN_X = 64


def _screen_to_hwc(frame):
  arr = np.asarray(frame)
  if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
    arr = np.transpose(arr[:3], (1, 2, 0))
  if arr.ndim == 2:
    arr = np.stack([arr, arr, arr], axis=2)
  return arr[:, :, :3]


def _process_frame(frame):
  return process_frame(_screen_to_hwc(frame), crop=np.s_[0:400, :, :], size=(SCREEN_Y, SCREEN_X))


class DoomTakeCoverWrapper(gym.Env):
  metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 50}

  def __init__(self, render_mode=False, load_model=True):
    self.render_mode = render_mode
    reset_graph()
    self.vae = ConvVAE(batch_size=1, gpu_mode=False, is_training=False, reuse=True)
    self.rnn = Model(hps_sample, gpu_mode=False)
    if load_model:
      self.vae.load_json(os.path.join(model_path_name, 'vae.json'))
      self.rnn.load_json(os.path.join(model_path_name, 'rnn.json'))

    self.game = self._make_game(render_mode)
    self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
    self.outwidth = self.rnn.hps.seq_width
    self.obs_size = self.outwidth + model_rnn_size * model_state_space
    self.observation_space = spaces.Box(low=-50., high=50., shape=(self.obs_size,), dtype=np.float32)
    self.actual_observation_space = spaces.Box(low=0, high=255, shape=(SCREEN_Y, SCREEN_X, 3), dtype=np.uint8)
    self.zero_state = self.rnn.zero_state(batch_size=1)
    self.viewer = None
    self.np_random = np.random.default_rng()
    self.reset()

  def _make_game(self, render_mode):
    game = vzd.DoomGame()
    config_path = os.path.join(vzd.scenarios_path, "take_cover.cfg")
    game.load_config(config_path)
    game.set_screen_format(vzd.ScreenFormat.RGB24)
    game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
    game.set_window_visible(bool(render_mode))
    game.set_sound_enabled(False)
    game.init()
    return game

  def seed(self, seed=None):
    self.np_random = np.random.default_rng(seed)
    tf.random.set_seed(seed or 0)
    try:
      self.game.set_seed(int(seed or 0))
    except Exception:
      pass
    return [seed]

  def _action_for(self, action):
    action_value = float(np.asarray(action).reshape(-1)[0])
    buttons = [str(button).split(".")[-1] for button in self.game.get_available_buttons()]
    result = [0] * len(buttons)
    threshold = 0.3333
    for idx, name in enumerate(buttons):
      if "MOVE_LEFT" in name and action_value < -threshold:
        result[idx] = 1
      if "MOVE_RIGHT" in name and action_value > threshold:
        result[idx] = 1
    return result

  def _encode(self, img):
    simple_obs = np.copy(img).astype(float) / 255.0
    simple_obs = simple_obs.reshape(1, 64, 64, 3)
    mu, logvar = self.vae.encode_mu_logvar(simple_obs)
    return (mu + np.exp(logvar / 2.0) * self.np_random.standard_normal(logvar.shape))[0]

  def _decode(self, z):
    img = self.vae.decode(z.reshape(1, 64)) * 255.0
    return np.round(img).astype(np.uint8).reshape(64, 64, 3)

  def reset(self, *, seed=None, options=None):
    if seed is not None:
      self.seed(seed)
    self.game.new_episode()
    state = self.game.get_state()
    frame = state.screen_buffer if state is not None else np.zeros((480, 640, 3), dtype=np.uint8)
    self.current_obs = _process_frame(frame)
    self.rnn_state = self.zero_state
    self.z = self._encode(self.current_obs)
    self.restart = 1
    self.frame_count = 0
    return self._current_state()

  def _current_state(self):
    if model_state_space == 2:
      return np.concatenate([self.z, self.rnn_state.c.flatten(), self.rnn_state.h.flatten()], axis=0)
    return np.concatenate([self.z, self.rnn_state.h.flatten()], axis=0)

  def step(self, action):
    self.frame_count += 1
    action_value = float(np.asarray(action).reshape(-1)[0])
    _, _, _, _, self.rnn_state = self.rnn.step(
      self.z.reshape(1, self.outwidth),
      np.array([[action_value]], dtype=np.float32),
      np.array([[self.restart]], dtype=np.float32),
      self.rnn_state,
    )

    reward = self.game.make_action(self._action_for(action))
    done = self.game.is_episode_finished()
    if done:
      frame = np.zeros((480, 640, 3), dtype=np.uint8)
      self.restart = 1
    else:
      state = self.game.get_state()
      frame = state.screen_buffer if state is not None else np.zeros((480, 640, 3), dtype=np.uint8)
      self.restart = 0
    self.current_obs = _process_frame(frame)
    self.z = self._encode(self.current_obs)
    return self._current_state(), reward, done, {}

  def render(self, mode='human'):
    state = self.game.get_state()
    img = _screen_to_hwc(state.screen_buffer) if state is not None else np.zeros((480, 640, 3), dtype=np.uint8)
    small_img = resize_image(self.current_obs, (img.shape[0], img.shape[0]))
    vae_img = resize_image(self._decode(self.z), (img.shape[0], img.shape[0]))
    all_img = np.concatenate((img, small_img, vae_img), axis=1)
    if mode == 'rgb_array':
      return all_img
    if self.viewer is None:
      self.viewer = SimpleImageViewer()
    self.viewer.imshow(all_img)

  def close(self):
    if self.viewer is not None:
      self.viewer.close()
      self.viewer = None
    if getattr(self, "game", None) is not None:
      self.game.close()


def make_env(env_name="doom", seed=-1, render_mode=False):
  env = DoomTakeCoverWrapper(render_mode=render_mode)
  if seed >= 0:
    env.seed(seed)
  return env
