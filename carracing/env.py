import os
import sys

import gymnasium as gym
import numpy as np
from gymnasium import spaces

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from worldmodels_compat import SimpleImageViewer, process_frame, reset_env, step_env

SCREEN_X = 64
SCREEN_Y = 64


def _process_frame(frame):
  return process_frame(frame, crop=np.s_[0:84, :, :], size=(64, 64))


class CarRacingWrapper(gym.Wrapper):
  def __init__(self, full_episode=False, render_mode=None):
    env = None
    for env_id in ("CarRacing-v3", "CarRacing-v2"):
      try:
        env = gym.make(env_id, render_mode=render_mode)
        break
      except Exception:
        continue
    if env is None:
      env = gym.make("CarRacing-v2", render_mode=render_mode)
    super().__init__(env)
    self.full_episode = full_episode
    self.observation_space = spaces.Box(low=0, high=255, shape=(SCREEN_X, SCREEN_Y, 3), dtype=np.uint8)
    self._seed = None
    self.current_frame = None
    self.viewer = None
    self.requested_render_mode = render_mode

  def seed(self, seed=None):
    self._seed = seed
    self.action_space.seed(seed)
    return [seed]

  def reset(self, **kwargs):
    if "seed" not in kwargs and self._seed is not None:
      kwargs["seed"] = self._seed
      self._seed = None
    obs = reset_env(self.env, kwargs.get("seed"))
    self.current_frame = _process_frame(obs)
    return self.current_frame

  def step(self, action):
    obs, reward, done, info = step_env(self.env, np.asarray(action, dtype=np.float32))
    if self.full_episode:
      done = False
    self.current_frame = _process_frame(obs)
    return self.current_frame, reward, done, info

  def render(self, mode='human'):
    frame = self.env.render()
    if mode == 'rgb_array':
      return frame
    if frame is None:
      return None
    if self.viewer is None:
      self.viewer = SimpleImageViewer()
    self.viewer.imshow(frame)
    return None

  def close(self):
    if self.viewer is not None:
      self.viewer.close()
      self.viewer = None
    return self.env.close()


def make_env(env_name, seed=-1, render_mode=False, full_episode=False):
  # The native Gymnasium human renderer can segfault when TensorFlow has already
  # initialized native libraries. Render rgb_array and display it ourselves.
  mode = "rgb_array"
  env = CarRacingWrapper(full_episode=full_episode, render_mode=mode)
  if seed >= 0:
    env.seed(seed)
  return env


if __name__ == "__main__":
  from pygame.locals import K_DOWN, K_LEFT, K_RIGHT, K_UP, KEYDOWN, KEYUP, QUIT
  import pygame

  env = make_env("carracing", render_mode=True)
  action = np.array([0.0, 0.0, 0.0], dtype=np.float32)
  pygame.init()
  while True:
    obs = env.reset()
    total_reward = 0.0
    for steps in range(1000):
      for event in pygame.event.get():
        if event.type == QUIT:
          raise SystemExit
        if event.type in (KEYDOWN, KEYUP):
          pressed = event.type == KEYDOWN
          if event.key == K_LEFT:
            action[0] = -1.0 if pressed else 0.0
          if event.key == K_RIGHT:
            action[0] = 1.0 if pressed else 0.0
          if event.key == K_UP:
            action[1] = 1.0 if pressed else 0.0
          if event.key == K_DOWN:
            action[2] = 0.8 if pressed else 0.0
      obs, reward, done, info = env.step(action)
      total_reward += reward
      env.render()
      if steps % 200 == 0 or done:
        print("step", steps, "total_reward", total_reward)
      if done:
        break
