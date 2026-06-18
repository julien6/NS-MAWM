import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from worldmodels_compat import resize_image

from env import CarRacingWrapper

SCREEN_X = 64
SCREEN_Y = 64
FACTOR = 6.25


class CarRacingRenderWrapper(CarRacingWrapper):
  def __init__(self, render_mode=None):
    super().__init__(full_episode=False, render_mode=render_mode)
    self.vae_frame = None
    self.frame_count = 0

  def render(self, mode='human'):
    if mode == "rgb_array" and self.current_frame is not None and self.vae_frame is not None:
      img = np.concatenate((self.current_frame, self.vae_frame), axis=1)
      return resize_image(img, (int(np.round(SCREEN_Y * FACTOR)), int(np.round(SCREEN_X * FACTOR)) * 2))
    return self.env.render()


def make_env(env_name, seed=-1, render_mode=False):
  mode = "rgb_array"
  env = CarRacingRenderWrapper(render_mode=mode)
  if seed >= 0:
    env.seed(seed)
  return env
