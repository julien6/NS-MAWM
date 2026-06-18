import os
import time
from collections import namedtuple

import numpy as np
from PIL import Image


LegacyLSTMState = namedtuple("LegacyLSTMState", ["c", "h"])


def resize_image(image, size):
  """Resize an image using Pillow with the old scipy.misc.imresize size order."""
  arr = np.asarray(image)
  if len(size) != 2:
    raise ValueError("size must be (height, width)")
  height, width = int(size[0]), int(size[1])
  if arr.dtype != np.uint8:
    arr = np.clip(arr * 255.0 if arr.max(initial=0) <= 1.0 else arr, 0, 255).astype(np.uint8)
  result = Image.fromarray(arr).resize((width, height), Image.BILINEAR)
  return np.asarray(result)


def process_frame(frame, crop=None, size=(64, 64)):
  obs = np.asarray(frame)
  if crop is not None:
    obs = obs[crop]
  obs = obs.astype(float) / 255.0
  obs = resize_image(obs, size).astype(float) / 255.0
  return ((1.0 - obs) * 255).round().astype(np.uint8)


def reset_env(env, seed=None):
  if seed is not None:
    result = env.reset(seed=seed)
  else:
    result = env.reset()
  if isinstance(result, tuple) and len(result) == 2:
    return result[0]
  return result


def step_env(env, action):
  result = env.step(action)
  if len(result) == 5:
    obs, reward, terminated, truncated, info = result
    return obs, reward, bool(terminated or truncated), info
  return result


def seed_env(env, seed=None):
  if hasattr(env, "reset"):
    try:
      env.reset(seed=seed)
      return [seed]
    except TypeError:
      pass
  if hasattr(env, "seed"):
    return env.seed(seed)
  return [seed]


def softmax_rows(x):
  x = np.asarray(x, dtype=np.float32)
  x = x - np.max(x, axis=1, keepdims=True)
  e = np.exp(x)
  return e / np.sum(e, axis=1, keepdims=True)


def legacy_lstm_step(x, state, kernel, bias, forget_bias=1.0):
  x = np.asarray(x, dtype=np.float32)
  if x.ndim == 1:
    x = x.reshape(1, -1)
  c = np.asarray(state.c, dtype=np.float32)
  h = np.asarray(state.h, dtype=np.float32)
  concat = np.concatenate([x, h], axis=1)
  gates = np.matmul(concat, kernel) + bias
  i, j, f, o = np.split(gates, 4, axis=1)
  new_c = c * sigmoid(f + forget_bias) + sigmoid(i) * np.tanh(j)
  new_h = np.tanh(new_c) * sigmoid(o)
  return new_h, LegacyLSTMState(new_c, new_h)


def sigmoid(x):
  return 1.0 / (1.0 + np.exp(-x))


class SimpleImageViewer:
  def __init__(self):
    self.root = None
    self.label = None
    self.photo = None
    self.window = None

  def imshow(self, arr):
    try:
      import tkinter as tk
      from PIL import ImageTk
      img = Image.fromarray(np.asarray(arr, dtype=np.uint8))
      if self.root is None:
        self.root = tk.Tk()
        self.root.title("World Models")
        self.window = self.root
        self.label = tk.Label(self.root)
        self.label.pack()
      self.photo = ImageTk.PhotoImage(img)
      self.label.configure(image=self.photo)
      self.root.update_idletasks()
      self.root.update()
    except Exception:
      return

  def close(self):
    try:
      if self.root is not None:
        self.root.destroy()
    except Exception:
      pass
    self.root = None
    self.label = None
    self.photo = None
    self.window = None

  def wait(self, seconds):
    if seconds <= 0 or self.root is None:
      return
    deadline = time.time() + seconds
    while time.time() < deadline:
      try:
        self.root.update_idletasks()
        self.root.update()
      except Exception:
        break
      time.sleep(0.05)


def project_root_for(file_path):
  return os.path.dirname(os.path.dirname(os.path.abspath(file_path)))
