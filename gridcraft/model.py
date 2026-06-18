import argparse
import json
import os
import pickle
import struct
import subprocess
import sys
import time

import numpy as np

from dream_env import make_env as make_dream_env
from env import make_env
from exp_config import ACTION_SIZE, BLOCK_CLASSES, CONTROLLER_INPUT_SIZE, ENTITY_CLASSES, GRID_CELLS, GRID_FEATURES, TERRAIN_CLASSES
from rnn.rnn import GridcraftRNN, rnn_init_state, rnn_next_state, rnn_output
from vae.vae import GridcraftVAE


class TabularRenderWorker:
  def __init__(self, display=False):
    self.display = display
    worker_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tabular_render_worker.py")
    self.process = subprocess.Popen(
      [sys.executable, worker_path],
      stdin=subprocess.PIPE,
      stdout=subprocess.PIPE,
      stderr=subprocess.DEVNULL,
    )

  def render(self, observation):
    self._write({"cmd": "render_human" if self.display else "render", "observation": observation})
    return self._read()

  def wait(self, seconds):
    if seconds <= 0:
      return
    self._write({"cmd": "wait", "seconds": seconds})
    self._read()

  def close(self):
    try:
      self._write({"cmd": "close"})
    except Exception:
      pass
    try:
      self.process.wait(timeout=2)
    except subprocess.TimeoutExpired:
      self.process.terminate()

  def _write(self, message):
    payload = pickle.dumps(message, protocol=pickle.HIGHEST_PROTOCOL)
    self.process.stdin.write(struct.pack("!I", len(payload)))
    self.process.stdin.write(payload)
    self.process.stdin.flush()

  def _read(self):
    header = self.process.stdout.read(4)
    if not header:
      raise RuntimeError("tabular render worker stopped before returning a frame")
    size = struct.unpack("!I", header)[0]
    payload = self.process.stdout.read(size)
    if len(payload) != size:
      raise RuntimeError("tabular render worker returned a truncated frame")
    return pickle.loads(payload)


def make_model(load_model=True):
  return GridcraftController(load_world_model=load_model)


class GridcraftController:
  def __init__(self, vae_path="vae/vae.json", rnn_path="rnn/rnn.json", load_world_model=True):
    self.vae = GridcraftVAE()
    self.rnn = GridcraftRNN()
    if load_world_model:
      self.vae.load_json(vae_path)
      self.rnn.load_json(rnn_path)
    self.input_size = CONTROLLER_INPUT_SIZE
    self.output_size = ACTION_SIZE
    self.param_count = (self.input_size + 1) * self.output_size
    self.weight = np.zeros((self.input_size, self.output_size), dtype=np.float32)
    self.bias = np.zeros((self.output_size,), dtype=np.float32)
    self.rnn_state = rnn_init_state(self.rnn)

  def reset(self):
    self.rnn_state = rnn_init_state(self.rnn)

  def encode_obs(self, obs):
    mu, _ = self.vae.encode_mu_logvar(obs.reshape(1, -1))
    return mu[0]

  def get_action(self, z):
    h = rnn_output(self.rnn_state, z)
    logits = np.matmul(h, self.weight) + self.bias
    return int(np.argmax(logits))

  def observe_transition(self, z, action):
    self.rnn_state = rnn_next_state(self.rnn, z, action, self.rnn_state)

  def set_model_params(self, params):
    params = np.asarray(params, dtype=np.float32)
    self.bias = params[:self.output_size]
    self.weight = params[self.output_size:].reshape(self.input_size, self.output_size)

  def get_model_params(self):
    return np.concatenate([self.bias, self.weight.reshape(-1)]).astype(float)

  def load_model(self, filename):
    with open(filename) as f:
      data = json.load(f)
    self.set_model_params(data[0])

  def save_model(self, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w") as f:
      json.dump([self.get_model_params().round(6).tolist()], f, indent=2)


def _decode_grid_text(controller, z):
  decoded = controller.vae.decode(z.reshape(1, -1))[0]
  cursor = 0
  terrain = decoded[cursor:cursor + GRID_CELLS * TERRAIN_CLASSES].reshape(GRID_CELLS, TERRAIN_CLASSES).argmax(axis=1)
  cursor += GRID_CELLS * TERRAIN_CLASSES
  blocks = decoded[cursor:cursor + GRID_CELLS * BLOCK_CLASSES].reshape(GRID_CELLS, BLOCK_CLASSES).argmax(axis=1)
  cursor += GRID_CELLS * BLOCK_CLASSES
  entities = decoded[cursor:cursor + GRID_CELLS * ENTITY_CLASSES].reshape(GRID_CELLS, ENTITY_CLASSES).argmax(axis=1)
  chars = []
  for t, b, e in zip(terrain, blocks, entities):
    if e:
      chars.append("A" if e == 1 else "E")
    elif b:
      chars.append("#")
    elif t == 1:
      chars.append(".")
    elif t == 2:
      chars.append("~")
    else:
      chars.append(" ")
  return "\n".join("".join(chars[i:i + 7]) for i in range(0, GRID_CELLS, 7))


def decode_tabular_observation(controller, z):
  decoded = controller.vae.decode(z.reshape(1, -1))[0]
  cursor = 0
  terrain = decoded[cursor:cursor + GRID_CELLS * TERRAIN_CLASSES].reshape(GRID_CELLS, TERRAIN_CLASSES).argmax(axis=1)
  cursor += GRID_CELLS * TERRAIN_CLASSES
  blocks = decoded[cursor:cursor + GRID_CELLS * BLOCK_CLASSES].reshape(GRID_CELLS, BLOCK_CLASSES).argmax(axis=1)
  cursor += GRID_CELLS * BLOCK_CLASSES
  entities = decoded[cursor:cursor + GRID_CELLS * ENTITY_CLASSES].reshape(GRID_CELLS, ENTITY_CLASSES).argmax(axis=1)
  self_vec = decoded[GRID_FEATURES:]
  hp_hunger = np.clip(np.rint(self_vec[:2] * 20.0), 0, 20)
  inventory = np.clip(np.rint(self_vec[2:] * 10.0), 0, 99)
  return {
    "agent_0": {
      "grid": np.stack([
        terrain.reshape(7, 7),
        blocks.reshape(7, 7),
        entities.reshape(7, 7),
      ]).astype(np.int8),
      "self": np.concatenate([hp_hunger, inventory]).astype(np.int16),
    }
  }


def _wait_real_viewer(env, seconds):
  viewer = getattr(env, "viewer", None)
  if viewer is not None:
    viewer.wait(seconds)


def simulate_real(controller, train_mode=False, render_mode=False, episodes=1, seed=1, max_steps=500, num_episode=None, max_len=-1, render_delay=0.1, render_hold=0.0):
  if num_episode is not None:
    episodes = num_episode
  if max_len > 0:
    max_steps = max_len
  rewards = []
  lengths = []
  renderer = TabularRenderWorker(display=True) if render_mode else None
  for episode in range(episodes):
    env = make_env(seed=seed + episode, render_mode=False, max_steps=max_steps)
    obs = env.reset(seed=seed + episode)
    controller.reset()
    total_reward = 0.0
    for t in range(max_steps):
      z = controller.encode_obs(obs)
      action = controller.get_action(z)
      next_obs, reward, done, info = env.step(action)
      controller.observe_transition(z, action)
      total_reward += reward
      obs = next_obs
      if render_mode:
        renderer.render({env.agent_id: env.last_obs})
        if render_delay > 0:
          time.sleep(render_delay)
      if done:
        break
    env.close()
    rewards.append(total_reward)
    lengths.append(t + 1)
  if renderer is not None:
    renderer.wait(render_hold)
    renderer.close()
  return rewards, lengths


def simulate_dream(controller, render_mode=False, episodes=1, seed=1, max_steps=500, num_episode=None, max_len=-1, render_delay=0.1, render_hold=0.0):
  if num_episode is not None:
    episodes = num_episode
  if max_len > 0:
    max_steps = max_len
  rewards = []
  lengths = []
  renderer = TabularRenderWorker(display=True) if render_mode else None
  for episode in range(episodes):
    env = make_dream_env(seed=seed + episode, max_steps=max_steps)
    z = env.reset(seed=seed + episode)
    controller.reset()
    total_reward = 0.0
    for t in range(max_steps):
      controller.rnn_state = env.rnn_state
      action = controller.get_action(z)
      z, reward, done, info = env.step(action)
      controller.rnn_state = env.rnn_state
      total_reward += reward
      if render_mode:
        renderer.render(decode_tabular_observation(controller, z))
        if render_delay > 0:
          time.sleep(render_delay)
      if done:
        break
    env.close()
    rewards.append(total_reward)
    lengths.append(t + 1)
  if renderer is not None:
    renderer.wait(render_hold)
    renderer.close()
  return rewards, lengths


def simulate(controller, env_name="gridcraftreal", train_mode=False, render_mode=False, episodes=1, seed=1, max_steps=500, num_episode=None, max_len=-1, render_delay=0.1, render_hold=0.0):
  if env_name == "gridcraftrnn":
    return simulate_dream(controller, render_mode=render_mode, episodes=episodes, seed=seed, max_steps=max_steps, num_episode=num_episode, max_len=max_len, render_delay=render_delay, render_hold=render_hold)
  return simulate_real(controller, train_mode=train_mode, render_mode=render_mode, episodes=episodes, seed=seed, max_steps=max_steps, num_episode=num_episode, max_len=max_len, render_delay=render_delay, render_hold=render_hold)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("env_name", choices=["gridcraftreal", "gridcraftrnn"])
  parser.add_argument("mode", choices=["render", "norender"])
  parser.add_argument("model_json", nargs="?", default=None)
  parser.add_argument("--episodes", type=int, default=None)
  parser.add_argument("--max-steps", type=int, default=500)
  parser.add_argument("--seed", type=int, default=1)
  parser.add_argument("--render-delay", type=float, default=0.1)
  parser.add_argument("--render-hold", type=float, default=None)
  args = parser.parse_args()

  controller = make_model(load_model=True)
  if args.model_json:
    controller.load_model(args.model_json)
  else:
    rng = np.random.default_rng(args.seed)
    controller.set_model_params(rng.standard_normal(controller.param_count) * 0.01)

  render_mode = args.mode == "render"
  episodes = args.episodes if args.episodes is not None else (1 if render_mode else 100)
  render_hold = args.render_hold if args.render_hold is not None else (10.0 if render_mode else 0.0)
  rewards, lengths = simulate(
    controller,
    env_name=args.env_name,
    render_mode=render_mode,
    episodes=episodes,
    seed=args.seed,
    max_steps=args.max_steps,
    render_delay=args.render_delay,
    render_hold=render_hold,
  )
  print("rewards", rewards)
  print("mean_reward", float(np.mean(rewards)), "std", float(np.std(rewards)), "mean_length", float(np.mean(lengths)))


if __name__ == "__main__":
  main()
