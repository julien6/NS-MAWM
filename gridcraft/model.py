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
from exp_config import ACTION_SIZE, CONTROLLER_INPUT_SIZE
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

  def render(self, observation, world=None):
    self._write({"cmd": "render_human" if self.display else "render", "observation": observation, "world": world})
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


def decode_tabular_observation(controller, z):
  return {"agent_0": controller.vae.decode_tabular(z)}


def world_snapshot(env):
  world = env.env.world
  agents = {}
  observations = {}
  for agent_id, agent in world.agents.items():
    agents[agent_id] = {
      "x": int(agent.x),
      "y": int(agent.y),
      "hp": int(agent.hp),
      "hunger": int(agent.hunger),
      "inventory": {str(int(item)): int(count) for item, count in agent.inventory.items()},
      "inventory_order": [int(item) for item in agent.inventory_order],
      "equipped": int(agent.equipped) if agent.equipped is not None else None,
      "alive": bool(agent.alive),
    }
  for agent_id, observation in world.observations().items():
    observations[agent_id] = {
      "grid": np.asarray(observation["grid"], dtype=np.int8),
      "self": np.asarray(observation["self"], dtype=np.int16),
    }
  return {
    "terrain": np.asarray(world.terrain, dtype=np.int8),
    "blocks": np.asarray(world.blocks, dtype=np.int8),
    "items": [
      {
        "item": int(item.item),
        "count": int(item.count),
        "x": int(item.x),
        "y": int(item.y),
      }
      for item in world.items
    ],
    "mobs": [
      {
        "mob_id": int(mob.mob_id),
        "x": int(mob.x),
        "y": int(mob.y),
        "hp": int(mob.hp),
        "alive": bool(mob.alive),
      }
      for mob in world.mobs
    ],
    "agents": agents,
    "observations": observations,
  }


def predict_rnn_next_z(controller, z, action, rng=None, mode="mean"):
  logmix, mean, logstd, reward, done_logit, next_state = controller.rnn.step(z, action, controller.rnn_state)
  mix = np.exp(logmix - np.max(logmix, axis=1, keepdims=True))
  mix = mix / np.sum(mix, axis=1, keepdims=True)
  if mode == "mean":
    next_z = np.sum(mix * mean, axis=1).astype(np.float32)
  elif mode == "mode":
    component = np.argmax(mix, axis=1)
    next_z = mean[np.arange(controller.rnn.z_size), component].astype(np.float32)
  elif mode == "sample":
    if rng is None:
      rng = np.random.default_rng()
    next_z = np.zeros((controller.rnn.z_size,), dtype=np.float32)
    for j in range(controller.rnn.z_size):
      component = int(rng.choice(np.arange(controller.rnn.num_mixture), p=mix[j]))
      next_z[j] = mean[j, component] + np.exp(logstd[j, component]) * rng.standard_normal()
  else:
    raise ValueError(f"unknown imagination mode: {mode}")
  controller.rnn_state = next_state
  return next_z


def compare_world_model_random(controller, render_mode=False, episodes=1, seed=1, max_steps=500, render_delay=0.1, render_hold=0.0, imagination_mode="mean"):
  rng = np.random.default_rng(seed)
  rewards = []
  lengths = []
  renderer = TabularRenderWorker(display=render_mode) if render_mode else None
  mismatches = []
  for episode in range(episodes):
    env = make_env(seed=seed + episode, render_mode=False, max_steps=max_steps)
    obs = env.reset(seed=seed + episode)
    controller.reset()
    total_reward = 0.0
    episode_mismatches = []
    for t in range(max_steps):
      z = controller.encode_obs(obs)
      action = int(rng.integers(0, ACTION_SIZE))
      imagined_z = predict_rnn_next_z(controller, z, action, rng=rng, mode=imagination_mode)
      next_obs, reward, done, info = env.step(action)
      imagined_obs = decode_tabular_observation(controller, imagined_z)["agent_0"]
      real_obs = env.last_obs
      grid_mismatch = float(np.mean(real_obs["grid"] != imagined_obs["grid"]))
      self_mse = float(np.mean((np.asarray(real_obs["self"], dtype=np.float32) - np.asarray(imagined_obs["self"], dtype=np.float32)) ** 2))
      episode_mismatches.append((grid_mismatch, self_mse))
      total_reward += reward
      obs = next_obs
      if render_mode:
        renderer.render({"agent_0": imagined_obs}, world=world_snapshot(env))
        if render_delay > 0:
          time.sleep(render_delay)
      if done:
        break
    env.close()
    rewards.append(total_reward)
    lengths.append(t + 1)
    if episode_mismatches:
      mismatches.append(np.mean(np.asarray(episode_mismatches), axis=0))
  if renderer is not None:
    renderer.wait(render_hold)
    renderer.close()
  if mismatches:
    mismatch_arr = np.asarray(mismatches)
    print("mean_grid_mismatch", float(np.mean(mismatch_arr[:, 0])), "mean_self_mse", float(np.mean(mismatch_arr[:, 1])))
  return rewards, lengths


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


def simulate(controller, env_name="gridcraftreal", train_mode=False, render_mode=False, episodes=1, seed=1, max_steps=500, num_episode=None, max_len=-1, render_delay=0.1, render_hold=0.0, imagination_mode="mean"):
  if env_name == "gridcraftcompare":
    return compare_world_model_random(controller, render_mode=render_mode, episodes=episodes, seed=seed, max_steps=max_steps, render_delay=render_delay, render_hold=render_hold, imagination_mode=imagination_mode)
  if env_name == "gridcraftrnn":
    return simulate_dream(controller, render_mode=render_mode, episodes=episodes, seed=seed, max_steps=max_steps, num_episode=num_episode, max_len=max_len, render_delay=render_delay, render_hold=render_hold)
  return simulate_real(controller, train_mode=train_mode, render_mode=render_mode, episodes=episodes, seed=seed, max_steps=max_steps, num_episode=num_episode, max_len=max_len, render_delay=render_delay, render_hold=render_hold)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("env_name", choices=["gridcraftreal", "gridcraftrnn", "gridcraftcompare"])
  parser.add_argument("mode", choices=["render", "norender"])
  parser.add_argument("model_json", nargs="?", default=None)
  parser.add_argument("--episodes", type=int, default=None)
  parser.add_argument("--max-steps", type=int, default=500)
  parser.add_argument("--seed", type=int, default=1)
  parser.add_argument("--render-delay", type=float, default=0.1)
  parser.add_argument("--render-hold", type=float, default=None)
  parser.add_argument("--imagination-mode", choices=["mean", "mode", "sample"], default="mean")
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
    imagination_mode=args.imagination_mode,
  )
  print("rewards", rewards)
  print("mean_reward", float(np.mean(rewards)), "std", float(np.std(rewards)), "mean_length", float(np.mean(lengths)))


if __name__ == "__main__":
  main()
