import argparse
import json
import os

import numpy as np

from env import ACTION_SIZE, make_env
from model import predict_rnn_next_z
from rnn.rnn import GridcraftRNN, rnn_init_state
from vae.vae import GridcraftVAE


def compare_tabular(real_obs, imagined_obs):
  real_grid = np.asarray(real_obs["grid"], dtype=np.int16)
  imagined_grid = np.asarray(imagined_obs["grid"], dtype=np.int16)
  real_self = np.asarray(real_obs["self"], dtype=np.float32)
  imagined_self = np.asarray(imagined_obs["self"], dtype=np.float32)
  return {
    "grid_mismatch": float(np.mean(real_grid != imagined_grid)),
    "terrain_mismatch": float(np.mean(real_grid[0] != imagined_grid[0])),
    "block_mismatch": float(np.mean(real_grid[1] != imagined_grid[1])),
    "entity_mismatch": float(np.mean(real_grid[2] != imagined_grid[2])),
    "self_mse": float(np.mean((real_self - imagined_self) ** 2)),
  }


def average_metrics(rows):
  if not rows:
    return {}
  keys = rows[0].keys()
  return {key: float(np.mean([row[key] for row in rows])) for key in keys}


def evaluate_vae(args):
  vae = GridcraftVAE()
  vae.load_json(args.vae_json)
  rows = []
  for episode in range(args.episodes):
    if args.progress_every and (episode == 0 or (episode + 1) % args.progress_every == 0):
      print(f"vae_eval episode {episode + 1}/{args.episodes}", flush=True)
    rng = np.random.default_rng(args.seed + episode)
    env = make_env(seed=args.seed + episode, render_mode=False, max_steps=args.max_steps)
    obs = env.reset(seed=args.seed + episode)
    for _ in range(args.max_steps):
      mu, _ = vae.encode_mu_logvar(obs.reshape(1, -1))
      imagined_obs = vae.decode_tabular(mu[0])
      rows.append(compare_tabular(env.last_obs, imagined_obs))
      action = int(rng.integers(0, ACTION_SIZE))
      obs, reward, done, info = env.step(action)
      if done:
        break
    env.close()
  return average_metrics(rows)


def evaluate_rnn_one_step(args):
  rng = np.random.default_rng(args.seed)
  vae = GridcraftVAE()
  vae.load_json(args.vae_json)
  rnn = GridcraftRNN()
  rnn.load_json(args.rnn_json)
  rows = []
  for episode in range(args.episodes):
    if args.progress_every and (episode == 0 or (episode + 1) % args.progress_every == 0):
      print(f"rnn_eval episode {episode + 1}/{args.episodes}", flush=True)
    env = make_env(seed=args.seed + episode, render_mode=False, max_steps=args.max_steps)
    obs = env.reset(seed=args.seed + episode)
    state = rnn_init_state(rnn)
    for _ in range(args.max_steps):
      mu, _ = vae.encode_mu_logvar(obs.reshape(1, -1))
      action = int(rng.integers(0, ACTION_SIZE))
      proxy = type("Proxy", (), {"rnn": rnn, "rnn_state": state})()
      imagined_z = predict_rnn_next_z(proxy, mu[0], action, rng=rng, mode=args.imagination_mode)
      state = proxy.rnn_state
      next_obs, reward, done, info = env.step(action)
      imagined_obs = vae.decode_tabular(imagined_z)
      rows.append(compare_tabular(env.last_obs, imagined_obs))
      obs = next_obs
      if done:
        break
    env.close()
  return average_metrics(rows)


def main():
  parser = argparse.ArgumentParser()
  group = parser.add_mutually_exclusive_group(required=True)
  group.add_argument("--vae-only", action="store_true")
  group.add_argument("--rnn-one-step", action="store_true")
  parser.add_argument("--vae-json", default="vae/vae.json")
  parser.add_argument("--rnn-json", default="rnn/rnn.json")
  parser.add_argument("--episodes", type=int, default=10)
  parser.add_argument("--max-steps", type=int, default=100)
  parser.add_argument("--seed", type=int, default=1)
  parser.add_argument("--imagination-mode", choices=["mean", "mode", "sample"], default="mean")
  parser.add_argument("--progress-every", type=int, default=10)
  parser.add_argument("--out", default=None)
  args = parser.parse_args()

  if args.vae_only:
    metrics = evaluate_vae(args)
    default_out = "trainlog/vae_eval.json"
  else:
    metrics = evaluate_rnn_one_step(args)
    default_out = "trainlog/rnn_eval.json"

  out_path = args.out or default_out
  out_dir = os.path.dirname(out_path)
  if out_dir:
    os.makedirs(out_dir, exist_ok=True)
  with open(out_path, "w") as f:
    json.dump(metrics, f, indent=2)
  print(json.dumps(metrics, indent=2))
  print("saved", out_path)


if __name__ == "__main__":
  main()
