import argparse
import json
import os

import numpy as np

from env import ACTION_SIZE, make_env
from experiment_logging import add_wandb_args, logger_from_args, should_log_wandb_videos
from experiment_metrics import summarize_compounding_error
from model import predict_rnn_next_z
from ns_symbolic import (
  NS_VARIANTS,
  apply_symbolic_projection,
  compare_with_symbolic,
  symbolic_transition,
  tabular_mask_to_vector_mask,
  tabular_to_vector,
)
from rnn.rnn import GridcraftRNN, rnn_init_state
from vae.vae import GridcraftVAE
from video_logging import record_world_model_comparison_video
from wandb_schema import GENERAL, WORLD_MODEL_EVALUATION
from progress_logging import append_progress


def compare_tabular(real_obs, imagined_obs, current_obs=None, action=None, symbolic_coverage=1.0):
  real_grid = np.asarray(real_obs["grid"], dtype=np.int16)
  imagined_grid = np.asarray(imagined_obs["grid"], dtype=np.int16)
  real_self = np.asarray(real_obs["self"], dtype=np.float32)
  imagined_self = np.asarray(imagined_obs["self"], dtype=np.float32)
  metrics = {
    "grid_mismatch": float(np.mean(real_grid != imagined_grid)),
    "terrain_mismatch": float(np.mean(real_grid[0] != imagined_grid[0])),
    "block_mismatch": float(np.mean(real_grid[1] != imagined_grid[1])),
    "entity_mismatch": float(np.mean(real_grid[2] != imagined_grid[2])),
    "self_mse": float(np.mean((real_self - imagined_self) ** 2)),
  }
  if current_obs is not None and action is not None:
    metrics.update(compare_with_symbolic(imagined_obs, current_obs, action, coverage=symbolic_coverage))
    _, mask = symbolic_transition(current_obs, action, coverage=symbolic_coverage)
    mask_vec = tabular_mask_to_vector_mask(mask)
    real_vec = tabular_to_vector(real_obs)
    imagined_vec = tabular_to_vector(imagined_obs)
    metrics["determinable_mismatch"] = float(np.mean(np.abs(real_vec[mask_vec] - imagined_vec[mask_vec]))) if np.any(mask_vec) else 0.0
    metrics["undeterminable_mismatch"] = float(np.mean(np.abs(real_vec[~mask_vec] - imagined_vec[~mask_vec]))) if np.any(~mask_vec) else 0.0
  else:
    metrics.update({
      "rvr": 0.0,
      "determinable_mismatch": 0.0,
      "undeterminable_mismatch": 0.0,
      "determinable_count": 0.0,
    })
  return metrics


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
  rnn.load_json(resolve_rnn_json(args))
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
      current_obs = env.last_obs
      proxy = type("Proxy", (), {"rnn": rnn, "rnn_state": state})()
      imagined_z = predict_rnn_next_z(proxy, mu[0], action, rng=rng, mode=args.imagination_mode)
      state = proxy.rnn_state
      next_obs, reward, done, info = env.step(action)
      imagined_obs = vae.decode_tabular(imagined_z)
      imagined_obs, _ = apply_symbolic_projection(imagined_obs, current_obs, action, args.ns_variant, coverage=args.symbolic_coverage)
      rows.append(compare_tabular(env.last_obs, imagined_obs, current_obs=current_obs, action=action, symbolic_coverage=args.symbolic_coverage))
      obs = next_obs
      if done:
        break
    env.close()
  return average_metrics(rows)


def evaluate_rnn_horizon(args):
  rng = np.random.default_rng(args.seed + 100000)
  vae = GridcraftVAE()
  vae.load_json(args.vae_json)
  rnn = GridcraftRNN()
  rnn.load_json(resolve_rnn_json(args))
  rows = []
  horizon = min(args.horizon_steps, args.max_steps)
  for episode in range(args.episodes):
    env = make_env(seed=args.seed + episode, render_mode=False, max_steps=args.max_steps)
    obs = env.reset(seed=args.seed + episode)
    state = rnn_init_state(rnn)
    imagined_current = env.last_obs
    z, _ = vae.encode_mu_logvar(obs.reshape(1, -1))
    z = z[0]
    for _ in range(horizon):
      action = int(rng.integers(0, ACTION_SIZE))
      proxy = type("Proxy", (), {"rnn": rnn, "rnn_state": state})()
      imagined_z = predict_rnn_next_z(proxy, z, action, rng=rng, mode=args.imagination_mode)
      state = proxy.rnn_state
      next_obs, reward, done, info = env.step(action)
      imagined_obs = vae.decode_tabular(imagined_z)
      imagined_obs, _ = apply_symbolic_projection(imagined_obs, imagined_current, action, args.ns_variant, coverage=args.symbolic_coverage)
      rows.append(compare_tabular(env.last_obs, imagined_obs, current_obs=imagined_current, action=action, symbolic_coverage=args.symbolic_coverage))
      imagined_current = imagined_obs
      z = imagined_z
      if done:
        break
    env.close()
  metrics = average_metrics(rows)
  return {f"horizon_{key}@{horizon}": value for key, value in metrics.items()}


def resolve_rnn_json(args):
  if args.rnn_json:
    return args.rnn_json
  candidates = {
    "neural": "rnn/rnn.neural.json",
    "regularization": "rnn/rnn.regularization.json",
    "projection": "rnn/rnn.neural.json",
    "residual": "rnn/rnn.residual.json",
  }
  path = candidates.get(args.ns_variant)
  return path if path and os.path.exists(path) else "rnn/rnn.json"


def main():
  parser = argparse.ArgumentParser()
  group = parser.add_mutually_exclusive_group(required=True)
  group.add_argument("--vae-only", action="store_true")
  group.add_argument("--rnn-one-step", action="store_true")
  parser.add_argument("--vae-json", default="vae/vae.json")
  parser.add_argument("--rnn-json", default=None)
  parser.add_argument("--ns-variant", choices=NS_VARIANTS, default="neural")
  parser.add_argument("--symbolic-coverage", type=float, default=1.0)
  parser.add_argument("--episodes", type=int, default=10)
  parser.add_argument("--max-steps", type=int, default=100)
  parser.add_argument("--seed", type=int, default=1)
  parser.add_argument("--imagination-mode", choices=["mean", "mode", "sample"], default="mean")
  parser.add_argument("--progress-every", type=int, default=10)
  parser.add_argument("--horizon-steps", type=int, default=0)
  parser.add_argument("--horizons", nargs="+", type=int, default=None)
  parser.add_argument("--out", default=None)
  parser.add_argument("--progress-log", default=None)
  add_wandb_args(parser)
  args = parser.parse_args()

  logger = logger_from_args(
    args,
    config=vars(args),
    default_group=f"eval_{args.ns_variant}",
    default_name=f"gridcraft-eval-{args.ns_variant}-seed{args.seed}",
    tags=["gridcraft", "evaluation", args.ns_variant],
    info_sections=[GENERAL, WORLD_MODEL_EVALUATION],
    out_dir=(os.path.dirname(args.out) or "trainlog") if args.out else "trainlog",
  )

  if args.vae_only:
    metrics = evaluate_vae(args)
    default_out = "trainlog/vae_eval.json"
  else:
    metrics = evaluate_rnn_one_step(args)
    horizons = args.horizons if args.horizons is not None else ([args.horizon_steps] if args.horizon_steps > 0 else [])
    for horizon in horizons:
      args.horizon_steps = int(horizon)
      metrics.update(evaluate_rnn_horizon(args))
    default_out = "trainlog/rnn_eval.json"

  out_path = args.out or default_out
  out_dir = os.path.dirname(out_path)
  if out_dir:
    os.makedirs(out_dir, exist_ok=True)
  with open(out_path, "w") as f:
    json.dump(metrics, f, indent=2)
  logger.log(metrics, namespace="wm_evaluation")
  compounding_metrics = summarize_compounding_error(metrics)
  logger.log(compounding_metrics, namespace="wm_evaluation")
  append_progress(args.progress_log, metrics, namespace="wm_evaluation")
  append_progress(args.progress_log, compounding_metrics, namespace="wm_evaluation")
  if args.rnn_one_step and should_log_wandb_videos(args):
    frames = record_world_model_comparison_video(
      vae_json=args.vae_json,
      rnn_json=resolve_rnn_json(args),
      ns_variant=args.ns_variant,
      symbolic_coverage=args.symbolic_coverage,
      seed=args.seed,
      episodes=args.video_episodes,
      max_steps=args.video_max_steps,
      imagination_mode=args.imagination_mode,
    )
    logger.log_video(
      "video_real_vs_imagined",
      frames,
      fps=args.video_fps,
      namespace="wm_evaluation",
    )
  logger.save_json(out_path, metrics)
  logger.finish()
  print(json.dumps(metrics, indent=2))
  print("saved", out_path)


if __name__ == "__main__":
  main()
