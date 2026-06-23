import argparse
import json
import os
import subprocess
import sys

import numpy as np

from experiment_logging import add_wandb_args, logger_from_args
from ns_symbolic import symbolic_batch_targets
from progress_logging import append_progress
from rnn.rnn import GridcraftRNN
from vae.vae import GridcraftVAE
from wandb_schema import GENERAL, WORLD_MODEL_EVALUATION, WORLD_MODEL_TRAINING


def load_series(path):
  data = np.load(path)
  z_all = data["z"].astype(np.float32)
  action_all = data["action"].astype(np.int64)
  reward_all = data["reward"].astype(np.float32)
  done_all = data["done"].astype(np.bool_)
  obs_all = data["obs"].astype(np.float32) if "obs" in data.files else None
  episodes = []
  lengths = data["length"] if "length" in data.files else np.full((len(z_all),), z_all.shape[1])
  for i, length in enumerate(lengths):
    n = int(length)
    episodes.append((z_all[i, :n],
                     obs_all[i, :n] if obs_all is not None else None,
                     action_all[i, :n],
                     reward_all[i, :n],
                     done_all[i, :n]))
  return episodes, obs_all is not None


def sample_batch(episodes, batch_size, seq_len, rng):
  candidates = [ep for ep in episodes if len(ep[0]) > seq_len]
  if not candidates:
    raise RuntimeError("no episodes long enough for requested sequence length")
  z_batch = []
  obs_batch = []
  action_batch = []
  reward_batch = []
  done_batch = []
  for _ in range(batch_size):
    z, obs, action, reward, done = candidates[int(rng.integers(0, len(candidates)))]
    start = int(rng.integers(0, len(z) - seq_len))
    end = start + seq_len + 1
    z_batch.append(z[start:end])
    if obs is not None:
      obs_batch.append(obs[start:end])
    action_batch.append(action[start:end])
    reward_batch.append(np.pad(reward[start:start + seq_len], (0, 1)))
    done_batch.append(np.pad(done[start:start + seq_len], (0, 1)))
  return (np.asarray(z_batch, dtype=np.float32),
          np.asarray(obs_batch, dtype=np.float32) if obs_batch else None,
          np.asarray(action_batch, dtype=np.int64),
          np.asarray(reward_batch, dtype=np.float32),
          np.asarray(done_batch, dtype=np.bool_))


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--series", default="series/series.npz")
  parser.add_argument("--initial-z", default="series/initial_z.npz")
  parser.add_argument("--out-dir", default="rnn")
  parser.add_argument("--initial-z-out", default="initial_z/initial_z.json")
  parser.add_argument("--steps", type=int, default=10000)
  parser.add_argument("--batch-size", type=int, default=64)
  parser.add_argument("--seq-len", type=int, default=32)
  parser.add_argument("--seed", type=int, default=1)
  parser.add_argument("--ns-variant", choices=("neural", "regularization", "residual"), default="neural")
  parser.add_argument("--lambda-sym", type=float, default=1.0)
  parser.add_argument("--symbolic-coverage", type=float, default=1.0)
  parser.add_argument("--vae-json", default="vae/vae.json")
  parser.add_argument("--eval-every", type=int, default=0)
  parser.add_argument("--eval-out-dir", default=None)
  parser.add_argument("--eval-episodes", type=int, default=10)
  parser.add_argument("--eval-max-steps", type=int, default=100)
  parser.add_argument("--eval-horizons", nargs="+", type=int, default=None)
  parser.add_argument("--python", default=sys.executable)
  parser.add_argument("--progress-log", default=None)
  add_wandb_args(parser)
  args = parser.parse_args()

  config = vars(args).copy()
  logger = logger_from_args(
    args,
    config=config,
    default_group=f"rnn_{args.ns_variant}",
    default_name=f"gridcraft-rnn-{args.ns_variant}-seed{args.seed}",
    tags=["gridcraft", "rnn", args.ns_variant],
    info_sections=[GENERAL, WORLD_MODEL_TRAINING, WORLD_MODEL_EVALUATION],
    out_dir=args.out_dir,
  )
  rng = np.random.default_rng(args.seed)
  episodes, has_obs = load_series(args.series)
  args.seq_len = resolve_seq_len(args.seq_len, episodes)
  if args.ns_variant != "neural" and not has_obs:
    raise RuntimeError("series file has no obs array; rerun series.py before NS-MAWM training")
  model = GridcraftRNN()
  vae = None
  if args.ns_variant != "neural":
    vae = GridcraftVAE()
    vae.load_json(args.vae_json)
  os.makedirs(args.out_dir, exist_ok=True)

  initial_z = np.load(args.initial_z)["z"].astype(np.float32)
  os.makedirs(os.path.dirname(args.initial_z_out), exist_ok=True)
  with open(args.initial_z_out, "w") as f:
    json.dump(initial_z.tolist(), f)

  for step in range(args.steps):
    z, obs, action, reward, done = sample_batch(episodes, args.batch_size, args.seq_len, rng)
    kwargs = {}
    if args.ns_variant == "regularization":
      symbolic_target, symbolic_mask = symbolic_batch_targets(obs[:, :-1, :], action[:, :-1], coverage=args.symbolic_coverage)
      kwargs.update(symbolic_target=symbolic_target, symbolic_mask=symbolic_mask, vae_decoder=vae.decoder, lambda_sym=args.lambda_sym)
    elif args.ns_variant == "residual":
      symbolic_target, symbolic_mask = symbolic_batch_targets(obs[:, :-1, :], action[:, :-1], coverage=args.symbolic_coverage)
      kwargs.update(target_obs=obs[:, 1:, :], target_obs_mask=1.0 - symbolic_mask, vae_decoder=vae.decoder, lambda_sym=args.lambda_sym)
    loss, z_loss, mean_loss, symbolic_loss, residual_loss, reward_loss, done_loss = model.train_batch(z, action, reward, done, **kwargs)
    if step == 0 or (step + 1) % 100 == 0:
      print("step", step + 1, "loss", loss, "z", z_loss, "mean_mse", mean_loss, "symbolic", symbolic_loss, "residual", residual_loss, "reward", reward_loss, "done", done_loss)
      metrics = {
        "training_wm_total_loss": loss,
        "training_obs_loss": z_loss,
        "training_mean_mse": mean_loss,
        "training_symbolic_loss": symbolic_loss,
        "training_residual_loss": residual_loss,
        "training_reward_loss": reward_loss,
        "training_done_loss": done_loss,
      }
      logger.log(metrics, step=step + 1, namespace="wm_training")
      append_progress(args.progress_log, metrics, step=step + 1, namespace="wm_training")
    if args.eval_every > 0 and (step + 1) % args.eval_every == 0:
      eval_metrics = checkpoint_and_evaluate(model, args, step + 1)
      logger.log(eval_metrics, step=step + 1, namespace="wm_evaluation")
      append_progress(args.progress_log, eval_metrics, step=step + 1, namespace="wm_evaluation")

  out_name = "rnn.json" if args.ns_variant == "neural" else f"rnn.{args.ns_variant}.json"
  if args.ns_variant == "neural":
    model.save_json(os.path.join(args.out_dir, "rnn.neural.json"))
  model.save_json(os.path.join(args.out_dir, out_name))
  print("saved", os.path.join(args.out_dir, out_name))
  logger.finish()


def checkpoint_and_evaluate(model, args, step):
  eval_out_dir = args.eval_out_dir or os.path.join(args.out_dir, "eval")
  checkpoint_dir = os.path.join(args.out_dir, "checkpoints")
  os.makedirs(eval_out_dir, exist_ok=True)
  os.makedirs(checkpoint_dir, exist_ok=True)
  checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{step}.json")
  eval_path = os.path.join(eval_out_dir, f"world_model_step_{step}.json")
  model.save_json(checkpoint_path)
  eval_variant = "neural" if args.ns_variant == "neural" else args.ns_variant
  cmd = [
    args.python,
    "evaluate_world_model.py",
    "--rnn-one-step",
    "--vae-json", args.vae_json,
    "--rnn-json", checkpoint_path,
    "--ns-variant", eval_variant,
    "--symbolic-coverage", str(args.symbolic_coverage),
    "--episodes", str(args.eval_episodes),
    "--max-steps", str(args.eval_max_steps),
    "--out", eval_path,
    "--progress-every", "0",
  ]
  if args.eval_horizons:
    cmd.append("--horizons")
    cmd.extend(str(horizon) for horizon in args.eval_horizons)
  print(" ".join(cmd), flush=True)
  subprocess.run(cmd, check=True)
  with open(eval_path) as f:
    metrics = json.load(f)
  prefixed = {f"eval/{key}": value for key, value in metrics.items()}
  prefixed["eval/checkpoint_step"] = step
  return prefixed


def resolve_seq_len(seq_len, episodes):
  max_length = max((len(ep[0]) for ep in episodes), default=0)
  if max_length <= 1:
    raise RuntimeError("no episodes long enough for RNN training")
  if seq_len >= max_length:
    adjusted = max_length - 1
    print(f"adjusted seq_len from {seq_len} to {adjusted} for max episode length {max_length}", flush=True)
    return adjusted
  return seq_len


if __name__ == "__main__":
  main()
