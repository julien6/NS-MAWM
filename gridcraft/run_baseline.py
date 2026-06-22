import argparse
import json
import os
import subprocess
import sys

from experiment_config import get_baseline, list_baselines
from experiment_logging import add_wandb_args, logger_from_args


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--baseline-id", default=None)
  parser.add_argument("--list", action="store_true")
  parser.add_argument("--run-dir", default="runs")
  parser.add_argument("--python", default="../.venv/bin/python")
  parser.add_argument("--steps", type=int, default=10000)
  parser.add_argument("--batch-size", type=int, default=64)
  parser.add_argument("--seq-len", type=int, default=32)
  parser.add_argument("--lambda-sym", type=float, default=1.0)
  parser.add_argument("--episodes", type=int, default=100)
  parser.add_argument("--max-steps", type=int, default=500)
  parser.add_argument("--horizon-steps", type=int, default=50)
  parser.add_argument("--horizons", nargs="+", type=int, default=None)
  parser.add_argument("--seed", type=int, default=None)
  parser.add_argument("--dry-run", action="store_true")
  parser.add_argument("--skip-series", action="store_true")
  parser.add_argument("--skip-train", action="store_true")
  parser.add_argument("--series-limit", type=int, default=None)
  add_wandb_args(parser)
  args = parser.parse_args()

  if args.list:
    for baseline in list_baselines():
      print(json.dumps(baseline.to_dict(), sort_keys=True))
    return
  if not args.baseline_id:
    raise SystemExit("--baseline-id is required unless --list is used")

  baseline = get_baseline(args.baseline_id)
  seed = baseline.seed if args.seed is None else args.seed
  run_name = f"{baseline.baseline_id}_{baseline.ns_variant}_k{baseline.coverage}_seed{seed}"
  run_dir = os.path.join(args.run_dir, run_name)
  series_dir = os.path.join(run_dir, "series")
  rnn_dir = os.path.join(run_dir, "rnn")
  os.makedirs(run_dir, exist_ok=True)

  config = baseline.to_dict()
  config.update(vars(args))
  config["seed"] = seed
  config["run_dir"] = run_dir
  logger = logger_from_args(
    args,
    config=config,
    default_group=baseline.baseline_id,
    default_name=run_name,
    tags=["gridcraft", "baseline", baseline.baseline_id, baseline.ns_variant],
  )
  logger.save_json(os.path.join(run_dir, "baseline_config.json"), config)

  train_variant = "neural" if baseline.ns_variant == "projection" else baseline.ns_variant
  rnn_file = "rnn.neural.json" if train_variant == "neural" else f"rnn.{train_variant}.json"
  rnn_json = os.path.join(rnn_dir, rnn_file)

  commands = []
  if not args.skip_series:
    commands.append([
      args.python, "series.py",
      "--out-dir", series_dir,
    ] + (["--limit", str(args.series_limit)] if args.series_limit is not None else []))
  if not args.skip_train:
    cmd = [
      args.python, "rnn_train.py",
      "--series", os.path.join(series_dir, "series.npz"),
      "--initial-z", os.path.join(series_dir, "initial_z.npz"),
      "--out-dir", rnn_dir,
      "--initial-z-out", os.path.join(run_dir, "initial_z.json"),
      "--ns-variant", train_variant,
      "--symbolic-coverage", str(baseline.coverage),
      "--lambda-sym", str(args.lambda_sym),
      "--steps", str(args.steps),
      "--batch-size", str(args.batch_size),
      "--seq-len", str(args.seq_len),
      "--seed", str(seed),
    ]
    cmd.extend(wandb_cli_args(args, group=baseline.baseline_id, name=f"{run_name}_train", tags=[baseline.ns_variant, "train"]))
    commands.append(cmd)

  eval_out = os.path.join(run_dir, "eval.json")
  eval_cmd = [
    args.python, "evaluate_world_model.py",
    "--rnn-one-step",
    "--ns-variant", baseline.ns_variant,
    "--symbolic-coverage", str(baseline.coverage),
    "--rnn-json", rnn_json,
    "--episodes", str(args.episodes),
    "--max-steps", str(args.max_steps),
    "--horizon-steps", str(args.horizon_steps),
    "--seed", str(seed),
    "--out", eval_out,
  ]
  if args.horizons:
    eval_cmd.append("--horizons")
    eval_cmd.extend(str(horizon) for horizon in args.horizons)
  eval_cmd.extend(wandb_cli_args(args, group=baseline.baseline_id, name=f"{run_name}_eval", tags=[baseline.ns_variant, "eval"]))
  commands.append(eval_cmd)

  for cmd in commands:
    print(" ".join(cmd), flush=True)
    if not args.dry_run:
      subprocess.run(cmd, check=True)

  if os.path.exists(eval_out):
    with open(eval_out) as f:
      metrics = json.load(f)
    logger.log(metrics)
    logger.log_summary(metrics)
  logger.finish()


def wandb_cli_args(args, group, name, tags):
  if not args.wandb:
    return []
  cli = [
    "--wandb",
    "--wandb-project", args.wandb_project,
    "--wandb-group", group,
    "--wandb-name", name,
  ]
  if args.wandb_entity:
    cli.extend(["--wandb-entity", args.wandb_entity])
  if args.wandb_mode:
    cli.extend(["--wandb-mode", args.wandb_mode])
  all_tags = list(tags) + list(args.wandb_tags or [])
  if all_tags:
    cli.append("--wandb-tags")
    cli.extend(all_tags)
  return cli


if __name__ == "__main__":
  main()
