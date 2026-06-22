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
  parser.add_argument("--phase", choices=["world_model", "policy", "all"], default="world_model")
  parser.add_argument("--policy-baseline", choices=["real_mappo", "imagined_mappo", "mpc_cem"], default=None)
  parser.add_argument("--run-dir", default="runs")
  parser.add_argument("--python", default="../.venv/bin/python")
  parser.add_argument("--steps", type=int, default=10000)
  parser.add_argument("--batch-size", type=int, default=64)
  parser.add_argument("--seq-len", type=int, default=32)
  parser.add_argument("--eval-every", type=int, default=0)
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
  parser.add_argument("--record-dir", default="record")
  parser.add_argument("--skip-extract", action="store_true")
  parser.add_argument("--extract-episodes", type=int, default=5000)
  parser.add_argument("--extract-max-steps", type=int, default=None)
  parser.add_argument("--skip-vae", action="store_true")
  parser.add_argument("--train-vae", action="store_true")
  parser.add_argument("--vae-json", default=None)
  parser.add_argument("--vae-steps", type=int, default=10000)
  parser.add_argument("--vae-batch-size", type=int, default=256)
  parser.add_argument("--policy-updates", type=int, default=100)
  parser.add_argument("--episodes-per-update", type=int, default=8)
  parser.add_argument("--policy-eval-every", type=int, default=10)
  parser.add_argument("--policy-eval-episodes", type=int, default=10)
  parser.add_argument("--planning-horizon", type=int, default=15)
  parser.add_argument("--cem-samples", type=int, default=64)
  parser.add_argument("--learning-rate", type=float, default=3e-4)
  parser.add_argument("--rnn-json", default=None)
  parser.add_argument("--initial-z-json", default=None)
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
  vae_dir = os.path.join(run_dir, "vae")
  rnn_dir = os.path.join(run_dir, "rnn")
  eval_dir = os.path.join(run_dir, "eval")
  policy_dir = os.path.join(run_dir, "policy")
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
    tags=["gridcraft", "baseline", baseline.baseline_id, baseline.ns_variant, args.phase],
  )
  logger.save_json(os.path.join(run_dir, "baseline_config.json"), config)

  train_variant = "neural" if baseline.ns_variant == "projection" else baseline.ns_variant
  rnn_file = "rnn.neural.json" if train_variant == "neural" else f"rnn.{train_variant}.json"
  rnn_json = os.path.join(rnn_dir, rnn_file)
  vae_json = resolve_vae_path(args, vae_dir)

  commands = []
  if args.phase in ("world_model", "all") and should_extract(args):
    commands.append([
      args.python, "extract.py",
      "--episodes", str(args.extract_episodes),
      "--max-steps", str(args.extract_max_steps or args.max_steps),
      "--seed", str(seed),
      "--out-dir", args.record_dir,
    ])
  if args.phase in ("world_model", "all") and should_train_vae(args, vae_json):
    vae_json = os.path.join(vae_dir, "vae.json")
    commands.append([
      args.python, "vae_train.py",
      "--record-dir", args.record_dir,
      "--out-dir", vae_dir,
      "--steps", str(args.vae_steps),
      "--batch-size", str(args.vae_batch_size),
      "--seed", str(seed),
    ])
  if args.phase in ("world_model", "all") and not args.skip_series:
    commands.append([
      args.python, "series.py",
      "--record-dir", args.record_dir,
      "--model-dir", os.path.dirname(vae_json),
      "--out-dir", series_dir,
    ] + (["--limit", str(args.series_limit)] if args.series_limit is not None else []))
  if args.phase in ("world_model", "all") and not args.skip_train:
    cmd = [
      args.python, "rnn_train.py",
      "--series", os.path.join(series_dir, "series.npz"),
      "--initial-z", os.path.join(series_dir, "initial_z.npz"),
      "--out-dir", rnn_dir,
      "--initial-z-out", os.path.join(run_dir, "initial_z.json"),
      "--ns-variant", train_variant,
      "--symbolic-coverage", str(baseline.coverage),
      "--lambda-sym", str(args.lambda_sym),
      "--vae-json", vae_json,
      "--steps", str(args.steps),
      "--batch-size", str(args.batch_size),
      "--seq-len", str(args.seq_len),
      "--seed", str(seed),
      "--python", args.python,
    ]
    if args.eval_every > 0:
      cmd.extend([
        "--eval-every", str(args.eval_every),
        "--eval-out-dir", eval_dir,
        "--eval-episodes", str(args.episodes),
        "--eval-max-steps", str(args.max_steps),
      ])
      if args.horizons:
        cmd.append("--eval-horizons")
        cmd.extend(str(horizon) for horizon in args.horizons)
    cmd.extend(wandb_cli_args(args, group=baseline.baseline_id, name=f"{run_name}_train", tags=[baseline.ns_variant, "train"]))
    commands.append(cmd)

  eval_out = os.path.join(run_dir, "eval.json")
  if args.phase in ("world_model", "all"):
    eval_cmd = [
      args.python, "evaluate_world_model.py",
      "--rnn-one-step",
      "--ns-variant", baseline.ns_variant,
      "--symbolic-coverage", str(baseline.coverage),
      "--rnn-json", rnn_json,
      "--vae-json", vae_json,
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

  if args.phase in ("policy", "all"):
    policy_baseline = resolve_policy_baseline(args, baseline)
    policy_rnn_json = resolve_rnn_path(args, baseline, rnn_json, train_variant)
    policy_initial_z_json = resolve_initial_z_path(args, run_dir)
    policy_cmd = [
      args.python, "policy_baselines.py",
      "--policy-baseline", policy_baseline,
      "--baseline-id", baseline.baseline_id,
      "--run-name", f"{run_name}_{policy_baseline}",
      "--out-dir", policy_dir,
      "--seed", str(seed),
      "--max-steps", str(args.max_steps),
      "--updates", str(args.policy_updates),
      "--episodes-per-update", str(args.episodes_per_update),
      "--eval-every", str(args.policy_eval_every),
      "--eval-episodes", str(args.policy_eval_episodes),
      "--learning-rate", str(args.learning_rate),
      "--vae-json", vae_json,
      "--rnn-json", policy_rnn_json,
      "--initial-z-json", policy_initial_z_json,
      "--ns-variant", baseline.ns_variant,
      "--symbolic-coverage", str(baseline.coverage),
      "--planning-horizon", str(args.planning_horizon),
      "--cem-samples", str(args.cem_samples),
    ]
    policy_cmd.extend(wandb_cli_args(args, group=baseline.baseline_id, name=f"{run_name}_{policy_baseline}", tags=[baseline.ns_variant, "policy", policy_baseline]))
    commands.append(policy_cmd)

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


def resolve_policy_baseline(args, baseline):
  if args.policy_baseline:
    return args.policy_baseline
  if baseline.baseline_id == "B00":
    return "real_mappo"
  return "mpc_cem"


def resolve_vae_path(args, run_vae_dir):
  if args.vae_json:
    return args.vae_json
  run_path = os.path.join(run_vae_dir, "vae.json")
  if os.path.exists(run_path):
    return run_path
  return "vae/vae.json"


def should_extract(args):
  if args.skip_extract:
    return False
  if not os.path.isdir(args.record_dir):
    return True
  return not any(name.endswith(".npz") for name in os.listdir(args.record_dir))


def should_train_vae(args, vae_json):
  if args.skip_vae:
    return False
  if args.train_vae:
    return True
  return not os.path.exists(vae_json)


def resolve_rnn_path(args, baseline, run_rnn_json, train_variant):
  if args.rnn_json:
    return args.rnn_json
  if baseline.baseline_id == "B00":
    return "rnn/rnn.json"
  if os.path.exists(run_rnn_json):
    return run_rnn_json
  candidates = {
    "neural": "rnn/rnn.neural.json",
    "regularization": "rnn/rnn.regularization.json",
    "residual": "rnn/rnn.residual.json",
  }
  return candidates.get(train_variant, "rnn/rnn.json")


def resolve_initial_z_path(args, run_dir):
  if args.initial_z_json:
    return args.initial_z_json
  run_path = os.path.join(run_dir, "initial_z.json")
  return run_path if os.path.exists(run_path) else "initial_z/initial_z.json"


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
