import argparse
import os
import sys


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--python", default=sys.executable)
  parser.add_argument("--project", default=os.environ.get("WANDB_PROJECT", "ns-mawm-gridcraft"))
  parser.add_argument("--entity", default=os.environ.get("WANDB_ENTITY"))
  parser.add_argument("--mode", default=os.environ.get("WANDB_MODE"))
  args = parser.parse_args()

  try:
    import wandb
  except ImportError as exc:
    raise SystemExit("wandb is required to run W&B sweeps") from exc

  init_kwargs = {"project": args.project}
  if args.entity:
    init_kwargs["entity"] = args.entity
  if args.mode:
    init_kwargs["mode"] = args.mode
  run = wandb.init(**init_kwargs)
  config = dict(wandb.config)
  argv = build_run_baseline_argv(args.python, config, run.name)
  print("run_baseline.py " + " ".join(argv), flush=True)
  old_argv = sys.argv
  try:
    sys.argv = ["run_baseline.py"] + argv
    from run_baseline import main as run_baseline_main
    run_baseline_main()
  finally:
    sys.argv = old_argv
  wandb.finish()


def build_run_baseline_command(python, config, sweep_run_name):
  return [python, "run_baseline.py"] + build_run_baseline_argv(python, config, sweep_run_name)


def build_run_baseline_argv(python, config, sweep_run_name):
  if "baseline_policy" in config and config["baseline_policy"]:
    baseline_id, policy_baseline = parse_baseline_policy(config["baseline_policy"])
  else:
    baseline_id = required(config, "baseline_id")
    policy_baseline = config.get("policy_baseline")
  phase = config.get("phase", "world_model")
  if phase == "policy" and baseline_id == "B00" and policy_baseline is None:
    policy_baseline = "real_mappo"
  if phase == "policy" and baseline_id != "B00" and policy_baseline is None:
    policy_baseline = "mpc_cem"

  cmd = [
    "--baseline-id", str(baseline_id),
    "--phase", str(phase),
    "--python", python,
    "--seed", str(config.get("seed", 1)),
    "--run-dir", str(config.get("run_dir", "runs/sweeps")),
    "--wandb",
    "--wandb-name", str(sweep_run_name),
    "--no-subprocess-wandb",
  ]
  if policy_baseline:
    cmd.extend(["--policy-baseline", str(policy_baseline)])

  add_int(cmd, config, "steps")
  add_int(cmd, config, "batch_size", "--batch-size")
  add_int(cmd, config, "seq_len", "--seq-len")
  add_int(cmd, config, "eval_every", "--eval-every")
  add_float(cmd, config, "lambda_sym", "--lambda-sym")
  add_int(cmd, config, "episodes")
  add_int(cmd, config, "max_steps", "--max-steps")
  add_int(cmd, config, "horizon_steps", "--horizon-steps")
  add_list(cmd, config, "horizons")
  add_int(cmd, config, "series_limit", "--series-limit")
  add_int(cmd, config, "extract_episodes", "--extract-episodes")
  add_int(cmd, config, "extract_max_steps", "--extract-max-steps")
  add_int(cmd, config, "vae_steps", "--vae-steps")
  add_int(cmd, config, "vae_batch_size", "--vae-batch-size")
  add_int(cmd, config, "policy_updates", "--policy-updates")
  add_int(cmd, config, "episodes_per_update", "--episodes-per-update")
  add_int(cmd, config, "policy_eval_every", "--policy-eval-every")
  add_int(cmd, config, "policy_eval_episodes", "--policy-eval-episodes")
  add_int(cmd, config, "planning_horizon", "--planning-horizon")
  add_int(cmd, config, "cem_samples", "--cem-samples")
  add_int(cmd, config, "video_episodes", "--video-episodes")
  add_int(cmd, config, "video_max_steps", "--video-max-steps")
  add_int(cmd, config, "video_fps", "--video-fps")
  add_float(cmd, config, "learning_rate", "--learning-rate")
  add_str(cmd, config, "record_dir", "--record-dir")
  add_str(cmd, config, "vae_json", "--vae-json")
  add_str(cmd, config, "rnn_json", "--rnn-json")
  add_str(cmd, config, "initial_z_json", "--initial-z-json")

  if bool(config.get("skip_extract", False)):
    cmd.append("--skip-extract")
  if bool(config.get("skip_vae", False)):
    cmd.append("--skip-vae")
  if bool(config.get("train_vae", False)):
    cmd.append("--train-vae")
  if bool(config.get("skip_series", False)):
    cmd.append("--skip-series")
  if bool(config.get("skip_train", False)):
    cmd.append("--skip-train")
  if bool(config.get("no_wandb_info_panels", False)):
    cmd.append("--no-wandb-info-panels")
  if bool(config.get("no_wandb_videos", False)):
    cmd.append("--no-wandb-videos")
  return cmd


def required(config, key):
  if key not in config:
    raise ValueError(f"missing required sweep parameter: {key}")
  return config[key]


def parse_baseline_policy(value):
  text = str(value)
  if ":" not in text:
    raise ValueError("baseline_policy must use '<baseline_id>:<policy_baseline>'")
  baseline_id, policy_baseline = text.split(":", 1)
  return baseline_id, policy_baseline


def add_int(cmd, config, key, cli_key=None):
  if key in config and config[key] is not None:
    cmd.extend([cli_key or f"--{key.replace('_', '-')}", str(int(config[key]))])


def add_float(cmd, config, key, cli_key=None):
  if key in config and config[key] is not None:
    cmd.extend([cli_key or f"--{key.replace('_', '-')}", str(float(config[key]))])


def add_str(cmd, config, key, cli_key=None):
  if key in config and config[key] is not None:
    cmd.extend([cli_key or f"--{key.replace('_', '-')}", str(config[key])])


def add_list(cmd, config, key, cli_key=None):
  if key not in config or config[key] is None:
    return
  value = config[key]
  if isinstance(value, str):
    items = value.replace(",", " ").split()
  else:
    items = list(value)
  cmd.append(cli_key or f"--{key.replace('_', '-')}")
  cmd.extend(str(item) for item in items)


if __name__ == "__main__":
  main()
