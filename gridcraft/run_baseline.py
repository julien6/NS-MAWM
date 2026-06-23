import argparse
import json
import os
import subprocess
import sys
import time

from experiment_config import get_baseline, list_baselines
from experiment_logging import add_wandb_args, logger_from_args, normalize_wandb_tags, should_log_wandb_videos
from progress_logging import read_progress
from wandb_schema import GENERAL, MARL_EVALUATION, MARL_TRAINING, WORLD_MODEL_EVALUATION, WORLD_MODEL_TRAINING


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--baseline-id", default=None)
  parser.add_argument("--list", action="store_true")
  parser.add_argument("--phase", choices=["world_model", "policy", "all"], default="world_model")
  parser.add_argument("--policy-baseline", choices=["all", "real_mappo", "imagined_mappo", "mpc_cem"], default=None)
  parser.add_argument("--run-dir", default="runs")
  parser.add_argument("--python", default="../.venv/bin/python")
  parser.add_argument("--steps", type=int, default=10000)
  parser.add_argument("--batch-size", type=int, default=64)
  parser.add_argument("--seq-len", type=int, default=32)
  parser.add_argument("--eval-every", type=int, default=0)
  parser.add_argument("--lambda-sym", type=float, default=1.0)
  parser.add_argument("--episodes", "--eval-episodes", dest="episodes", type=int, default=100)
  parser.add_argument("--max-steps", "--eval-max-steps", dest="max_steps", type=int, default=500)
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
  parser.add_argument("--batched-cem", action="store_true")
  batch_z_group = parser.add_mutually_exclusive_group()
  batch_z_group.add_argument("--batched-cem-deterministic-z", dest="batched_cem_deterministic_z", action="store_true", default=True)
  batch_z_group.add_argument("--batched-cem-sample-z", dest="batched_cem_deterministic_z", action="store_false")
  parser.add_argument("--batched-cem-symbolic-mode", choices=["cpu_projection"], default="cpu_projection")
  parser.add_argument("--learning-rate", type=float, default=3e-4)
  parser.add_argument("--rnn-json", default=None)
  parser.add_argument("--initial-z-json", default=None)
  parser.add_argument(
    "--subprocess-wandb",
    action="store_true",
    help="Also create separate W&B runs for training/evaluation subprocesses.",
  )
  parser.add_argument(
    "--no-subprocess-wandb",
    action="store_true",
    help=argparse.SUPPRESS,
  )
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
  run_name = f"{baseline.baseline_slug}_seed{seed}"
  run_dir = os.path.join(args.run_dir, run_name)
  series_dir = os.path.join(run_dir, "series")
  vae_dir = os.path.join(run_dir, "vae")
  rnn_dir = os.path.join(run_dir, "rnn")
  eval_dir = os.path.join(run_dir, "eval")
  policy_dir = os.path.join(run_dir, "policy")
  progress_dir = os.path.join(run_dir, "progress")
  os.makedirs(run_dir, exist_ok=True)
  os.makedirs(progress_dir, exist_ok=True)
  world_model_enabled = baseline.baseline_id != "B00"

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
    info_sections=info_sections_for_phase(args.phase),
    out_dir=run_dir,
  )
  logger.save_json(os.path.join(run_dir, "baseline_config.json"), config)

  train_variant = "neural" if baseline.ns_variant == "projection" else baseline.ns_variant
  rnn_file = "rnn.neural.json" if train_variant == "neural" else f"rnn.{train_variant}.json"
  rnn_json = os.path.join(rnn_dir, rnn_file)
  vae_json = resolve_vae_path(args, vae_dir)

  commands = []
  if world_model_enabled and args.phase in ("world_model", "all") and should_extract(args):
    cmd = [
      args.python, "extract.py",
      "--episodes", str(args.extract_episodes),
      "--max-steps", str(args.extract_max_steps or args.max_steps),
      "--seed", str(seed),
      "--out-dir", args.record_dir,
      "--progress-log", os.path.join(progress_dir, "extract.jsonl"),
    ]
    cmd.extend(wandb_cli_args(args, group=baseline.baseline_id, name=f"{run_name}_extract", tags=[baseline.ns_variant, "extract"], include=args.subprocess_wandb and not args.no_subprocess_wandb))
    commands.append(cmd)
  if world_model_enabled and args.phase in ("world_model", "all") and should_train_vae(args, vae_json):
    vae_json = os.path.join(vae_dir, "vae.json")
    cmd = [
      args.python, "vae_train.py",
      "--record-dir", args.record_dir,
      "--out-dir", vae_dir,
      "--steps", str(args.vae_steps),
      "--batch-size", str(args.vae_batch_size),
      "--seed", str(seed),
      "--progress-log", os.path.join(progress_dir, "vae.jsonl"),
    ]
    cmd.extend(wandb_cli_args(args, group=baseline.baseline_id, name=f"{run_name}_vae", tags=[baseline.ns_variant, "vae"], include=args.subprocess_wandb and not args.no_subprocess_wandb))
    commands.append(cmd)
  if world_model_enabled and args.phase in ("world_model", "all") and not args.skip_series:
    cmd = [
      args.python, "series.py",
      "--record-dir", args.record_dir,
      "--model-dir", os.path.dirname(vae_json),
      "--out-dir", series_dir,
      "--progress-log", os.path.join(progress_dir, "series.jsonl"),
    ] + (["--limit", str(args.series_limit)] if args.series_limit is not None else [])
    cmd.extend(wandb_cli_args(args, group=baseline.baseline_id, name=f"{run_name}_series", tags=[baseline.ns_variant, "series"], include=args.subprocess_wandb and not args.no_subprocess_wandb))
    commands.append(cmd)
  if world_model_enabled and args.phase in ("world_model", "all") and not args.skip_train:
    train_seq_len = effective_seq_len(args.seq_len, args.max_steps)
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
      "--seq-len", str(train_seq_len),
      "--seed", str(seed),
      "--python", args.python,
      "--progress-log", os.path.join(progress_dir, "rnn.jsonl"),
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
    cmd.extend(wandb_cli_args(args, group=baseline.baseline_id, name=f"{run_name}_train", tags=[baseline.ns_variant, "train"], include=args.subprocess_wandb and not args.no_subprocess_wandb))
    commands.append(cmd)

  eval_out = os.path.join(run_dir, "eval.json")
  if world_model_enabled and args.phase in ("world_model", "all"):
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
      "--progress-log", os.path.join(progress_dir, "eval.jsonl"),
    ]
    if args.horizons:
      eval_cmd.append("--horizons")
      eval_cmd.extend(str(horizon) for horizon in args.horizons)
    eval_cmd.extend(video_cli_args(args))
    eval_cmd.extend(wandb_cli_args(args, group=baseline.baseline_id, name=f"{run_name}_eval", tags=[baseline.ns_variant, "eval"], include=args.subprocess_wandb and not args.no_subprocess_wandb))
    commands.append(eval_cmd)

  if args.phase in ("policy", "all"):
    policy_baselines = resolve_policy_baselines(args, baseline)
    if args.phase == "all" and not args.skip_train and args.rnn_json is None:
      policy_rnn_json = rnn_json
    else:
      policy_rnn_json = resolve_rnn_path(args, baseline, rnn_json, train_variant)
    if args.phase == "all" and not args.skip_train and args.initial_z_json is None:
      policy_initial_z_json = os.path.join(run_dir, "initial_z.json")
    else:
      policy_initial_z_json = resolve_initial_z_path(args, run_dir)
    policy_specs = []
    for policy_baseline in policy_baselines:
      policy_out_dir = os.path.join(policy_dir, policy_baseline)
      policy_progress_log = os.path.join(progress_dir, f"policy_{policy_baseline}.jsonl")
      policy_cmd = [
        args.python, "policy_baselines.py",
        "--policy-baseline", policy_baseline,
        "--baseline-id", baseline.baseline_id,
        "--run-name", f"{run_name}_{policy_baseline}",
        "--out-dir", policy_out_dir,
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
        "--progress-log", policy_progress_log,
      ]
      if args.batched_cem:
        policy_cmd.append("--batched-cem")
      if not args.batched_cem_deterministic_z:
        policy_cmd.append("--batched-cem-sample-z")
      policy_cmd.extend(["--batched-cem-symbolic-mode", args.batched_cem_symbolic_mode])
      policy_cmd.extend(video_cli_args(args))
      policy_cmd.extend(wandb_cli_args(args, group=baseline.baseline_id, name=f"{run_name}_{policy_baseline}", tags=[baseline.ns_variant, "policy", policy_baseline], include=args.subprocess_wandb and not args.no_subprocess_wandb))
      commands.append(policy_cmd)
      policy_specs.append({
        "policy_baseline": policy_baseline,
        "policy_dir": policy_out_dir,
        "rnn_json": policy_rnn_json,
        "initial_z_json": policy_initial_z_json,
      })
  else:
    policy_specs = []

  for command_index, cmd in enumerate(commands, start=1):
    print(" ".join(cmd), flush=True)
    if not args.dry_run:
      stage_name = command_stage_name(cmd)
      stage_namespace = stage_namespace_for(stage_name)
      stage_start = time.time()
      logger.log({
        "pipeline_stage_index": command_index,
        "pipeline_stage_started": 1,
        f"pipeline_stage_{stage_name}_started": 1,
      }, step=command_index * 2 - 1, namespace=stage_namespace)
      run_and_relay_progress(cmd, progress_path_for_command(cmd), logger, metric_prefix=progress_metric_prefix_for_command(cmd))
      logger.log({
        "pipeline_stage_index": command_index,
        "pipeline_stage_finished": 1,
        f"pipeline_stage_{stage_name}_finished": 1,
        "pipeline_stage_duration_sec": float(time.time() - stage_start),
      }, step=command_index * 2, namespace=stage_namespace)
  if args.dry_run:
    logger.finish()
    return

  if world_model_enabled and args.phase in ("world_model", "all") and os.path.exists(eval_out):
    with open(eval_out) as f:
      metrics = json.load(f)
    logger.log(metrics, namespace="wm_evaluation")
    logger.log_summary(metrics, namespace="wm_evaluation")
    log_parent_world_model_video(logger, args, baseline, vae_json, rnn_json)
  if args.phase in ("policy", "all"):
    for spec in policy_specs:
      policy_metrics = load_policy_metrics(spec["policy_dir"])
      if policy_metrics:
        prefixed_metrics = prefix_metrics(spec["policy_baseline"], policy_metrics)
        logger.log(prefixed_metrics, namespace="marl_evaluation")
        logger.log_summary(prefixed_metrics, namespace="marl_evaluation")
      log_parent_policy_video(
        logger,
        args,
        baseline,
        spec["policy_dir"],
        vae_json,
        spec["rnn_json"],
        spec["initial_z_json"],
        spec["policy_baseline"],
      )
  logger.finish()


def resolve_policy_baselines(args, baseline):
  if args.policy_baseline and args.policy_baseline != "all":
    return [args.policy_baseline]
  if baseline.baseline_id == "B00":
    return ["real_mappo"]
  if args.policy_baseline == "all" or args.phase == "all":
    return ["imagined_mappo", "mpc_cem"]
  return ["mpc_cem"]


def command_stage_name(cmd):
  if len(cmd) < 2:
    return "unknown"
  name = os.path.basename(cmd[1])
  return os.path.splitext(name)[0].replace("-", "_")


def progress_path_for_command(cmd):
  for index, arg in enumerate(cmd):
    if arg == "--progress-log" and index + 1 < len(cmd):
      return cmd[index + 1]
  return None


def run_and_relay_progress(cmd, progress_path, logger, metric_prefix=None):
  if progress_path and os.path.exists(progress_path):
    os.remove(progress_path)
  process = subprocess.Popen(cmd)
  offset = 0
  while process.poll() is None:
    offset = relay_progress(progress_path, offset, logger, metric_prefix=metric_prefix)
    time.sleep(1.0)
  offset = relay_progress(progress_path, offset, logger, metric_prefix=metric_prefix)
  if process.returncode != 0:
    raise subprocess.CalledProcessError(process.returncode, cmd)


def relay_progress(progress_path, offset, logger, metric_prefix=None):
  offset, events = read_progress(progress_path, offset)
  for event in events:
    logger.log(
      prefix_metrics(metric_prefix, event.get("metrics", {})),
      step=event.get("step"),
      namespace=event.get("namespace"),
    )
  return offset


def progress_metric_prefix_for_command(cmd):
  if command_stage_name(cmd) != "policy_baselines":
    return None
  for index, arg in enumerate(cmd):
    if arg == "--policy-baseline" and index + 1 < len(cmd):
      return cmd[index + 1]
  return None


def prefix_metrics(prefix, metrics):
  if not prefix:
    return metrics
  return {f"{prefix}/{key}": value for key, value in metrics.items()}


def stage_namespace_for(stage_name):
  if stage_name in ("policy_baselines",):
    return "marl_training"
  if stage_name in ("evaluate_world_model",):
    return "wm_evaluation"
  return "wm_training"


def effective_seq_len(seq_len, max_steps):
  if max_steps is None:
    return seq_len
  return max(1, min(int(seq_len), int(max_steps) - 1))


def load_policy_metrics(policy_dir):
  candidates = [os.path.join(policy_dir, "mpc_cem_summary.json")]
  if os.path.isdir(policy_dir):
    eval_files = sorted(
      os.path.join(policy_dir, name)
      for name in os.listdir(policy_dir)
      if name.startswith("policy_eval_step_") and name.endswith(".json")
    )
    candidates.extend(reversed(eval_files))
  candidates.append(os.path.join(policy_dir, "policy_summary.json"))
  merged = {}
  for path in candidates:
    if os.path.exists(path):
      with open(path) as f:
        merged.update(json.load(f))
  return merged


def log_parent_world_model_video(logger, args, baseline, vae_json, rnn_json):
  if not should_log_wandb_videos(args) or not os.path.exists(rnn_json):
    return
  from video_logging import record_world_model_comparison_video
  frames = record_world_model_comparison_video(
    vae_json=vae_json,
    rnn_json=rnn_json,
    ns_variant=baseline.ns_variant,
    symbolic_coverage=baseline.coverage,
    seed=args.seed or baseline.seed,
    episodes=args.video_episodes,
    max_steps=args.video_max_steps,
  )
  logger.log_video("video_real_vs_imagined", frames, fps=args.video_fps, namespace="wm_evaluation")


def log_parent_policy_video(logger, args, baseline, policy_dir, vae_json, rnn_json, initial_z_json, policy_baseline):
  if not should_log_wandb_videos(args):
    return
  from video_logging import record_actor_policy_evaluation_video, record_mpc_cem_evaluation_video
  if policy_baseline == "mpc_cem":
    if not os.path.exists(rnn_json):
      return
    frames = record_mpc_cem_evaluation_video(
      vae_json=vae_json,
      rnn_json=rnn_json,
      ns_variant=baseline.ns_variant,
      symbolic_coverage=baseline.coverage,
      seed=args.seed or baseline.seed,
      episodes=args.video_episodes,
      max_steps=args.video_max_steps,
      planning_horizon=args.planning_horizon,
      cem_samples=args.cem_samples,
    )
  else:
    weights = os.path.join(policy_dir, "policy.weights.h5")
    if not os.path.exists(weights):
      return
    from exp_config import OBS_SIZE, Z_SIZE
    from policy_baselines import ActorCritic
    import numpy as np
    obs_size = OBS_SIZE if policy_baseline == "real_mappo" else Z_SIZE
    actor = ActorCritic(obs_size=obs_size)
    actor(np.zeros((1, obs_size), dtype=np.float32))
    actor.load_weights(weights)
    frames = record_actor_policy_evaluation_video(
      actor,
      policy_baseline=policy_baseline,
      vae_json=vae_json,
      rnn_json=rnn_json,
      ns_variant=baseline.ns_variant,
      symbolic_coverage=baseline.coverage,
      seed=args.seed or baseline.seed,
      episodes=args.video_episodes,
      max_steps=args.video_max_steps,
    )
  logger.log_video(f"{policy_baseline}/video_policy_rollout", frames, fps=args.video_fps, namespace="marl_evaluation")


def info_sections_for_phase(phase):
  if phase == "world_model":
    return [GENERAL, WORLD_MODEL_TRAINING, WORLD_MODEL_EVALUATION]
  if phase == "policy":
    return [GENERAL, MARL_TRAINING, MARL_EVALUATION]
  return [GENERAL, WORLD_MODEL_TRAINING, WORLD_MODEL_EVALUATION, MARL_TRAINING, MARL_EVALUATION]


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
  run_rnn_dir = os.path.dirname(run_rnn_json)
  run_candidates = [
    run_rnn_json,
    os.path.join(run_rnn_dir, "rnn.json"),
    os.path.join(run_rnn_dir, "rnn.neural.json"),
    os.path.join(run_rnn_dir, f"rnn.{train_variant}.json"),
  ]
  for candidate in dedupe(run_candidates):
    if os.path.exists(candidate):
      return candidate
  candidates = {
    "neural": "rnn/rnn.neural.json",
    "regularization": "rnn/rnn.regularization.json",
    "residual": "rnn/rnn.residual.json",
  }
  global_candidates = [candidates.get(train_variant, "rnn/rnn.json"), "rnn/rnn.json"]
  for candidate in dedupe(global_candidates):
    if os.path.exists(candidate):
      return candidate
  raise SystemExit(
    "No trained RNN checkpoint found for model-based policy baseline "
    f"{baseline.baseline_id}. Expected one of: "
    + ", ".join(dedupe(run_candidates + global_candidates))
    + ". Run the world-model phase first, for example: "
    f'BASELINES="{baseline.baseline_id}" WANDB=1 ./run_world_model_baselines.bash'
  )


def resolve_initial_z_path(args, run_dir):
  if args.initial_z_json:
    return args.initial_z_json
  run_path = os.path.join(run_dir, "initial_z.json")
  return run_path if os.path.exists(run_path) else "initial_z/initial_z.json"


def dedupe(items):
  result = []
  seen = set()
  for item in items:
    if item not in seen:
      result.append(item)
      seen.add(item)
  return result


def wandb_cli_args(args, group, name, tags, include=True):
  if not args.wandb or not include:
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
  if not getattr(args, "wandb_info_panels", True):
    cli.append("--no-wandb-info-panels")
  all_tags = list(tags) + list(args.wandb_tags or [])
  all_tags = normalize_wandb_tags(all_tags)
  if all_tags:
    cli.append("--wandb-tags")
    cli.extend(all_tags)
  return cli


def video_cli_args(args):
  cli = [
    "--video-episodes", str(args.video_episodes),
    "--video-max-steps", str(args.video_max_steps),
    "--video-fps", str(args.video_fps),
  ]
  if not getattr(args, "wandb_videos", True):
    cli.append("--no-wandb-videos")
  return cli


if __name__ == "__main__":
  main()
