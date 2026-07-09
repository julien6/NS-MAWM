from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from marl_hpo_registry import build_trial_summary, trial_summary_path, write_trial_summary


ENV_OVERRIDES = {
    "num_agents": ("MARL_HPO_NUM_AGENTS", int),
    "device": ("MARL_HPO_DEVICE", str),
    "seed": ("MARL_HPO_SEED", int),
    "num_envs": ("MARL_HPO_NUM_ENVS", int),
    "max_steps": ("MARL_HPO_MAX_STEPS", int),
    "max_iters": ("MARL_HPO_MAX_ITERS", int),
    "eval_every_iters": ("MARL_HPO_EVAL_EVERY_ITERS", int),
    "eval_episodes": ("MARL_HPO_EVAL_EPISODES", int),
    "video_every_iters": ("MARL_HPO_VIDEO_EVERY_ITERS", int),
    "frames_per_batch": ("MARL_HPO_FRAMES_PER_BATCH", int),
    "train_batch_size": ("MARL_HPO_TRAIN_BATCH_SIZE", int),
    "optimizer_steps": ("MARL_HPO_OPTIMIZER_STEPS", int),
    "hidden_size": ("MARL_HPO_HIDDEN_SIZE", int),
    "lr": ("MARL_HPO_LR", float),
    "gamma": ("MARL_HPO_GAMMA", float),
    "polyak_tau": ("MARL_HPO_POLYAK_TAU", float),
    "alpha_init": ("MARL_HPO_ALPHA_INIT", float),
    "discrete_target_entropy_weight": ("MARL_HPO_DISCRETE_TARGET_ENTROPY_WEIGHT", float),
    "entropy_profile": ("MARL_HPO_ENTROPY_PROFILE", str),
    "memory_size": ("MARL_HPO_MEMORY_SIZE", int),
    "mb_imagined_horizon": ("MARL_HPO_MB_IMAGINED_HORIZON", int),
    "mb_imagined_branches": ("MARL_HPO_MB_IMAGINED_BRANCHES", int),
    "mb_lambda_imagined": ("MARL_HPO_MB_LAMBDA_IMAGINED", float),
    "mb_world_model_batch_size": ("MARL_HPO_MB_WORLD_MODEL_BATCH_SIZE", int),
    "mb_world_model_train_epochs": ("MARL_HPO_MB_WORLD_MODEL_TRAIN_EPOCHS", int),
}


def _latest_checkpoint(run_dir: Path) -> Path | None:
    checkpoints = sorted(
        run_dir.glob("**/checkpoints/checkpoint_*.pt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return checkpoints[0] if checkpoints else None


def _run_posthoc_policy_eval(
    *,
    script_dir: Path,
    run_dir: Path,
    selected_config: dict,
    baseline_id: str,
) -> dict[str, float | int | str]:
    checkpoint = _latest_checkpoint(run_dir)
    if checkpoint is None:
        return {"Policy hierarchy evaluation/posthoc_failed": 1.0}
    modes = os.environ.get(
        "MARL_HPO_POLICY_EVAL_MODES",
        "deterministic,mode,temp_0.25,sampled",
    )
    episodes = int(os.environ.get("MARL_HPO_POLICY_EVAL_EPISODES", str(selected_config["eval_episodes"])))
    out_dir = run_dir / "policy_hierarchy_eval"
    import evaluate_trained_policies_hierarchy

    old_argv = sys.argv
    try:
        sys.argv = [
            "evaluate_trained_policies_hierarchy.py",
            "--checkpoint",
            str(checkpoint),
            "--baseline-id",
            baseline_id,
            "--seed",
            str(selected_config["seed"]),
            "--num-agents",
            str(selected_config["num_agents"]),
            "--episodes",
            str(episodes),
            "--max-steps",
            str(selected_config["max_steps"]),
            "--modes",
            modes,
            "--device",
            str(selected_config["device"]),
            "--out-dir",
            str(out_dir),
        ]
        evaluate_trained_policies_hierarchy.main()
    finally:
        sys.argv = old_argv
    summary_path = out_dir / "policy_hierarchy_eval_summary.json"
    if not summary_path.exists():
        return {"Policy hierarchy evaluation/posthoc_failed": 1.0}
    with summary_path.open() as handle:
        rows = json.load(handle)
    metrics: dict[str, float | int | str] = {
        "Policy hierarchy evaluation/posthoc_failed": 0.0,
        "Policy hierarchy evaluation/posthoc_checkpoint": str(checkpoint),
    }
    for row in rows:
        mode = row.get("mode", "unknown")
        for key, value in row.items():
            if key == "mode":
                continue
            if isinstance(value, (int, float, str, bool)) or value is None:
                metrics[f"Policy hierarchy evaluation/{mode}/{key}"] = value
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hpo-family", choices=("masac_core", "mambpo_imagination"), default="masac_core")
    parser.add_argument("--baseline-id", default="B00_model-free-control")
    parser.add_argument("--wm-run-dir", default=os.environ.get("MARL_HPO_WM_RUN_DIR"))
    parser.add_argument("--num-agents", type=int, default=3)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--max-iters", type=int, default=50)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--save-folder", default=os.environ.get("MARL_HPO_TRIALS_DIR", "runs_benchmarl_marl_hpo"))
    parser.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "ns-mawm-gridcraft"))
    parser.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY"))
    args = parser.parse_args()

    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError("wandb is required to run the MARL HPO sweep") from exc

    run = wandb.init(project=args.wandb_project, entity=args.wandb_entity, job_type="marl_hpo")
    cfg = dict(wandb.config)
    fixed_config = os.environ.get("MARL_HPO_FIXED_CONFIG_JSON")
    if fixed_config:
        cfg.update(json.loads(fixed_config))
    script_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(script_dir))

    def cfg_value(name: str, default):
        env_spec = ENV_OVERRIDES.get(name)
        if env_spec is not None and env_spec[0] in os.environ:
            return env_spec[1](os.environ[env_spec[0]])
        return cfg[name] if name in cfg else default

    family = str(cfg_value("hpo_family", args.hpo_family))
    algorithm = "masac" if family == "masac_core" else "mambpo"
    baseline_id = str(cfg_value("baseline_id", args.baseline_id if family == "masac_core" else "B10_neural_k0.0"))
    save_folder = Path(args.save_folder) / family / (run.id or run.name or "trial")
    wm_run_dir = str(cfg_value("wm_run_dir", args.wm_run_dir or ""))
    if family == "mambpo_imagination" and not wm_run_dir:
        raise FileNotFoundError("mambpo_imagination HPO requires --wm-run-dir or MARL_HPO_WM_RUN_DIR")

    selected_config = {
        "hpo_family": family,
        "baseline_id": baseline_id,
        "num_agents": cfg_value("num_agents", args.num_agents),
        "num_envs": cfg_value("num_envs", args.num_envs),
        "max_steps": cfg_value("max_steps", args.max_steps),
        "max_iters": cfg_value("max_iters", args.max_iters),
        "frames_per_batch": cfg_value("frames_per_batch", 4096),
        "train_batch_size": cfg_value("train_batch_size", 1024),
        "optimizer_steps": cfg_value("optimizer_steps", 4),
        "eval_every_iters": cfg_value("eval_every_iters", 5),
        "eval_episodes": cfg_value("eval_episodes", 4),
        "video_every_iters": cfg_value("video_every_iters", 0),
        "hidden_size": cfg_value("hidden_size", 256),
        "lr": cfg_value("lr", 5e-5),
        "gamma": cfg_value("gamma", 0.99),
        "polyak_tau": cfg_value("polyak_tau", 0.005),
        "alpha_init": cfg_value("alpha_init", 1.0),
        "discrete_target_entropy_weight": cfg_value("discrete_target_entropy_weight", 0.2),
        "entropy_profile": cfg_value("entropy_profile", "standard"),
        "memory_size": cfg_value("memory_size", 1_000_000),
        "mb_world_model_train_epochs": cfg_value("mb_world_model_train_epochs", 0 if family == "mambpo_imagination" else 5),
        "mb_world_model_batch_size": cfg_value("mb_world_model_batch_size", 256),
        "mb_world_model_hidden_size": cfg_value("mb_world_model_hidden_size", 256),
        "mb_imagined_horizon": cfg_value("mb_imagined_horizon", 3),
        "mb_imagined_branches": cfg_value("mb_imagined_branches", 4),
        "mb_lambda_imagined": cfg_value("mb_lambda_imagined", 0.5),
        "device": cfg_value("device", args.device),
        "seed": cfg_value("seed", args.seed),
    }
    runner_args = [
        "run_benchmarl_marl_gridcraft.py",
        "--algorithm", algorithm,
        "--baseline-id", baseline_id,
        "--num-agents", str(selected_config["num_agents"]),
        "--num-envs", str(selected_config["num_envs"]),
        "--max-steps", str(selected_config["max_steps"]),
        "--max-iters", str(selected_config["max_iters"]),
        "--frames-per-batch", str(selected_config["frames_per_batch"]),
        "--marl-train-batch-size", str(selected_config["train_batch_size"]),
        "--marl-optimizer-steps", str(selected_config["optimizer_steps"]),
        "--marl-eval-every-iters", str(selected_config["eval_every_iters"]),
        "--marl-eval-episodes", str(selected_config["eval_episodes"]),
        "--marl-video-every-iters", str(selected_config["video_every_iters"]),
        "--marl-hidden-size", str(selected_config["hidden_size"]),
        "--marl-lr", str(selected_config["lr"]),
        "--marl-gamma", str(selected_config["gamma"]),
        "--marl-polyak-tau", str(selected_config["polyak_tau"]),
        "--marl-alpha-init", str(selected_config["alpha_init"]),
        "--marl-discrete-target-entropy-weight", str(selected_config["discrete_target_entropy_weight"]),
        "--marl-entropy-profile", str(selected_config["entropy_profile"]),
        "--marl-memory-size", str(selected_config["memory_size"]),
        "--mb-world-model-train-epochs", str(selected_config["mb_world_model_train_epochs"]),
        "--mb-world-model-batch-size", str(selected_config["mb_world_model_batch_size"]),
        "--mb-world-model-hidden-size", str(selected_config["mb_world_model_hidden_size"]),
        "--mb-imagined-horizon", str(selected_config["mb_imagined_horizon"]),
        "--mb-imagined-branches", str(selected_config["mb_imagined_branches"]),
        "--mb-lambda-imagined", str(selected_config["mb_lambda_imagined"]),
        "--device", str(selected_config["device"]),
        "--seed", str(selected_config["seed"]),
        "--save-folder", str(save_folder),
        "--wandb",
        "--wandb-project", str(args.wandb_project),
        "--wandb-id", str(run.id),
        "--wandb-name", str(run.name),
        "--wandb-group", "marl-hpo",
        "--no-wandb-videos",
    ]
    if family == "mambpo_imagination":
        runner_args.extend(["--wm-run-dir", wm_run_dir])

    import run_benchmarl_marl_gridcraft

    trial_start = time.time()
    old_argv = sys.argv
    old_keep_wandb_open = os.environ.get("NS_MAWM_KEEP_WANDB_OPEN")
    try:
        sys.argv = runner_args
        os.environ["NS_MAWM_KEEP_WANDB_OPEN"] = "1"
        run_benchmarl_marl_gridcraft.main()
    finally:
        sys.argv = old_argv
        if old_keep_wandb_open is None:
            os.environ.pop("NS_MAWM_KEEP_WANDB_OPEN", None)
        else:
            os.environ["NS_MAWM_KEEP_WANDB_OPEN"] = old_keep_wandb_open

    metrics = {}
    if wandb.run is not None:
        for key, value in dict(wandb.run.summary).items():
            if isinstance(value, (int, float, str, bool)) or value is None:
                metrics[str(key)] = value
    summary_root = script_dir / save_folder
    summary_files = sorted(summary_root.glob("*_marl_summary.json"))
    if summary_files:
        with summary_files[-1].open() as handle:
            local_summary = json.load(handle)
        metrics.update(local_summary.get("metrics", {}))
    try:
        posthoc_metrics = _run_posthoc_policy_eval(
            script_dir=script_dir,
            run_dir=summary_root,
            selected_config=selected_config,
            baseline_id=baseline_id,
        )
        metrics.update(posthoc_metrics)
        numeric_posthoc = {
            key: value
            for key, value in posthoc_metrics.items()
            if isinstance(value, (int, float))
        }
        if numeric_posthoc:
            wandb.log(numeric_posthoc)
    except Exception as exc:
        metrics["Policy hierarchy evaluation/posthoc_failed"] = 1.0
        metrics["Policy hierarchy evaluation/posthoc_error"] = repr(exc)
        wandb.log({"Policy hierarchy evaluation/posthoc_failed": 1.0})
    payload = build_trial_summary(
        family=family,
        run_dir=summary_root,
        config={**cfg, **selected_config},
        metrics=metrics,
        wandb_run_url=run.url,
        trial_id=run.id,
        sweep_id=getattr(run, "sweep_id", None),
        stage=os.environ.get("MARL_HPO_CURRENT_STAGE", "screen"),
        num_agents=int(selected_config["num_agents"]),
        external_checkpoint_dir=(
            str(Path(wm_run_dir) / "checkpoints") if family == "mambpo_imagination" else None
        ),
    )
    payload["trial_wall_time"] = time.time() - trial_start
    payload["metrics"]["trial_wall_time"] = payload["trial_wall_time"]
    write_trial_summary(trial_summary_path(summary_root), payload)
    run.summary["marl_hpo_family"] = family
    run.summary["marl_hpo_score"] = payload["score"]
    run.summary["trial_wall_time"] = payload["trial_wall_time"]
    run.summary["marl_hpo_trial_summary_path"] = str(trial_summary_path(summary_root))
    wandb.finish()


if __name__ == "__main__":
    main()
