from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from wm_hpo_registry import (
    baseline_for_hpo_family,
    build_trial_summary,
    pstr_profile_for_hpo_family,
    trial_summary_path,
    write_trial_summary,
)

ENV_OVERRIDES = {
    "num_agents": ("HPO_NUM_AGENTS", int),
    "device": ("HPO_DEVICE", str),
    "seed": ("HPO_SEED", int),
    "episodes": ("HPO_EPISODES", int),
    "max_steps": ("HPO_MAX_STEPS", int),
    "num_envs": ("HPO_NUM_ENVS", int),
    "vae_steps": ("HPO_VAE_STEPS", int),
    "rnn_steps": ("HPO_RNN_STEPS", int),
    "eval_every": ("HPO_EVAL_EVERY", int),
    "video_every": ("HPO_VIDEO_EVERY", int),
    "horizons": ("HPO_HORIZONS", lambda value: [int(part) for part in str(value).replace(",", " ").split()]),
    "wm_num_workers": ("HPO_WM_NUM_WORKERS", int),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-id", default="B10_neural_k0.0")
    parser.add_argument(
        "--hpo-family",
        choices=("neural_k0.0", "structured_neural_k0.0", "regularization_k0.3", "regularization_k0.6", "residual_k0.3", "residual_k0.6"),
        default="neural_k0.0",
    )
    parser.add_argument("--num-agents", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--episodes", type=int, default=4096)
    parser.add_argument("--max-steps", type=int, default=256)
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--vae-steps", type=int, default=10000)
    parser.add_argument("--rnn-steps", type=int, default=10000)
    parser.add_argument("--eval-every", type=int, default=2500)
    parser.add_argument("--dataset-dir", default="datasets/gridcraft")
    parser.add_argument("--run-dir", default=os.environ.get("HPO_TRIALS_DIR", "runs_benchmarl_hpo"))
    parser.add_argument("--shared-model-dir", default=os.environ.get("HPO_SHARED_MODEL_DIR", "shared_models_hpo"))
    parser.add_argument("--wandb-project", default="ns-mawm-gridcraft")
    parser.add_argument("--wandb-entity", default=None)
    args = parser.parse_args()

    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError("wandb is required to run the world-model HPO sweep") from exc

    run = wandb.init(project=args.wandb_project, entity=args.wandb_entity, job_type="world_model_hpo")
    cfg = dict(wandb.config)
    fixed_config = os.environ.get("HPO_FIXED_CONFIG_JSON")
    if fixed_config:
        cfg.update(json.loads(fixed_config))
    script_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(script_dir))

    def cfg_value(name: str, default):
        env_spec = ENV_OVERRIDES.get(name)
        if env_spec is not None and env_spec[0] in os.environ:
            return env_spec[1](os.environ[env_spec[0]])
        return cfg[name] if name in cfg else default

    hpo_family = str(cfg_value("hpo_family", args.hpo_family))
    baseline_id = str(cfg_value("baseline_id", baseline_for_hpo_family(hpo_family)))
    trial_root = Path(args.run_dir) / hpo_family / (run.id or run.name or "trial")
    shared_model_dir = Path(args.shared_model_dir) / hpo_family

    runner_args = [
        "run_benchmarl_gridcraft.py",
        "--baseline-id", baseline_id,
        "--phase", "world_model",
        "--num-agents", str(cfg_value("num_agents", args.num_agents)),
        "--device", str(cfg_value("device", args.device)),
        "--seed", str(cfg_value("seed", args.seed)),
        "--episodes", str(cfg_value("episodes", args.episodes)),
        "--max-steps", str(cfg_value("max_steps", args.max_steps)),
        "--num-envs", str(cfg_value("num_envs", args.num_envs)),
        "--vae-steps", str(cfg_value("vae_steps", args.vae_steps)),
        "--rnn-steps", str(cfg_value("rnn_steps", args.rnn_steps)),
        "--eval-every", str(cfg_value("eval_every", args.eval_every)),
        "--video-every", str(cfg_value("video_every", 0)),
        "--horizons", *[str(value) for value in cfg_value("horizons", [1, 5, 10, 25])],
        "--wm-batch-size", str(cfg_value("wm_batch_size", 2048)),
        "--wm-num-workers", str(cfg_value("wm_num_workers", 4)),
        "--seq-len", str(cfg_value("seq_len", 32)),
        "--world-model-arch", str(cfg_value("world_model_arch", "structured" if hpo_family == "structured_neural_k0.0" else "vae_mdn_rnn")),
        "--learning-rate", str(cfg_value("learning_rate", 1e-3)),
        "--vae-z-size", str(cfg_value("vae_z_size", 64)),
        "--vae-hidden-size", str(cfg_value("vae_hidden_size", 512)),
        "--vae-kl-tolerance", str(cfg_value("vae_kl_tolerance", 0.5)),
        "--rnn-size", str(cfg_value("rnn_size", 128)),
        "--rnn-num-mixture", str(cfg_value("rnn_num_mixture", 5)),
        "--mean-mse-weight", str(cfg_value("mean_mse_weight", 10.0)),
        "--reward-loss-weight", str(cfg_value("reward_loss_weight", 1.0)),
        "--done-loss-weight", str(cfg_value("done_loss_weight", 1.0)),
        "--event-loss-weight", str(cfg_value("event_loss_weight", 5.0)),
        "--grid-embed-dim", str(cfg_value("grid_embed_dim", 32)),
        "--cnn-channels", str(cfg_value("cnn_channels", 128)),
        "--self-hidden-size", str(cfg_value("self_hidden_size", 128)),
        "--agent-hidden-size", str(cfg_value("agent_hidden_size", 256)),
        "--attention-heads", str(cfg_value("attention_heads", 4)),
        "--num-attention-layers", str(cfg_value("num_attention_layers", 1)),
        "--transition-hidden-size", str(cfg_value("transition_hidden_size", 256)),
        "--lambda-sym", str(cfg_value("lambda_sym", 1.0)),
        "--lambda-residual", str(cfg_value("lambda_residual", 0.25)),
        "--dataset-dir", str(args.dataset_dir),
        "--run-dir", str(trial_root),
        "--shared-model-dir", str(shared_model_dir),
        "--wandb",
        "--wandb-project", str(args.wandb_project),
        "--wandb-name", str(run.name),
        "--wandb-group", "world-model-hpo",
        "--no-wandb-videos",
    ]
    if hpo_family != "neural_k0.0":
        from pstr_profiles import profile_rules

        runner_args.extend(["--enabled-pstr-rules", *profile_rules(pstr_profile_for_hpo_family(hpo_family))])
    if args.wandb_entity:
        runner_args.extend(["--wandb-entity", str(args.wandb_entity)])

    import run_benchmarl_gridcraft

    trial_start = time.time()
    old_argv = sys.argv
    old_keep_wandb_open = os.environ.get("NS_MAWM_KEEP_WANDB_OPEN")
    try:
        sys.argv = runner_args
        os.environ["NS_MAWM_KEEP_WANDB_OPEN"] = "1"
        run_benchmarl_gridcraft.main()
    finally:
        sys.argv = old_argv
        if old_keep_wandb_open is None:
            os.environ.pop("NS_MAWM_KEEP_WANDB_OPEN", None)
        else:
            os.environ["NS_MAWM_KEEP_WANDB_OPEN"] = old_keep_wandb_open

    run_name = f"{baseline_id}_a{cfg_value('num_agents', args.num_agents)}_seed{cfg_value('seed', args.seed)}"
    completed_run_dir = trial_root / run_name
    summary_path = completed_run_dir / "eval" / "world_model_summary.json"
    metrics = {}
    if summary_path.exists():
        with summary_path.open() as handle:
            metrics = json.load(handle)
    dataset_path = metrics.get("dataset_path") or cfg.get("dataset_path")
    resolved_config = {
        **cfg,
        "baseline_id": baseline_id,
        "hpo_family": hpo_family,
        "num_agents": int(cfg_value("num_agents", args.num_agents)),
        "seed": int(cfg_value("seed", args.seed)),
        "episodes": int(cfg_value("episodes", args.episodes)),
        "max_steps": int(cfg_value("max_steps", args.max_steps)),
        "num_envs": int(cfg_value("num_envs", args.num_envs)),
        "vae_steps": int(cfg_value("vae_steps", args.vae_steps)),
        "rnn_steps": int(cfg_value("rnn_steps", args.rnn_steps)),
    }
    payload = build_trial_summary(
        hpo_family=hpo_family,
        run_dir=completed_run_dir,
        config=resolved_config,
        metrics=metrics,
        wandb_run_url=run.url,
        dataset_path=dataset_path,
        sweep_id=getattr(run, "sweep_id", None),
        trial_id=run.id,
        stage=os.environ.get("HPO_CURRENT_STAGE", "screen"),
        num_agents=int(cfg_value("num_agents", args.num_agents)),
    )
    payload["trial_wall_time"] = time.time() - trial_start
    payload["metrics"]["trial_wall_time"] = payload["trial_wall_time"]
    write_trial_summary(trial_summary_path(completed_run_dir), payload)
    if metrics:
        run.summary["hpo_family"] = hpo_family
        run.summary["wm_hpo_score"] = payload["score"]
        run.summary["trial_wall_time"] = payload["trial_wall_time"]
        run.summary["hpo_trial_summary_path"] = str(trial_summary_path(completed_run_dir))
    wandb.finish()


if __name__ == "__main__":
    main()
