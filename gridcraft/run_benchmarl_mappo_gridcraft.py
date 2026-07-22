from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
BENCHMARL_DIR = Path(os.environ.get("BENCHMARL_DIR", ROOT / "BenchMARL"))
sys.path.insert(0, str(BENCHMARL_DIR))
sys.path.insert(0, str(ROOT / "vGridcraft"))
sys.path.insert(0, str(ROOT / "gridcraft"))

from vgridcraft import VGridcraftConfig, VectorizedGridcraftEnv
from vgridcraft.env import ACTION_SUCCESS_EVENTS, EVENT_INDEX, EVENT_NAMES, REWARD_COMPONENT_NAMES
from pstr_profiles import active_rules_for_baseline, profile_name_from_baseline, rules_to_csv
from benchmarl.experiment.callback import Callback


def normalize_wandb_tags(tags, max_len=64):
    normalized = []
    seen = set()
    for tag in tags:
        text = str(tag).strip()
        if not text:
            continue
        if len(text) > max_len:
            text = text[:max_len]
        if text not in seen:
            normalized.append(text)
            seen.add(text)
    return normalized


ACTION_NAMES = [
    "stay",
    "move_n",
    "move_s",
    "move_w",
    "move_e",
    "harvest",
    "pickup",
    "attack",
    "eat",
    "craft_plank",
    "craft_stick",
    "craft_wood_sword",
    "craft_stone_sword",
    "craft_wood_pickaxe",
    "craft_stone_pickaxe",
]


class MarlEvaluationVideoCallback(Callback):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def on_evaluation_end(self, rollouts):
        iteration = int(self.experiment.n_iters_performed)
        self._log_hierarchy(rollouts, prefix="evaluation")
        log_mambpo_imagined_evaluation(self.args, self.experiment, rollouts, iteration)
        if iteration <= 0:
            return
        if int(self.args.marl_video_every_iters) <= 0:
            return
        if iteration % int(self.args.marl_video_every_iters) != 0:
            return
        log_marl_evaluation_video(self.args, policy=self.experiment.policy, iteration=iteration)

    def on_batch_collected(self, batch):
        self._log_hierarchy([batch], prefix="training")
        self.experiment.logger.log(
            {
                "MARL Training/entropy_profile_phase_index": entropy_profile_phase_index(
                    self.args.marl_entropy_profile_phase
                ),
            },
            step=int(self.experiment.n_iters_performed),
        )

    def _log_hierarchy(self, batches, prefix):
        metrics = hierarchy_metrics_from_tensordicts(batches, prefix=prefix)
        if metrics:
            self.experiment.logger.log(metrics, step=int(self.experiment.n_iters_performed))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--num-agents", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--max-iters", type=int, default=2)
    parser.add_argument("--frames-per-batch", type=int, default=256)
    parser.add_argument("--marl-train-batch-size", "--mappo-minibatch-size", dest="marl_train_batch_size", type=int, default=1024)
    parser.add_argument("--marl-optimizer-steps", "--mappo-minibatch-iters", dest="marl_optimizer_steps", type=int, default=2)
    parser.add_argument("--marl-eval-every-iters", "--mappo-eval-every-iters", dest="marl_eval_every_iters", type=int, default=25)
    parser.add_argument("--marl-eval-episodes", "--mappo-eval-episodes", dest="marl_eval_episodes", type=int, default=4)
    parser.add_argument("--marl-video-every-iters", "--mappo-video-every-iters", dest="marl_video_every_iters", type=int, default=250)
    parser.add_argument("--marl-model", choices=["mlp", "lstm"], default=os.environ.get("MARL_MODEL", "lstm"))
    parser.add_argument("--marl-hidden-size", "--mappo-hidden-size", dest="marl_hidden_size", type=int, default=256)
    parser.add_argument("--marl-lstm-layers", type=int, default=int(os.environ.get("MARL_LSTM_LAYERS", "1")))
    parser.add_argument("--marl-lstm-dropout", type=float, default=float(os.environ.get("MARL_LSTM_DROPOUT", "0.0")))
    parser.add_argument("--marl-lstm-compile", action="store_true", default=os.environ.get("MARL_LSTM_COMPILE", "0") == "1")
    parser.add_argument("--marl-lr", type=float, default=5e-5)
    parser.add_argument("--marl-gamma", type=float, default=0.99)
    parser.add_argument("--marl-polyak-tau", type=float, default=0.005)
    parser.add_argument("--marl-alpha-init", type=float, default=1.0)
    parser.add_argument("--marl-discrete-target-entropy-weight", type=float, default=0.2)
    parser.add_argument(
        "--marl-entropy-profile",
        choices=["standard", "anneal", "low_entropy_finetune"],
        default=os.environ.get("MARL_ENTROPY_PROFILE", "standard"),
    )
    parser.add_argument("--marl-target-entropy-weight-start", type=float, default=None)
    parser.add_argument("--marl-target-entropy-weight-end", type=float, default=None)
    parser.add_argument("--marl-alpha-init-start", type=float, default=None)
    parser.add_argument("--marl-alpha-init-end", type=float, default=None)
    parser.add_argument("--marl-finetune-iters", type=int, default=0)
    parser.add_argument("--marl-memory-size", type=int, default=1_000_000)
    parser.add_argument(
        "--restore-marl-checkpoint",
        default=os.environ.get("RESTORE_MARL_CHECKPOINT"),
        help="Optional BenchMARL checkpoint_*.pt to continue MARL training from.",
    )
    parser.add_argument("--save-marl-checkpoint", action="store_true", default=True)
    parser.add_argument("--no-save-marl-checkpoint", dest="save_marl_checkpoint", action="store_false")
    parser.add_argument("--marl-checkpoint-interval", type=int, default=0)
    parser.add_argument(
        "--include-marl-buffer-in-checkpoint",
        action="store_true",
        default=os.environ.get("MARL_INCLUDE_BUFFER_IN_CHECKPOINT", "0") == "1",
        help=(
            "Include off-policy replay buffers in BenchMARL checkpoints. Disabled "
            "by default because MASAC/MAMBPO replay buffers can make HPO checkpoints "
            "large and fragile; policy evaluation only needs model, loss and collector state."
        ),
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--save-folder", default="runs_benchmarl/native_marl")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-id", default=os.environ.get("WANDB_RUN_ID"))
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument("--wandb-group", default=None)
    parser.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "ns-mawm-gridcraft"))
    parser.add_argument("--comparison-id", default=os.environ.get("COMPARISON_ID", ""))
    parser.add_argument("--wandb-step-offset", type=int, default=int(os.environ.get("WANDB_STEP_OFFSET", "0")))
    parser.add_argument("--wandb-videos", action="store_true", default=True)
    parser.add_argument("--no-wandb-videos", dest="wandb_videos", action="store_false")
    parser.add_argument("--video-max-steps", type=int, default=100)
    parser.add_argument("--video-fps", type=int, default=10)
    parser.add_argument("--algorithm", choices=["mappo", "mb_mappo", "masac", "mambpo"], default="masac")
    parser.add_argument("--baseline-id", default="")
    parser.add_argument("--wm-run-dir", default=None)
    parser.add_argument("--mb-world-model-train-epochs", type=int, default=5)
    parser.add_argument("--mb-world-model-batch-size", type=int, default=256)
    parser.add_argument("--mb-world-model-hidden-size", type=int, default=256)
    parser.add_argument("--mb-imagined-horizon", type=int, default=3)
    parser.add_argument("--mb-imagined-branches", type=int, default=4)
    parser.add_argument("--mb-lambda-imagined", type=float, default=0.5)
    parser.add_argument("--mambpo-imagination-mode", choices=("enabled", "disabled"), default="enabled")
    parser.add_argument("--marl-hpo-core-reused", type=float, default=0.0)
    parser.add_argument("--marl-hpo-core-score", type=float, default=None)
    parser.add_argument("--marl-hpo-core-config-path", default=None)
    parser.add_argument("--marl-hpo-imagination-reused", type=float, default=0.0)
    parser.add_argument("--marl-hpo-imagination-score", type=float, default=None)
    parser.add_argument("--marl-hpo-imagination-config-path", default=None)
    parser.add_argument("--marl-hpo-core-stage", default=None)
    parser.add_argument("--marl-hpo-imagination-stage", default=None)
    parser.add_argument("--marl-hpo-imagination-checkpoint-checksum", default=None)
    parser.add_argument("--marl-hpo-strict-mode", type=float, default=0.0)
    parser.add_argument("--wm-external-reused", type=float, default=0.0)
    parser.add_argument("--wm-external-run-dir", default=None)
    parser.add_argument("--wm-external-checkpoint-checksum", default=None)
    args = parser.parse_args()
    resolve_entropy_profile(args)
    warn_legacy_marl_names(args.algorithm)

    try:
        from benchmarl.algorithms import MambpoConfig, MappoConfig, MasacConfig, MBMappoConfig
        from benchmarl.environments import GridcraftTask
        from benchmarl.experiment import Experiment, ExperimentConfig
        from benchmarl.experiment.logger import Logger as BenchmarlLogger
        from benchmarl.models.lstm import LstmConfig
        from benchmarl.models.mlp import MlpConfig
    except ImportError as exc:
        raise SystemExit(
            "BenchMARL/TorchRL dependencies are missing. Run "
            "`../scripts/setup_benchmarl_gridcraft_env.sh` first."
        ) from exc
    if args.wandb:
        patch_benchmarl_wandb_sections(BenchmarlLogger, step_offset=args.wandb_step_offset)

    task = GridcraftTask.SURVIVAL.get_from_yaml()
    task.config.update({
        "num_agents": args.num_agents,
        "max_steps": args.max_steps,
        "seed": args.seed,
    })
    if args.algorithm == "mb_mappo":
        algorithm_config = MBMappoConfig.get_from_yaml()
        algorithm_config.world_model.train_epochs = args.mb_world_model_train_epochs
        algorithm_config.world_model.batch_size = args.mb_world_model_batch_size
        algorithm_config.world_model.hidden_sizes = [
            args.mb_world_model_hidden_size,
            args.mb_world_model_hidden_size,
        ]
        algorithm_config.imagined_rollouts.horizon = args.mb_imagined_horizon
        algorithm_config.imagined_rollouts.num_branches = args.mb_imagined_branches
        algorithm_config.imagined_rollouts.lambda_imagined = args.mb_lambda_imagined
        algorithm_config.imagined_rollouts.use_for_actor = False
        if args.wm_run_dir:
            ns_variant, ns_coverage = infer_ns_settings(args.baseline_id)
            enabled_pstr_rules = active_rules_for_baseline(args.baseline_id)
            algorithm_config.world_model.external_model_type = (
                "gridcraft_structured" if "structured" in str(args.baseline_id) or str(args.baseline_id).startswith("B11") else "gridcraft_vae_mdn_rnn"
            )
            algorithm_config.world_model.external_checkpoint_dir = str(Path(args.wm_run_dir) / "checkpoints")
            algorithm_config.world_model.external_ns_variant = ns_variant
            algorithm_config.world_model.external_ns_coverage = ns_coverage
            algorithm_config.world_model.external_num_agents = args.num_agents
            algorithm_config.world_model.external_enabled_pstr_rules = rules_to_csv(enabled_pstr_rules)
            algorithm_config.world_model.train_epochs = 0
            algorithm_config.world_model.predict_done = True
    elif args.algorithm == "mambpo":
        algorithm_config = MambpoConfig.get_from_yaml()
        algorithm_config.alpha_init = args.marl_alpha_init
        algorithm_config.discrete_target_entropy_weight = args.marl_discrete_target_entropy_weight
        algorithm_config.world_model.train_steps = args.mb_world_model_train_epochs
        algorithm_config.world_model.batch_size = args.mb_world_model_batch_size
        algorithm_config.world_model.hidden_sizes = [
            args.mb_world_model_hidden_size,
            args.mb_world_model_hidden_size,
        ]
        algorithm_config.imagined_rollouts.rollout_length = args.mb_imagined_horizon
        algorithm_config.imagined_rollouts.model_batch_size = (
            args.mb_imagined_branches * args.mb_world_model_batch_size
        )
        algorithm_config.imagined_rollouts.real_ratio = max(
            0.0, min(1.0, 1.0 - args.mb_lambda_imagined)
        )
        if args.mambpo_imagination_mode == "disabled":
            algorithm_config.imagined_rollouts.real_ratio = 1.0
            algorithm_config.imagined_rollouts.model_batch_size = 0
        if args.wm_run_dir:
            ns_variant, ns_coverage = infer_ns_settings(args.baseline_id)
            enabled_pstr_rules = active_rules_for_baseline(args.baseline_id)
            checkpoint_dir = Path(args.wm_run_dir) / "checkpoints"
            algorithm_config.world_model.external_model_type = (
                "gridcraft_structured" if "structured" in str(args.baseline_id) or str(args.baseline_id).startswith("B11") else "gridcraft_vae_mdn_rnn"
            )
            algorithm_config.world_model.external_checkpoint_dir = str(checkpoint_dir)
            algorithm_config.world_model.external_ns_variant = ns_variant
            algorithm_config.world_model.external_ns_coverage = ns_coverage
            algorithm_config.world_model.external_num_agents = args.num_agents
            algorithm_config.world_model.external_enabled_pstr_rules = rules_to_csv(enabled_pstr_rules)
            algorithm_config.world_model.train_steps = 0
            algorithm_config.world_model.predict_done = True
            if algorithm_config.world_model.external_model_type == "gridcraft_structured":
                missing_external = not (checkpoint_dir / "structured_wm.pt").exists()
            else:
                missing_external = not (checkpoint_dir / "vae.pt").exists() or not (checkpoint_dir / "rnn.pt").exists()
            if missing_external:
                raise FileNotFoundError(
                    "MAMBPO model-based downstream requires trained Gridcraft "
                    f"world-model checkpoints at {checkpoint_dir}"
                )
    elif args.algorithm == "masac":
        algorithm_config = MasacConfig.get_from_yaml()
        algorithm_config.alpha_init = args.marl_alpha_init
        algorithm_config.discrete_target_entropy_weight = args.marl_discrete_target_entropy_weight
    else:
        algorithm_config = MappoConfig.get_from_yaml()
    ns_variant, ns_coverage = infer_ns_settings(args.baseline_id)
    active_pstr_rules = active_rules_for_baseline(args.baseline_id)
    checkpoint_dir = str(Path(args.wm_run_dir) / "checkpoints") if args.wm_run_dir else "none"
    print(
        "[routing] "
        f"baseline_id={args.baseline_id or 'unknown'} "
        f"algorithm={args.algorithm} "
        f"ns_variant={ns_variant} "
        f"ns_coverage={ns_coverage} "
        f"pstr_profile={profile_name_from_baseline(args.baseline_id)} "
        f"enabled_pstr_count={len(active_pstr_rules)} "
        f"wm_checkpoint_dir={checkpoint_dir} "
        f"external_wm={int(args.algorithm in {'mb_mappo', 'mambpo'} and bool(args.wm_run_dir))} "
        f"num_agents={args.num_agents} "
        f"seed={args.seed} "
        f"marl_model={args.marl_model} "
        f"entropy_profile={args.marl_entropy_profile} "
        f"entropy_phase={args.marl_entropy_profile_phase} "
        f"alpha_init={args.marl_alpha_init} "
        f"target_entropy_weight={args.marl_discrete_target_entropy_weight}",
        flush=True,
    )
    on_policy = algorithm_config.on_policy()
    experiment_config = ExperimentConfig.get_from_yaml()
    experiment_config.sampling_device = args.device
    experiment_config.train_device = args.device
    experiment_config.buffer_device = args.device
    experiment_config.lr = args.marl_lr
    experiment_config.gamma = args.marl_gamma
    experiment_config.polyak_tau = args.marl_polyak_tau
    experiment_config.off_policy_memory_size = args.marl_memory_size
    experiment_config.prefer_continuous_actions = False
    experiment_config.max_n_iters = args.max_iters
    experiment_config.max_n_frames = None
    experiment_config.restore_file = args.restore_marl_checkpoint
    experiment_config.checkpoint_at_end = bool(args.save_marl_checkpoint)
    experiment_config.checkpoint_interval = int(args.marl_checkpoint_interval)
    experiment_config.keep_checkpoints_num = 3
    experiment_config.exclude_buffer_from_checkpoint = not bool(args.include_marl_buffer_in_checkpoint)
    effective_frames_per_batch = int(args.frames_per_batch)
    if args.num_envs > 0 and effective_frames_per_batch % int(args.num_envs) != 0:
        effective_frames_per_batch = int(
            -(-effective_frames_per_batch // int(args.num_envs)) * int(args.num_envs)
        )
        print(
            "[marl] adjusted frames_per_batch "
            f"from {args.frames_per_batch} to {effective_frames_per_batch} "
            f"so it is divisible by num_envs={args.num_envs}",
            flush=True,
        )
    if on_policy:
        experiment_config.on_policy_collected_frames_per_batch = effective_frames_per_batch
        experiment_config.on_policy_n_envs_per_worker = args.num_envs
        experiment_config.on_policy_minibatch_size = min(args.marl_train_batch_size, effective_frames_per_batch)
        experiment_config.on_policy_n_minibatch_iters = args.marl_optimizer_steps
    else:
        experiment_config.off_policy_collected_frames_per_batch = effective_frames_per_batch
        experiment_config.off_policy_n_envs_per_worker = args.num_envs
        experiment_config.off_policy_train_batch_size = min(args.marl_train_batch_size, effective_frames_per_batch)
        experiment_config.off_policy_n_optimizer_steps = args.marl_optimizer_steps
    experiment_config.evaluation_interval = effective_frames_per_batch * max(1, args.marl_eval_every_iters)
    experiment_config.evaluation_episodes = min(args.marl_eval_episodes, args.num_envs)
    experiment_config.render = False
    experiment_config.loggers = ["csv", "wandb"] if args.wandb else ["csv"]
    experiment_config.project_name = args.wandb_project
    if args.wandb:
        experiment_config.wandb_extra_kwargs = {
            **experiment_config.wandb_extra_kwargs,
            **({"id": args.wandb_id, "resume": "allow"} if args.wandb_id else {}),
            **({"name": args.wandb_name} if args.wandb_name else {}),
            **({"group": args.wandb_group} if args.wandb_group else {}),
            "tags": normalize_wandb_tags([
                "gridcraft",
                args.algorithm,
                "real-vgridcraft",
                args.baseline_id or "baseline-unknown",
                *([f"comparison:{args.comparison_id}"] if args.comparison_id else []),
            ]),
            "config": {
                "baseline_id": args.baseline_id,
                "comparison_id": args.comparison_id,
                "algorithm": args.algorithm,
                "environment_dynamics_version": VGridcraftConfig.environment_dynamics_version,
                "reward_schema_version": VGridcraftConfig.reward_schema_version,
                "marl_hpo_core_reused": args.marl_hpo_core_reused,
                "marl_hpo_core_score": args.marl_hpo_core_score,
                "marl_hpo_core_config_path": args.marl_hpo_core_config_path,
                "marl_hpo_imagination_reused": args.marl_hpo_imagination_reused,
                "marl_hpo_imagination_score": args.marl_hpo_imagination_score,
                "marl_hpo_imagination_config_path": args.marl_hpo_imagination_config_path,
                "marl_hpo_core_stage": args.marl_hpo_core_stage,
                "marl_hpo_imagination_stage": args.marl_hpo_imagination_stage,
                "marl_hpo_imagination_checkpoint_checksum": args.marl_hpo_imagination_checkpoint_checksum,
                "marl_hpo_strict_mode": args.marl_hpo_strict_mode,
                "wm_external_reused": args.wm_external_reused,
                "wm_external_run_dir": args.wm_external_run_dir,
                "wm_external_checkpoint_checksum": args.wm_external_checkpoint_checksum,
                "mambpo_imagination_mode": args.mambpo_imagination_mode,
                "mambpo_imagination_used_for_training": int(
                    args.algorithm == "mambpo" and args.mambpo_imagination_mode != "disabled" and args.mb_lambda_imagined > 0.0
                ),
                "marl_lr": args.marl_lr,
                "marl_gamma": args.marl_gamma,
                "marl_polyak_tau": args.marl_polyak_tau,
                "marl_alpha_init": args.marl_alpha_init,
                "marl_discrete_target_entropy_weight": args.marl_discrete_target_entropy_weight,
                "marl_entropy_profile": args.marl_entropy_profile,
                "marl_entropy_profile_phase": args.marl_entropy_profile_phase,
                "marl_target_entropy_weight_start": args.marl_target_entropy_weight_start,
                "marl_target_entropy_weight_end": args.marl_target_entropy_weight_end,
                "marl_alpha_init_start": args.marl_alpha_init_start,
                "marl_alpha_init_end": args.marl_alpha_init_end,
                "marl_finetune_iters": args.marl_finetune_iters,
                "marl_memory_size": args.marl_memory_size,
                "marl_restore_checkpoint": args.restore_marl_checkpoint,
                "marl_include_buffer_in_checkpoint": int(bool(args.include_marl_buffer_in_checkpoint)),
                "marl_exclude_buffer_from_checkpoint": int(bool(experiment_config.exclude_buffer_from_checkpoint)),
                "marl_frames_per_batch": args.frames_per_batch,
                "marl_effective_frames_per_batch": effective_frames_per_batch,
                "marl_train_batch_size": args.marl_train_batch_size,
                "marl_optimizer_steps": args.marl_optimizer_steps,
                "marl_model": args.marl_model,
                "marl_hidden_size": args.marl_hidden_size,
                "marl_lstm_layers": args.marl_lstm_layers,
                "marl_lstm_dropout": args.marl_lstm_dropout,
                "marl_lstm_compile": int(bool(args.marl_lstm_compile)),
                "mb_imagined_horizon": args.mb_imagined_horizon,
                "mb_imagined_branches": args.mb_imagined_branches,
                "mb_lambda_imagined": args.mb_lambda_imagined,
                "mambpo_imagination_mode": args.mambpo_imagination_mode,
            },
        }
    save_folder = (ROOT / "gridcraft" / args.save_folder).resolve()
    save_folder.mkdir(parents=True, exist_ok=True)
    experiment_config.save_folder = str(save_folder)
    if args.marl_model == "lstm":
        model_config = LstmConfig(
            hidden_size=args.marl_hidden_size,
            n_layers=args.marl_lstm_layers,
            bias=True,
            dropout=args.marl_lstm_dropout,
            compile=bool(args.marl_lstm_compile),
            mlp_num_cells=[args.marl_hidden_size],
            mlp_layer_class=nn.Linear,
            mlp_activation_class=nn.Tanh,
        )
        critic_model_config = LstmConfig(
            hidden_size=args.marl_hidden_size,
            n_layers=args.marl_lstm_layers,
            bias=True,
            dropout=args.marl_lstm_dropout,
            compile=bool(args.marl_lstm_compile),
            mlp_num_cells=[args.marl_hidden_size],
            mlp_layer_class=nn.Linear,
            mlp_activation_class=nn.Tanh,
        )
    else:
        model_config = MlpConfig(num_cells=[args.marl_hidden_size, args.marl_hidden_size], activation_class=nn.Tanh, layer_class=nn.Linear)
        critic_model_config = MlpConfig(num_cells=[args.marl_hidden_size, args.marl_hidden_size], activation_class=nn.Tanh, layer_class=nn.Linear)
    experiment = Experiment(
        task=task,
        algorithm_config=algorithm_config,
        model_config=model_config,
        critic_model_config=critic_model_config,
        seed=args.seed,
        config=experiment_config,
        callbacks=[MarlEvaluationVideoCallback(args)] if args.wandb else None,
    )
    if args.wandb and os.environ.get("WANDB_MODE") == "offline":
        print(
            "[wandb] offline mode may create one local directory per process even when "
            "the same wandb id is reused. Online sync/resume merges them under the "
            "single configured run id.",
            flush=True,
        )
    experiment.run()
    write_marl_run_summary(args, experiment)


def warn_legacy_marl_names(algorithm: str) -> None:
    legacy_flags = [
        flag for flag in sys.argv[1:]
        if flag.startswith("--mappo-")
    ]
    legacy_env = [
        name for name in (
            "MAPPO_MINIBATCH_SIZE",
            "MAPPO_MINIBATCH_ITERS",
            "MAPPO_EVAL_EVERY_ITERS",
            "MAPPO_EVAL_EPISODES",
            "MAPPO_VIDEO_EVERY_ITERS",
            "MAPPO_HIDDEN_SIZE",
        )
        if name in os.environ
    ]
    if algorithm in {"masac", "mambpo"} and (legacy_flags or legacy_env):
        details = ", ".join(legacy_flags + legacy_env)
        print(
            "[naming] Deprecated MAPPO_* nomenclature used for a generic MARL "
            f"runner ({algorithm}): {details}. Prefer MARL_* env vars and "
            "--marl-* CLI flags.",
            flush=True,
        )


def resolve_entropy_profile(args) -> None:
    args.marl_entropy_profile_phase = "standard"
    if args.marl_entropy_profile == "standard":
        return
    if args.marl_entropy_profile == "anneal":
        args.marl_entropy_profile_phase = "anneal_static_consolidation_v1"
        args.marl_alpha_init = (
            args.marl_alpha_init_end
            if args.marl_alpha_init_end is not None
            else min(args.marl_alpha_init, 0.2)
        )
        args.marl_discrete_target_entropy_weight = (
            args.marl_target_entropy_weight_end
            if args.marl_target_entropy_weight_end is not None
            else min(args.marl_discrete_target_entropy_weight, 0.05)
        )
        return
    if args.marl_entropy_profile == "low_entropy_finetune":
        args.marl_entropy_profile_phase = "low_entropy_static_consolidation_v1"
        args.marl_alpha_init = (
            args.marl_alpha_init_end
            if args.marl_alpha_init_end is not None
            else min(args.marl_alpha_init, 0.05)
        )
        args.marl_discrete_target_entropy_weight = (
            args.marl_target_entropy_weight_end
            if args.marl_target_entropy_weight_end is not None
            else min(args.marl_discrete_target_entropy_weight, 0.01)
        )


def entropy_profile_phase_index(phase: str) -> float:
    return {
        "standard": 0.0,
        "anneal_static_consolidation_v1": 1.0,
        "low_entropy_static_consolidation_v1": 2.0,
    }.get(str(phase), -1.0)


def write_marl_run_summary(args, experiment) -> None:
    effective_frames_per_batch = int(args.frames_per_batch)
    if args.num_envs > 0 and effective_frames_per_batch % int(args.num_envs) != 0:
        effective_frames_per_batch = int(
            -(-effective_frames_per_batch // int(args.num_envs)) * int(args.num_envs)
        )
    metrics = {
        "mean_return": float(getattr(experiment, "mean_return", 0.0)),
        "total_frames": float(getattr(experiment, "total_frames", 0)),
        "n_iters_performed": float(getattr(experiment, "n_iters_performed", 0)),
    }
    try:
        import wandb

        if wandb.run is not None:
            for key, value in dict(wandb.run.summary).items():
                if isinstance(value, (int, float, str, bool)) or value is None:
                    metrics[str(key)] = value
    except Exception:
        pass
    save_root = (ROOT / "gridcraft" / args.save_folder).resolve()
    save_root.mkdir(parents=True, exist_ok=True)
    run_name = args.wandb_name or f"{args.algorithm}_{args.baseline_id or 'baseline'}_seed{args.seed}"
    checkpoint_candidates = sorted(
        save_root.rglob("checkpoints/checkpoint_*.pt"),
        key=lambda path: path.stat().st_mtime,
    )
    checkpoint_path = str(checkpoint_candidates[-1]) if checkpoint_candidates else None
    required_metric_keys = (
        "MARL Evaluation/eval_real_reward_auc",
        "MARL Evaluation/eval_real_reward_curve_mean",
        "MARL Evaluation/eval_real_reward_stability_std",
        "MARL Evaluation/eval_real_reward_point_count",
        "MARL Evaluation/eval_imagined_reward",
        "MARL Evaluation/real_imagined_reward_gap",
        "MARL Training/imagined_batch_size",
        "MARL Training/imagination_used_for_training",
    )
    if "MARL Evaluation/eval_real_reward_point_count" not in metrics:
        n_iters = metrics.get("n_iters_performed")
        if isinstance(n_iters, (int, float)):
            metrics["MARL Evaluation/eval_real_reward_point_count"] = float(n_iters)
    if args.algorithm == "mambpo":
        imagination_used = int(args.mambpo_imagination_mode == "enabled" and args.mb_lambda_imagined > 0.0)
        metrics.setdefault("MARL Training/imagination_used_for_training", imagination_used)
        metrics.setdefault(
            "MARL Training/imagined_batch_size",
            0.0 if not imagination_used else float(args.mb_world_model_batch_size),
        )
    missing_marl_metrics = [key for key in required_metric_keys if key not in metrics]
    for key in missing_marl_metrics:
        metrics[key] = None
    created_at = datetime.now(timezone.utc).isoformat()
    external_wm_run_dir = args.wm_external_run_dir or args.wm_run_dir
    path = save_root / f"{run_name}_marl_summary.json"
    with path.open("w") as handle:
        json.dump(
            {
                "baseline_id": args.baseline_id,
                "algorithm": args.algorithm,
                "seed": args.seed,
                "comparison_id": args.comparison_id,
                "created_at": created_at,
                "checkpoint_path": checkpoint_path,
                "marl_model": args.marl_model,
                "external_wm_run_dir": external_wm_run_dir,
                "mb_lambda_imagined": args.mb_lambda_imagined,
                "imagination_used_for_training": metrics.get("MARL Training/imagination_used_for_training"),
                "missing_marl_metrics": missing_marl_metrics,
                "config": {
                    "baseline_id": args.baseline_id,
                    "comparison_id": args.comparison_id,
                    "algorithm": args.algorithm,
                    "seed": args.seed,
                    "marl_model": args.marl_model,
                    "save_folder": args.save_folder,
                    "checkpoint_path": checkpoint_path,
                    "created_at": created_at,
                    "external_wm_run_dir": external_wm_run_dir,
                    "mb_lambda_imagined": args.mb_lambda_imagined,
                    "imagination_used_for_training": metrics.get("MARL Training/imagination_used_for_training"),
                },
                "environment_dynamics_version": VGridcraftConfig.environment_dynamics_version,
                "reward_schema_version": VGridcraftConfig.reward_schema_version,
                "metrics": metrics,
                "hyperparameters": {
                    "frames_per_batch": args.frames_per_batch,
                    "effective_frames_per_batch": effective_frames_per_batch,
                    "train_batch_size": args.marl_train_batch_size,
                    "optimizer_steps": args.marl_optimizer_steps,
                    "hidden_size": args.marl_hidden_size,
                    "lr": args.marl_lr,
                    "gamma": args.marl_gamma,
                    "polyak_tau": args.marl_polyak_tau,
                    "alpha_init": args.marl_alpha_init,
                    "discrete_target_entropy_weight": args.marl_discrete_target_entropy_weight,
                    "entropy_profile": args.marl_entropy_profile,
                    "entropy_profile_phase": args.marl_entropy_profile_phase,
                    "target_entropy_weight_start": args.marl_target_entropy_weight_start,
                    "target_entropy_weight_end": args.marl_target_entropy_weight_end,
                    "alpha_init_start": args.marl_alpha_init_start,
                    "alpha_init_end": args.marl_alpha_init_end,
                    "finetune_iters": args.marl_finetune_iters,
                    "memory_size": args.marl_memory_size,
                    "mb_imagined_horizon": args.mb_imagined_horizon,
                    "mb_imagined_branches": args.mb_imagined_branches,
                    "mb_lambda_imagined": args.mb_lambda_imagined,
                    "mb_world_model_batch_size": args.mb_world_model_batch_size,
                    "mb_world_model_train_epochs": args.mb_world_model_train_epochs,
                },
            },
            handle,
            indent=2,
        )


def patch_benchmarl_wandb_sections(logger_class, step_offset=0):
    original_log = logger_class.log

    def routed_log(self, dict_to_log, step=None):
        routed = route_benchmarl_metrics(dict_to_log)
        reward_keys = (
            "MARL Evaluation/eval_agents_reward_episode_reward_mean",
            "MARL Evaluation/eval_reward_episode_reward_mean",
        )
        reward_value = next(
            (routed[key] for key in reward_keys if key in routed),
            None,
        )
        if reward_value is not None:
            curve = getattr(self, "_gridcraft_eval_reward_curve", [])
            curve.append(float(reward_value))
            self._gridcraft_eval_reward_curve = curve
            routed["MARL Evaluation/eval_real_reward_auc"] = curve_auc(curve)
            routed["MARL Evaluation/eval_real_reward_curve_mean"] = float(
                np.mean(curve)
            )
            routed["MARL Evaluation/eval_real_reward_stability_std"] = float(
                np.std(curve[-5:])
            )
            routed["MARL Evaluation/eval_real_reward_point_count"] = float(len(curve))
        routed_step = None if step is None else int(step_offset) + int(step)
        for logger in self.loggers:
            if logger.__class__.__name__ == "WandbLogger":
                logger.experiment.log(routed, step=routed_step, commit=False)
            else:
                for key, value in routed.items():
                    logger.log_scalar(key.replace("/", "_"), value, step=routed_step)

    logger_class.log = routed_log


def curve_auc(values) -> float:
    values = np.asarray(values, dtype=np.float64)
    if values.size < 2:
        return 0.0
    return float(((values[:-1] + values[1:]) * 0.5).sum())


def route_benchmarl_metrics(metrics):
    routed = {}
    for key, value in metrics.items():
        key = str(key)
        if key.startswith("hierarchy/"):
            routed[f"Reward hierarchy diagnosis/{canonical_benchmarl_key(key.split('/', 1)[1])}"] = value
        elif key.startswith("mambpo/"):
            routed[f"MARL Training/{canonical_mambpo_key(key)}"] = value
        elif key.startswith("train/"):
            routed[f"MARL Training/{canonical_benchmarl_key(key)}"] = value
        elif key.startswith("collection/"):
            routed[f"MARL Training/{canonical_benchmarl_key(key)}"] = value
        elif key.startswith("eval/"):
            routed[f"MARL Evaluation/{canonical_benchmarl_key(key)}"] = value
        elif key.startswith("timers/evaluation"):
            routed[f"MARL Evaluation/{canonical_benchmarl_key(key)}"] = value
        elif key.startswith("timers/"):
            routed[f"MARL Training/{canonical_benchmarl_key(key)}"] = value
        elif key.startswith("counters/"):
            routed[f"MARL Training/{canonical_benchmarl_key(key)}"] = value
        else:
            routed[f"MARL Training/{canonical_benchmarl_key(key)}"] = value
    return routed


def hierarchy_metrics_from_tensordicts(tensordicts, prefix):
    events = []
    attempts = []
    components = []
    levels = []
    complexity = []
    complexity_exp = []
    complexity_unique = []
    for td in tensordicts:
        for target, key in (
            (attempts, ("next", "agents", "action_attempts")),
            (events, ("next", "agents", "event_success")),
            (components, ("next", "agents", "reward_components")),
            (levels, ("next", "agents", "task_level_max")),
            (complexity, ("next", "agents", "complexity_cumulative")),
            (complexity_exp, ("next", "agents", "complexity_exponential_cumulative")),
            (complexity_unique, ("next", "agents", "complexity_unique")),
        ):
            try:
                target.append(td.get(key).detach().float().reshape(-1, td.get(key).shape[-1]))
            except (KeyError, AttributeError):
                pass
    if not events:
        return {}
    event_tensor = torch.cat(events, dim=0)
    attempt_tensor = torch.cat(attempts, dim=0) if attempts else None
    component_tensor = torch.cat(components, dim=0)
    result = {}
    for index, name in enumerate(EVENT_NAMES):
        result[f"hierarchy/{prefix}_event_count_{name}"] = float(event_tensor[:, index].sum().cpu())
        result[f"hierarchy/{prefix}_event_rate_{name}"] = float(event_tensor[:, index].mean().cpu())
    for index, name in enumerate(REWARD_COMPONENT_NAMES):
        result[f"hierarchy/{prefix}_reward_{name}"] = float(component_tensor[:, index].sum().cpu())
    component_index = {name: index for index, name in enumerate(REWARD_COMPONENT_NAMES)}
    milestone = component_tensor[:, component_index["milestone"]]
    combat = sum(
        component_tensor[:, component_index[name]]
        for name in ("attack_hit", "mob_kill")
    )
    exploration_craft = sum(
        component_tensor[:, component_index[name]]
        for name in (
            "exploration",
            "harvest_wood",
            "harvest_apple",
            "harvest_stone",
            "pickup_item",
            "eat_apple",
            "craft_plank",
            "craft_stick",
            "craft_wood_tool",
            "craft_stone_tool",
        )
    )
    penalties = sum(
        component_tensor[:, component_index[name]]
        for name in ("mob_damage", "starvation_damage", "episode_death")
    )
    positive_without_milestone = component_tensor.clamp_min(0).sum(dim=1) - milestone.clamp_min(0)
    result[f"hierarchy/{prefix}_reward_dense"] = float(
        positive_without_milestone.sum().cpu()
    )
    result[f"hierarchy/{prefix}_reward_milestone"] = float(milestone.sum().cpu())
    result[f"hierarchy/{prefix}_reward_combat"] = float(combat.sum().cpu())
    result[f"hierarchy/{prefix}_reward_exploration_craft"] = float(
        exploration_craft.sum().cpu()
    )
    result[f"hierarchy/{prefix}_reward_penalties"] = float(penalties.sum().cpu())
    if attempt_tensor is not None:
        move_indices = [ACTION_NAMES.index(name) for name in ("move_n", "move_s", "move_w", "move_e")]
        move_attempts = float(attempt_tensor[:, move_indices].sum().cpu())
        move_successes = sum(
            float(event_tensor[:, EVENT_INDEX[event]].sum().cpu())
            for event in ("move_new_cell", "move_known_cell")
        )
        result[f"hierarchy/{prefix}_action_attempts_move"] = move_attempts
        result[f"hierarchy/{prefix}_action_success_rate_move"] = (
            move_successes / move_attempts if move_attempts > 0 else 0.0
        )
        for action_index, action_name in enumerate(ACTION_NAMES):
            if action_name.startswith("move_"):
                continue
            attempts_count = float(attempt_tensor[:, action_index].sum().cpu())
            result[f"hierarchy/{prefix}_action_attempts_{action_name}"] = attempts_count
            success_events = ACTION_SUCCESS_EVENTS.get(action_name, ())
            success_count = sum(
                float(event_tensor[:, EVENT_INDEX[event]].sum().cpu())
                for event in success_events
            )
            result[f"hierarchy/{prefix}_action_success_rate_{action_name}"] = (
                success_count / attempts_count if attempts_count > 0 else 0.0
            )
    if levels:
        result[f"hierarchy/{prefix}_task_level_max"] = float(torch.cat(levels).max().cpu())
    for name, rows in (
        ("complexity_cumulative", complexity),
        ("complexity_exponential_cumulative", complexity_exp),
        ("complexity_unique", complexity_unique),
    ):
        if rows:
            values = torch.cat(rows)
            result[f"hierarchy/{prefix}_{name}_mean"] = float(values.mean().cpu())
            result[f"hierarchy/{prefix}_{name}_max"] = float(values.max().cpu())
    return result


def canonical_mambpo_key(key):
    name = key.split("/", 1)[1]
    mapping = {
        "world_model_loss": "imagination_world_model_loss",
        "world_model_obs_loss": "imagination_world_model_obs_loss",
        "world_model_reward_loss": "imagination_world_model_reward_loss",
        "world_model_done_loss": "imagination_world_model_done_loss",
        "training_imagined_reward": "training_imagined_reward",
        "training_imagined_reward_mean_step": "training_imagined_reward_mean_step",
        "training_sampled_imagined_reward": "training_sampled_imagined_reward",
        "real_ratio": "real_ratio",
        "imagined_ratio": "imagined_ratio",
        "model_rollout_length": "model_rollout_length",
        "model_buffer_size": "model_buffer_size",
        "model_batch_size": "model_batch_size",
        "real_batch_size": "real_batch_size",
        "imagined_batch_size": "imagined_batch_size",
        "imagination_used_for_training": "imagination_used_for_training",
    }
    return mapping.get(name, f"imagination_{canonical_benchmarl_key(name)}")


def canonical_benchmarl_key(key):
    return key.replace("/", "_").replace(" ", "_")


def log_mambpo_imagined_evaluation(args, experiment, rollouts, iteration):
    algorithm = getattr(experiment, "algorithm", None)
    if not hasattr(algorithm, "evaluate_imagined_rollouts"):
        return
    if not args.wandb:
        return
    try:
        import wandb
    except ImportError:
        return
    if not rollouts:
        return
    metrics = {}
    for group in getattr(experiment, "train_group_map", {}).keys():
        try:
            group_metrics = algorithm.evaluate_imagined_rollouts(group, rollouts[0])
        except Exception as exc:
            write_imagined_eval_error(args, exc)
            group_metrics = {
                "mambpo/eval_imagined_generation_failed": torch.tensor(
                    1.0, device=algorithm.device
                ),
                "mambpo/eval_imagined_error_hash": torch.tensor(
                    float(abs(hash(str(exc))) % 1000000), device=algorithm.device
                ),
            }
        for key, value in group_metrics.items():
            value = value.mean().item() if hasattr(value, "mean") else value
            metrics[f"MARL Evaluation/{canonical_mambpo_eval_key(key)}"] = value
    if not metrics:
        return
    step = int(args.wandb_step_offset) + int(iteration)
    wandb.log(metrics, step=step, commit=False)


def write_imagined_eval_error(args, exc: Exception) -> None:
    try:
        out_dir = Path(args.wm_run_dir) if args.wm_run_dir else ROOT / "gridcraft" / args.save_folder
        out_dir.mkdir(parents=True, exist_ok=True)
        with (out_dir / "imagined_eval_errors.log").open("a", encoding="utf-8") as handle:
            handle.write(f"{type(exc).__name__}: {exc}\n")
    except Exception:
        pass


def canonical_mambpo_eval_key(key):
    name = str(key).split("/", 1)[1]
    mapping = {
        "eval_imagined_reward": "eval_imagined_reward",
        "eval_imagined_episode_length": "eval_imagined_episode_length",
        "real_imagined_reward_gap": "real_imagined_reward_gap",
        "eval_imagined_generation_failed": "eval_imagined_generation_failed",
        "eval_imagined_error_hash": "eval_imagined_error_hash",
    }
    return mapping.get(name, f"imagination_{canonical_benchmarl_key(name)}")


def infer_ns_settings(baseline_id: str) -> tuple[str, float]:
    text = str(baseline_id)
    variant = "neural"
    for candidate in ("neural", "regularization", "projection", "residual"):
        if candidate in text:
            variant = candidate
            break
    coverage = 0.0
    if "_k" in text:
        try:
            coverage = float(text.rsplit("_k", 1)[1])
        except ValueError:
            coverage = 0.0
    return variant, coverage


def log_marl_evaluation_video(args, policy=None, iteration=None):
    if not args.wandb or not args.wandb_videos:
        return
    try:
        import wandb
    except ImportError:
        return
    created_run = False
    if wandb.run is None:
        wandb.init(
            project=args.wandb_project,
            id=args.wandb_id,
            resume="allow" if args.wandb_id else None,
            name=args.wandb_name,
            group=args.wandb_group,
            tags=normalize_wandb_tags([
                "gridcraft",
                args.algorithm,
                "real-vgridcraft",
                args.baseline_id or "baseline-unknown",
                *([f"comparison:{args.comparison_id}"] if args.comparison_id else []),
            ]),
            config={"baseline_id": args.baseline_id, "comparison_id": args.comparison_id},
        )
        created_run = True
    try:
        frames = record_real_policy_video(args, policy=policy)
        video = np.transpose(np.asarray(frames, dtype=np.uint8), (0, 3, 1, 2))
        policy_iteration = int(args.max_iters) + 1 if iteration is None else int(iteration)
        step = int(args.wandb_step_offset) + policy_iteration
        wandb.log(
            {
                "MARL Evaluation/video_policy_rollout": wandb.Video(video, fps=args.video_fps, format="mp4"),
                "MARL Evaluation/video_policy_rollout_logged": 1,
                "MARL Evaluation/video_policy_rollout_frame_count": len(frames),
                "MARL Evaluation/video_policy_rollout_iteration": policy_iteration,
            },
            step=step,
        )
    except Exception as exc:
        policy_iteration = int(args.max_iters) + 1 if iteration is None else int(iteration)
        step = int(args.wandb_step_offset) + policy_iteration
        wandb.log(
            {
                "MARL Evaluation/video_policy_rollout_logged": 0,
                "MARL Evaluation/video_policy_rollout_generation_failed": 1,
                "MARL Evaluation/video_policy_rollout_error": str(exc),
                "MARL Evaluation/video_policy_rollout_iteration": policy_iteration,
            },
            step=step,
        )
    finally:
        if created_run:
            wandb.finish()


@torch.no_grad()
def record_real_policy_video(args, policy=None):
    from tensordict import TensorDict
    from torchrl.envs.utils import ExplorationType, set_exploration_type

    device = torch.device(args.device)
    config = VGridcraftConfig(num_agents=args.num_agents, max_steps=args.max_steps, seed=args.seed)
    env = VectorizedGridcraftEnv(num_envs=1, num_agents=args.num_agents, device=device, seed=args.seed + 9200, config=config)
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed + 9201)
    frames = []
    obs = env.reset()
    cumulative_reward = 0.0
    initial_frame = env.render(
        env_index=0,
        mode="rgb_array",
        overlay_info={
            "step": 0,
            "action": "initial",
            "reward": 0.0,
            "cumulative_reward": 0.0,
            "done": False,
        },
    )
    frames.append(initial_frame[:, :, :3] if initial_frame.shape[-1] == 4 else initial_frame)
    try:
        with set_exploration_type(ExplorationType.DETERMINISTIC):
            for step_index in range(max(1, int(args.video_max_steps))):
                if policy is None:
                    action = torch.randint(0, config.action_size, (1, args.num_agents), generator=generator, device=device)
                else:
                    td = TensorDict(
                        {
                            "agents": TensorDict(
                                {"observation": obs["vector"].float()},
                                batch_size=torch.Size([1, args.num_agents]),
                                device=device,
                            )
                        },
                        batch_size=torch.Size([1]),
                        device=device,
                    )
                    out = policy(td)
                    action = out.get(("agents", "action")).long()
                    if action.ndim == 3 and action.shape[-1] == 1:
                        action = action.squeeze(-1)
                obs, reward, done, truncated, _ = env.step(action)
                reward_value = float(reward.mean().detach().cpu())
                cumulative_reward += reward_value
                joint_action = {
                    f"agent_{agent_idx}": int(action[0, agent_idx].detach().cpu())
                    for agent_idx in range(args.num_agents)
                }
                episode_done = bool((done | truncated).all())
                frame = env.render(
                    env_index=0,
                    mode="rgb_array",
                    overlay_info={
                        "step": step_index + 1,
                        "action": format_joint_action(joint_action),
                        "reward": reward_value,
                        "cumulative_reward": cumulative_reward,
                        "done": episode_done,
                    },
                )
                frames.append(frame[:, :, :3] if frame.shape[-1] == 4 else frame)
                if episode_done:
                    break
    finally:
        env.close()
    return frames


def format_joint_action(joint_action: dict[str, int]) -> dict[str, str]:
    return {
        agent_id: ACTION_NAMES[action] if 0 <= int(action) < len(ACTION_NAMES) else str(int(action))
        for agent_id, action in joint_action.items()
    }


if __name__ == "__main__":
    main()
