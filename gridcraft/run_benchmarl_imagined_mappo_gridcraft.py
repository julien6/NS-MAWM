from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
BENCHMARL_DIR = Path(os.environ.get("BENCHMARL_DIR", ROOT / "BenchMARL"))
sys.path.insert(0, str(BENCHMARL_DIR))
sys.path.insert(0, str(ROOT / "vGridcraft"))
sys.path.insert(0, str(ROOT / "gridcraft"))

from run_benchmarl_mappo_gridcraft import (
    MappoEvaluationVideoCallback,
    infer_ns_settings,
    patch_benchmarl_wandb_sections,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-id", required=True)
    parser.add_argument("--wm-run-dir", required=True)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--num-agents", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--max-iters", type=int, default=50)
    parser.add_argument("--frames-per-batch", type=int, default=2048)
    parser.add_argument("--mappo-minibatch-size", type=int, default=1024)
    parser.add_argument("--mappo-minibatch-iters", type=int, default=2)
    parser.add_argument("--mappo-eval-every-iters", type=int, default=25)
    parser.add_argument("--mappo-eval-episodes", type=int, default=4)
    parser.add_argument("--mappo-video-every-iters", type=int, default=250)
    parser.add_argument("--mappo-hidden-size", type=int, default=256)
    parser.add_argument("--dream-start-noise", type=float, default=1.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--save-folder", default="runs_benchmarl/imagined_mappo")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-id", default=os.environ.get("WANDB_RUN_ID"))
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument("--wandb-group", default=None)
    parser.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "ns-mawm-gridcraft"))
    parser.add_argument("--wandb-step-offset", type=int, default=int(os.environ.get("WANDB_STEP_OFFSET", "0")))
    parser.add_argument("--wandb-videos", action="store_true", default=True)
    parser.add_argument("--no-wandb-videos", dest="wandb_videos", action="store_false")
    parser.add_argument("--video-max-steps", type=int, default=100)
    parser.add_argument("--video-fps", type=int, default=10)
    args = parser.parse_args()

    try:
        from benchmarl.algorithms import MappoConfig
        from benchmarl.environments import GridcraftTask
        from benchmarl.experiment import Experiment, ExperimentConfig
        from benchmarl.experiment.logger import Logger as BenchmarlLogger
        from benchmarl.models.mlp import MlpConfig
    except ImportError as exc:
        raise SystemExit("BenchMARL/TorchRL dependencies are missing.") from exc

    if args.wandb:
        patch_benchmarl_wandb_sections(BenchmarlLogger, step_offset=args.wandb_step_offset)

    ns_variant, ns_coverage = infer_ns_settings(args.baseline_id)
    checkpoint_dir = Path(args.wm_run_dir) / "checkpoints"
    if not (checkpoint_dir / "vae.pt").exists() or not (checkpoint_dir / "rnn.pt").exists():
        raise FileNotFoundError(f"Missing world model checkpoints in {checkpoint_dir}")

    task = GridcraftTask.SURVIVAL.get_from_yaml()
    task.config.update({
        "env_kind": "dream",
        "checkpoint_dir": str(checkpoint_dir),
        "ns_variant": ns_variant,
        "ns_coverage": ns_coverage,
        "start_noise": args.dream_start_noise,
        "num_agents": args.num_agents,
        "max_steps": args.max_steps,
        "seed": args.seed,
    })
    algorithm_config = MappoConfig.get_from_yaml()
    experiment_config = ExperimentConfig.get_from_yaml()
    experiment_config.sampling_device = args.device
    experiment_config.train_device = args.device
    experiment_config.buffer_device = args.device
    experiment_config.prefer_continuous_actions = False
    experiment_config.max_n_iters = args.max_iters
    experiment_config.max_n_frames = None
    experiment_config.on_policy_collected_frames_per_batch = args.frames_per_batch
    experiment_config.on_policy_n_envs_per_worker = args.num_envs
    experiment_config.on_policy_minibatch_size = min(args.mappo_minibatch_size, args.frames_per_batch)
    experiment_config.on_policy_n_minibatch_iters = args.mappo_minibatch_iters
    experiment_config.evaluation_interval = args.frames_per_batch * max(1, args.mappo_eval_every_iters)
    experiment_config.evaluation_episodes = min(args.mappo_eval_episodes, args.num_envs)
    experiment_config.render = False
    experiment_config.loggers = ["csv", "wandb"] if args.wandb else ["csv"]
    experiment_config.project_name = args.wandb_project
    if args.wandb:
        experiment_config.wandb_extra_kwargs = {
            **experiment_config.wandb_extra_kwargs,
            **({"id": args.wandb_id, "resume": "allow"} if args.wandb_id else {}),
            **({"name": args.wandb_name} if args.wandb_name else {}),
            **({"group": args.wandb_group} if args.wandb_group else {}),
            "tags": ["gridcraft", "imagined-mappo", "world-model-policy", args.baseline_id],
        }

    save_folder = (ROOT / "gridcraft" / args.save_folder).resolve()
    save_folder.mkdir(parents=True, exist_ok=True)
    experiment_config.save_folder = str(save_folder)
    model_config = MlpConfig(num_cells=[args.mappo_hidden_size, args.mappo_hidden_size], activation_class=nn.Tanh, layer_class=nn.Linear)
    critic_model_config = MlpConfig(num_cells=[args.mappo_hidden_size, args.mappo_hidden_size], activation_class=nn.Tanh, layer_class=nn.Linear)
    experiment = Experiment(
        task=task,
        algorithm_config=algorithm_config,
        model_config=model_config,
        critic_model_config=critic_model_config,
        seed=args.seed,
        config=experiment_config,
        callbacks=[MappoEvaluationVideoCallback(args)] if args.wandb and args.wandb_videos else None,
    )
    print(
        f"=== BenchMARL MAPPO-in-WM ({args.baseline_id}) envs={args.num_envs} "
        f"agents={args.num_agents} ns={ns_variant}/k{ns_coverage} ===",
        flush=True,
    )
    experiment.run()


if __name__ == "__main__":
    main()
