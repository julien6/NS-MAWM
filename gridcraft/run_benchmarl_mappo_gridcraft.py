from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
BENCHMARL_DIR = Path(os.environ.get("BENCHMARL_DIR", ROOT / "BenchMARL"))
sys.path.insert(0, str(BENCHMARL_DIR))
sys.path.insert(0, str(ROOT / "vGridcraft"))

from vgridcraft import VGridcraftConfig, VectorizedGridcraftEnv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--num-agents", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--max-iters", type=int, default=2)
    parser.add_argument("--frames-per-batch", type=int, default=256)
    parser.add_argument("--mappo-minibatch-size", type=int, default=1024)
    parser.add_argument("--mappo-minibatch-iters", type=int, default=2)
    parser.add_argument("--mappo-eval-every-iters", type=int, default=25)
    parser.add_argument("--mappo-eval-episodes", type=int, default=4)
    parser.add_argument("--mappo-hidden-size", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--save-folder", default="runs_benchmarl/native_mappo")
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
    )
    experiment.run()
    log_marl_evaluation_video(args)


def patch_benchmarl_wandb_sections(logger_class, step_offset=0):
    original_log = logger_class.log

    def routed_log(self, dict_to_log, step=None):
        routed = route_benchmarl_metrics(dict_to_log)
        routed_step = None if step is None else int(step_offset) + int(step)
        for logger in self.loggers:
            if logger.__class__.__name__ == "WandbLogger":
                logger.experiment.log(routed, step=routed_step, commit=False)
            else:
                for key, value in routed.items():
                    logger.log_scalar(key.replace("/", "_"), value, step=routed_step)

    logger_class.log = routed_log


def route_benchmarl_metrics(metrics):
    routed = {}
    for key, value in metrics.items():
        key = str(key)
        if key.startswith("train/"):
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


def canonical_benchmarl_key(key):
    return key.replace("/", "_").replace(" ", "_")


def log_marl_evaluation_video(args):
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
        )
        created_run = True
    try:
        frames = record_real_policy_video(args)
        video = np.transpose(np.asarray(frames, dtype=np.uint8), (0, 3, 1, 2))
        step = int(args.wandb_step_offset) + int(args.max_iters) * int(args.frames_per_batch) + 1
        wandb.log(
            {
                "MARL Evaluation/video_policy_rollout": wandb.Video(video, fps=args.video_fps, format="mp4"),
                "MARL Evaluation/video_policy_rollout_logged": 1,
                "MARL Evaluation/video_policy_rollout_frame_count": len(frames),
            },
            step=step,
        )
    except Exception as exc:
        step = int(args.wandb_step_offset) + int(args.max_iters) * int(args.frames_per_batch) + 1
        wandb.log(
            {
                "MARL Evaluation/video_policy_rollout_logged": 0,
                "MARL Evaluation/video_policy_rollout_generation_failed": 1,
                "MARL Evaluation/video_policy_rollout_error": str(exc),
            },
            step=step,
        )
    finally:
        if created_run:
            wandb.finish()


@torch.no_grad()
def record_real_policy_video(args):
    device = torch.device(args.device)
    config = VGridcraftConfig(num_agents=args.num_agents, max_steps=args.max_steps, seed=args.seed)
    env = VectorizedGridcraftEnv(num_envs=1, num_agents=args.num_agents, device=device, seed=args.seed + 9200, config=config)
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed + 9201)
    frames = []
    env.reset()
    for _ in range(max(1, int(args.video_max_steps))):
        action = torch.randint(0, config.action_size, (1, args.num_agents), generator=generator, device=device)
        _, _, done, truncated, _ = env.step(action)
        frame = env.render(env_index=0, mode="rgb_array")
        frames.append(frame[:, :, :3] if frame.shape[-1] == 4 else frame)
        if bool((done | truncated).all()):
            break
    env.close()
    return frames


if __name__ == "__main__":
    main()
