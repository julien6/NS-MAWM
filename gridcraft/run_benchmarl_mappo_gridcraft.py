from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from torch import nn

ROOT = Path(__file__).resolve().parents[1]
BENCHMARL_DIR = Path(os.environ.get("BENCHMARL_DIR", ROOT / "BenchMARL"))
sys.path.insert(0, str(BENCHMARL_DIR))
sys.path.insert(0, str(ROOT / "vGridcraft"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--num-agents", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--max-iters", type=int, default=2)
    parser.add_argument("--frames-per-batch", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--save-folder", default="runs_benchmarl/native_mappo")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-id", default=os.environ.get("WANDB_RUN_ID"))
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument("--wandb-group", default=None)
    parser.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "ns-mawm-gridcraft"))
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
        patch_benchmarl_wandb_sections(BenchmarlLogger)

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
    experiment_config.on_policy_minibatch_size = min(128, args.frames_per_batch)
    experiment_config.on_policy_n_minibatch_iters = 4
    experiment_config.evaluation_interval = args.frames_per_batch
    experiment_config.evaluation_episodes = min(4, args.num_envs)
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
    model_config = MlpConfig(num_cells=[256, 256], activation_class=nn.Tanh, layer_class=nn.Linear)
    critic_model_config = MlpConfig(num_cells=[256, 256], activation_class=nn.Tanh, layer_class=nn.Linear)
    experiment = Experiment(
        task=task,
        algorithm_config=algorithm_config,
        model_config=model_config,
        critic_model_config=critic_model_config,
        seed=args.seed,
        config=experiment_config,
    )
    experiment.run()


def patch_benchmarl_wandb_sections(logger_class):
    original_log = logger_class.log

    def routed_log(self, dict_to_log, step=None):
        routed = route_benchmarl_metrics(dict_to_log)
        return original_log(self, routed, step=step)

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


if __name__ == "__main__":
    main()
