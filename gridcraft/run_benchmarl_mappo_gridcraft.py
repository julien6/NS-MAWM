from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "BenchMARL"))
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
    args = parser.parse_args()

    try:
        from benchmarl.algorithms import MappoConfig
        from benchmarl.environments import GridcraftTask
        from benchmarl.experiment import Experiment, ExperimentConfig
        from benchmarl.models.mlp import MlpConfig
    except ImportError as exc:
        raise SystemExit(
            "BenchMARL/TorchRL dependencies are missing. Run "
            "`../scripts/setup_benchmarl_gridcraft_env.sh` first."
        ) from exc

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
    experiment_config.project_name = "ns-mawm-gridcraft"
    experiment_config.save_folder = str((ROOT / "gridcraft" / args.save_folder).resolve())
    model_config = MlpConfig(num_cells=[256, 256])
    critic_model_config = MlpConfig(num_cells=[256, 256])
    experiment = Experiment(
        task=task,
        algorithm_config=algorithm_config,
        model_config=model_config,
        critic_model_config=critic_model_config,
        seed=args.seed,
        config=experiment_config,
    )
    experiment.run()


if __name__ == "__main__":
    main()
