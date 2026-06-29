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
sys.path.insert(0, str(ROOT / "gridcraft"))

from vgridcraft import VGridcraftConfig, VectorizedGridcraftEnv
from pstr_profiles import active_rules_for_baseline, profile_name_from_baseline, rules_to_csv
from benchmarl.experiment.callback import Callback

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
        log_mambpo_imagined_evaluation(self.args, self.experiment, rollouts, iteration)
        if iteration <= 0:
            return
        if int(self.args.marl_video_every_iters) <= 0:
            return
        if iteration % int(self.args.marl_video_every_iters) != 0:
            return
        log_marl_evaluation_video(self.args, policy=self.experiment.policy, iteration=iteration)


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
    parser.add_argument("--marl-hidden-size", "--mappo-hidden-size", dest="marl_hidden_size", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--save-folder", default="runs_benchmarl/native_marl")
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
    parser.add_argument("--algorithm", choices=["mappo", "mb_mappo", "masac", "mambpo"], default="masac")
    parser.add_argument("--baseline-id", default="")
    parser.add_argument("--wm-run-dir", default=None)
    parser.add_argument("--mb-world-model-train-epochs", type=int, default=5)
    parser.add_argument("--mb-world-model-batch-size", type=int, default=256)
    parser.add_argument("--mb-world-model-hidden-size", type=int, default=256)
    parser.add_argument("--mb-imagined-horizon", type=int, default=3)
    parser.add_argument("--mb-imagined-branches", type=int, default=4)
    parser.add_argument("--mb-lambda-imagined", type=float, default=0.5)
    args = parser.parse_args()
    warn_legacy_marl_names(args.algorithm)

    try:
        from benchmarl.algorithms import MambpoConfig, MappoConfig, MasacConfig, MBMappoConfig
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
            algorithm_config.world_model.external_model_type = "gridcraft_vae_mdn_rnn"
            algorithm_config.world_model.external_checkpoint_dir = str(Path(args.wm_run_dir) / "checkpoints")
            algorithm_config.world_model.external_ns_variant = ns_variant
            algorithm_config.world_model.external_ns_coverage = ns_coverage
            algorithm_config.world_model.external_num_agents = args.num_agents
            algorithm_config.world_model.external_enabled_pstr_rules = rules_to_csv(enabled_pstr_rules)
            algorithm_config.world_model.train_epochs = 0
            algorithm_config.world_model.predict_done = True
    elif args.algorithm == "mambpo":
        algorithm_config = MambpoConfig.get_from_yaml()
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
        if args.wm_run_dir:
            ns_variant, ns_coverage = infer_ns_settings(args.baseline_id)
            enabled_pstr_rules = active_rules_for_baseline(args.baseline_id)
            checkpoint_dir = Path(args.wm_run_dir) / "checkpoints"
            algorithm_config.world_model.external_model_type = "gridcraft_vae_mdn_rnn"
            algorithm_config.world_model.external_checkpoint_dir = str(checkpoint_dir)
            algorithm_config.world_model.external_ns_variant = ns_variant
            algorithm_config.world_model.external_ns_coverage = ns_coverage
            algorithm_config.world_model.external_num_agents = args.num_agents
            algorithm_config.world_model.external_enabled_pstr_rules = rules_to_csv(enabled_pstr_rules)
            algorithm_config.world_model.train_steps = 0
            algorithm_config.world_model.predict_done = True
            if not (checkpoint_dir / "vae.pt").exists() or not (checkpoint_dir / "rnn.pt").exists():
                raise FileNotFoundError(
                    "MAMBPO model-based downstream requires trained Gridcraft "
                    f"world-model checkpoints at {checkpoint_dir}"
                )
    elif args.algorithm == "masac":
        algorithm_config = MasacConfig.get_from_yaml()
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
        f"seed={args.seed}",
        flush=True,
    )
    on_policy = algorithm_config.on_policy()
    experiment_config = ExperimentConfig.get_from_yaml()
    experiment_config.sampling_device = args.device
    experiment_config.train_device = args.device
    experiment_config.buffer_device = args.device
    experiment_config.prefer_continuous_actions = False
    experiment_config.max_n_iters = args.max_iters
    experiment_config.max_n_frames = None
    if on_policy:
        experiment_config.on_policy_collected_frames_per_batch = args.frames_per_batch
        experiment_config.on_policy_n_envs_per_worker = args.num_envs
        experiment_config.on_policy_minibatch_size = min(args.marl_train_batch_size, args.frames_per_batch)
        experiment_config.on_policy_n_minibatch_iters = args.marl_optimizer_steps
    else:
        experiment_config.off_policy_collected_frames_per_batch = args.frames_per_batch
        experiment_config.off_policy_n_envs_per_worker = args.num_envs
        experiment_config.off_policy_train_batch_size = min(args.marl_train_batch_size, args.frames_per_batch)
        experiment_config.off_policy_n_optimizer_steps = args.marl_optimizer_steps
    experiment_config.evaluation_interval = args.frames_per_batch * max(1, args.marl_eval_every_iters)
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
            "tags": ["gridcraft", args.algorithm, "real-vgridcraft", args.baseline_id or "baseline-unknown"],
        }
    save_folder = (ROOT / "gridcraft" / args.save_folder).resolve()
    save_folder.mkdir(parents=True, exist_ok=True)
    experiment_config.save_folder = str(save_folder)
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
        if key.startswith("mambpo/"):
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
