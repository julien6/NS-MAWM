from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-id", required=True)
    parser.add_argument("--downstream", choices=["mambpo", "mb_mappo", "mpc_cem", "imagined_mappo", "dyna_actor_critic"], default="mambpo")
    parser.add_argument("--num-agents", type=int, default=1)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--episodes", type=int, default=1024)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--vae-steps", type=int, default=5000)
    parser.add_argument("--rnn-steps", type=int, default=5000)
    parser.add_argument("--wm-batch-size", type=int, default=512)
    parser.add_argument("--wm-num-workers", type=int, default=2)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--video-every", type=int, default=1000)
    parser.add_argument("--marl-num-envs", type=int, default=64)
    parser.add_argument("--marl-max-iters", type=int, default=50)
    parser.add_argument("--frames-per-batch", type=int, default=2048)
    parser.add_argument("--mappo-minibatch-size", type=int, default=1024)
    parser.add_argument("--mappo-minibatch-iters", type=int, default=2)
    parser.add_argument("--mappo-eval-every-iters", type=int, default=25)
    parser.add_argument("--mappo-eval-episodes", type=int, default=4)
    parser.add_argument("--mappo-video-every-iters", type=int, default=250)
    parser.add_argument("--mappo-hidden-size", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "ns-mawm-gridcraft"))
    parser.add_argument("--wandb-id", default=None)
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument("--wandb-group", default=None)
    parser.add_argument("--video-max-steps", type=int, default=100)
    parser.add_argument("--video-fps", type=int, default=10)
    args = parser.parse_args()

    old_keep = os.environ.get("NS_MAWM_KEEP_WANDB_OPEN")
    os.environ["NS_MAWM_KEEP_WANDB_OPEN"] = "1"
    run_name = args.wandb_name or f"{args.baseline_id}_a{args.num_agents}_full_seed{args.seed}"
    wandb_id = args.wandb_id or f"{run_name}_{os.getpid()}"
    try:
        wm_run_dir = Path("runs_benchmarl") / f"{args.baseline_id}_a{args.num_agents}_seed{args.seed}"
        run_world_model(args, wandb_id, run_name)
        if args.downstream == "mambpo":
            run_mambpo(args, wandb_id, run_name, wm_run_dir)
        elif args.downstream == "mb_mappo":
            run_mb_mappo(args, wandb_id, run_name, wm_run_dir)
        elif args.downstream == "mpc_cem":
            run_mpc_cem(args, wandb_id, run_name, wm_run_dir)
        elif args.downstream == "imagined_mappo":
            run_imagined_mappo(args, wandb_id, run_name, wm_run_dir)
        else:
            run_dyna_actor_critic(args, wandb_id, run_name, wm_run_dir)
    finally:
        if old_keep is None:
            os.environ.pop("NS_MAWM_KEEP_WANDB_OPEN", None)
        else:
            os.environ["NS_MAWM_KEEP_WANDB_OPEN"] = old_keep
        if args.wandb:
            try:
                import wandb
                if wandb.run is not None:
                    wandb.finish()
            except ImportError:
                pass


def invoke(module_name: str, argv: list[str]) -> None:
    module = __import__(module_name)
    old_argv = sys.argv[:]
    sys.argv = [f"{module_name}.py", *argv]
    try:
        module.main()
    finally:
        sys.argv = old_argv


def wandb_args(args, wandb_id, run_name) -> list[str]:
    if not args.wandb:
        return []
    return [
        "--wandb",
        "--wandb-project", args.wandb_project,
        "--wandb-id", wandb_id,
        "--wandb-name", run_name,
        "--wandb-group", args.wandb_group or args.baseline_id,
    ]


def run_world_model(args, wandb_id, run_name) -> None:
    invoke("run_benchmarl_gridcraft", [
        "--baseline-id", args.baseline_id,
        "--phase", "world_model",
        "--num-envs", str(args.num_envs),
        "--num-agents", str(args.num_agents),
        "--episodes", str(args.episodes),
        "--max-steps", str(args.max_steps),
        "--vae-steps", str(args.vae_steps),
        "--rnn-steps", str(args.rnn_steps),
        "--wm-batch-size", str(args.wm_batch_size),
        "--wm-num-workers", str(args.wm_num_workers),
        "--eval-every", str(args.eval_every),
        "--video-every", str(args.video_every),
        "--device", args.device,
        "--seed", str(args.seed),
        "--video-max-steps", str(args.video_max_steps),
        "--video-fps", str(args.video_fps),
        *wandb_args(args, wandb_id, run_name),
    ])


def common_mappo_args(args, wandb_id, run_name, wm_run_dir) -> list[str]:
    return [
        "--baseline-id", args.baseline_id,
        "--wm-run-dir", str(wm_run_dir),
        "--num-envs", str(args.marl_num_envs),
        "--num-agents", str(args.num_agents),
        "--max-steps", str(args.max_steps),
        "--max-iters", str(args.marl_max_iters),
        "--frames-per-batch", str(args.frames_per_batch),
        "--mappo-minibatch-size", str(args.mappo_minibatch_size),
        "--mappo-minibatch-iters", str(args.mappo_minibatch_iters),
        "--mappo-eval-every-iters", str(args.mappo_eval_every_iters),
        "--mappo-eval-episodes", str(args.mappo_eval_episodes),
        "--mappo-video-every-iters", str(args.mappo_video_every_iters),
        "--mappo-hidden-size", str(args.mappo_hidden_size),
        "--device", args.device,
        "--seed", str(args.seed),
        "--wandb-step-offset", str(args.vae_steps + args.rnn_steps + 1000),
        "--video-max-steps", str(args.video_max_steps),
        "--video-fps", str(args.video_fps),
        *wandb_args(args, wandb_id, run_name),
    ]


def run_mambpo(args, wandb_id, run_name, wm_run_dir) -> None:
    invoke("run_benchmarl_mappo_gridcraft", [
        "--algorithm", "mambpo",
        *common_mappo_args(args, wandb_id, run_name, wm_run_dir),
    ])


def run_mb_mappo(args, wandb_id, run_name, wm_run_dir) -> None:
    invoke("run_benchmarl_mappo_gridcraft", [
        "--algorithm", "mb_mappo",
        *common_mappo_args(args, wandb_id, run_name, wm_run_dir),
    ])


def run_imagined_mappo(args, wandb_id, run_name, wm_run_dir) -> None:
    invoke("run_benchmarl_imagined_mappo_gridcraft", common_mappo_args(args, wandb_id, run_name, wm_run_dir))


def run_dyna_actor_critic(args, wandb_id, run_name, wm_run_dir) -> None:
    invoke("run_benchmarl_dyna_gridcraft", [
        "--baseline-id", args.baseline_id,
        "--wm-run-dir", str(wm_run_dir),
        "--num-envs", str(args.marl_num_envs),
        "--num-agents", str(args.num_agents),
        "--max-steps", str(args.max_steps),
        "--max-iters", str(args.marl_max_iters),
        "--device", args.device,
        "--seed", str(args.seed),
        "--wandb-step-offset", str(args.vae_steps + args.rnn_steps + 1000),
        "--video-max-steps", str(args.video_max_steps),
        "--video-fps", str(args.video_fps),
        *wandb_args(args, wandb_id, run_name),
    ])


def run_mpc_cem(args, wandb_id, run_name, wm_run_dir) -> None:
    invoke("run_benchmarl_mpc_cem_gridcraft", [
        "--baseline-id", args.baseline_id,
        "--wm-run-dir", str(wm_run_dir),
        "--num-envs", str(args.marl_num_envs),
        "--num-agents", str(args.num_agents),
        "--max-steps", str(args.max_steps),
        "--device", args.device,
        "--seed", str(args.seed),
        "--wandb-step-offset", str(args.vae_steps + args.rnn_steps + 1000),
        "--video-max-steps", str(args.video_max_steps),
        "--video-fps", str(args.video_fps),
        *wandb_args(args, wandb_id, run_name),
    ])


if __name__ == "__main__":
    main()
