from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vGridcraft"))
sys.path.insert(0, str(ROOT / "gridcraft"))

from experiment_logging import add_wandb_args, logger_from_args, should_log_wandb_videos
from ns_symbolic import NS_VARIANTS, apply_symbolic_projection, tabular_to_vector
from torch_world_model import TorchGridcraftRNN, TorchGridcraftVAE
from torch_world_model.models import ACTION_SIZE
from vgridcraft import VGridcraftConfig, VectorizedGridcraftEnv
from wandb_schema import GENERAL, MARL_EVALUATION
from run_benchmarl_dyna_gridcraft import apply_ns_mawm_to_latent_step


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-id", required=True)
    parser.add_argument("--wm-run-dir", required=True)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--num-agents", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--planning-horizon", type=int, default=15)
    parser.add_argument("--cem-samples", type=int, default=128)
    parser.add_argument("--cem-iters", type=int, default=3)
    parser.add_argument("--cem-elite-frac", type=float, default=0.2)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--wandb-step-offset", type=int, default=int(os.environ.get("WANDB_STEP_OFFSET", "0")))
    add_wandb_args(parser)
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    config = VGridcraftConfig(num_agents=args.num_agents, max_steps=args.max_steps, seed=args.seed)
    ns_variant, ns_coverage = infer_ns_settings(args.baseline_id)
    run_dir = Path(args.wm_run_dir)
    checkpoint_dir = run_dir / "checkpoints"

    vae = TorchGridcraftVAE().to(device)
    rnn = TorchGridcraftRNN().to(device)
    vae_path = checkpoint_dir / "vae.pt"
    rnn_path = checkpoint_dir / "rnn.pt"
    if not vae_path.exists() or not rnn_path.exists():
        raise FileNotFoundError(f"Missing world model checkpoints: {vae_path} / {rnn_path}")
    vae.load_state_dict(torch.load(vae_path, map_location=device))
    rnn.load_state_dict(torch.load(rnn_path, map_location=device))
    vae.eval()
    rnn.eval()

    logger = logger_from_args(
        args,
        config={
            **vars(args),
            "downstream_policy_backend": "mpc_cem_with_trained_world_model",
            "world_model_checkpoint_dir": str(checkpoint_dir),
            "gridcraft_config": config.__dict__,
            "ns_variant": ns_variant,
            "ns_coverage": ns_coverage,
        },
        default_group=args.baseline_id,
        default_name=f"{args.baseline_id}_a{args.num_agents}_mpc_cem_seed{args.seed}",
        tags=["gridcraft", "mpc-cem", "world-model-policy", args.baseline_id],
        info_sections=[GENERAL, MARL_EVALUATION],
        out_dir=str(run_dir / "mpc_cem_policy"),
    )

    metrics, frames = evaluate_mpc_cem(vae, rnn, args, config, device, ns_variant, ns_coverage)
    step = int(args.wandb_step_offset) + 1
    logger.log(metrics, step=step, namespace="marl_evaluation")
    if should_log_wandb_videos(args) and frames:
        logged = logger.log_video(
            "video_policy_rollout",
            frames,
            fps=args.video_fps,
            step=step,
            namespace="marl_evaluation",
        )
        logger.log(
            {
                "video_policy_rollout_logged": int(bool(logged)),
                "video_policy_rollout_frame_count": len(frames),
            },
            step=step,
            namespace="marl_evaluation",
        )
    out_dir = run_dir / "mpc_cem_policy"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "mpc_cem_summary.json").write_text(json.dumps(metrics, indent=2))
    logger.log_summary(metrics, namespace="marl_evaluation")
    logger.finish()


@torch.no_grad()
def evaluate_mpc_cem(vae, rnn, args, config, device, ns_variant, ns_coverage):
    env = VectorizedGridcraftEnv(
        num_envs=args.num_envs,
        num_agents=args.num_agents,
        device=device,
        seed=args.seed + 31000,
        config=config,
    )
    obs = env.reset()
    returns = torch.zeros((args.num_envs, args.num_agents), device=device)
    lengths = torch.zeros((args.num_envs,), device=device)
    active_env = torch.ones((args.num_envs,), dtype=torch.bool, device=device)
    frames = []
    imagined_returns = []
    try:
        for step in range(max(1, int(args.max_steps))):
            flat_obs = obs["vector"].reshape(args.num_envs * args.num_agents, -1).float()
            z = vae.encode(flat_obs, sample=False)
            action, imagined_return = cem_plan_actions(vae, rnn, z, args, device, ns_variant, ns_coverage)
            action = action.reshape(args.num_envs, args.num_agents)
            imagined_returns.append(imagined_return.detach())
            obs, reward, done, truncated, _ = env.step(action)
            returns += reward * active_env[:, None].float()
            lengths += active_env.float()
            active_env &= ~(done | truncated)
            if should_record_frame(args, step):
                frame = env.render(env_index=0, mode="rgb_array")
                frames.append(frame[:, :, :3] if frame.shape[-1] == 4 else frame)
            if not bool(active_env.any()):
                break
    finally:
        env.close()
    imagined_return_value = torch.stack(imagined_returns).mean() if imagined_returns else torch.tensor(float("nan"))
    return {
        "planning_real_return": float(returns.mean().detach().cpu()),
        "eval_real_reward": float(returns.mean().detach().cpu()),
        "episode_length": float(lengths.mean().detach().cpu()),
        "planning_imagined_return": float(imagined_return_value.detach().cpu()),
        "policy_backend": "mpc_cem_with_trained_world_model",
        "real_interaction_ratio": 1.0,
        "imagined_planning_ratio": 1.0,
    }, frames


def should_record_frame(args, step: int) -> bool:
    if not should_log_wandb_videos(args):
        return False
    return step < max(1, int(args.video_max_steps))


@torch.no_grad()
def cem_plan_actions(vae, rnn, z, args, device, ns_variant="neural", ns_coverage=0.0):
    num_agents = int(args.num_agents)
    if z.shape[0] % num_agents != 0:
        raise ValueError(f"MPC-CEM latent batch {z.shape[0]} is not divisible by num_agents={num_agents}")
    env_batch = z.shape[0] // num_agents
    horizon = max(1, int(args.planning_horizon))
    samples = max(2, int(args.cem_samples))
    elite_count = max(1, int(samples * float(args.cem_elite_frac)))
    probs = torch.full((env_batch, num_agents, horizon, ACTION_SIZE), 1.0 / ACTION_SIZE, device=device)
    generator = torch.Generator(device=device)
    generator.manual_seed(int(args.seed) + int(torch.randint(0, 1_000_000, ()).cpu()))
    for _ in range(max(1, int(args.cem_iters))):
        sampled = torch.multinomial(
            probs.reshape(env_batch * num_agents * horizon, ACTION_SIZE),
            num_samples=samples,
            replacement=True,
            generator=generator,
        ).reshape(env_batch, num_agents, horizon, samples).permute(0, 3, 1, 2)
        returns = rollout_action_sequences(vae, rnn, z, sampled, args, ns_variant, ns_coverage)
        elite = returns.topk(elite_count, dim=1).indices
        elite_actions = torch.gather(
            sampled,
            dim=1,
            index=elite[:, :, None, None].expand(-1, -1, num_agents, horizon),
        )
        counts = torch.zeros_like(probs)
        counts.scatter_add_(
            3,
            elite_actions.permute(0, 2, 3, 1),
            torch.ones((env_batch, num_agents, horizon, elite_count), device=device),
        )
        probs = (counts + float(args.temperature)).div(
            counts.sum(dim=-1, keepdim=True) + float(args.temperature) * ACTION_SIZE
        )
    greedy = probs.argmax(dim=-1).permute(0, 2, 1)[:, None, :, :]
    final_returns = rollout_action_sequences(vae, rnn, z, greedy, args, ns_variant, ns_coverage)
    first_action = probs[:, :, 0].argmax(dim=-1)
    return first_action.reshape(env_batch * num_agents), final_returns.reshape(-1).mean()


@torch.no_grad()
def rollout_action_sequences(vae, rnn, z, sequences, args, ns_variant="neural", ns_coverage=0.0):
    env_batch, samples, num_agents, horizon = sequences.shape
    current_z = (
        z.reshape(env_batch, num_agents, -1)
        .unsqueeze(1)
        .expand(env_batch, samples, num_agents, z.shape[-1])
        .reshape(env_batch * samples * num_agents, -1)
    )
    flat_sequences = sequences.reshape(env_batch * samples * num_agents, horizon)
    returns = torch.zeros((env_batch * samples * num_agents,), device=z.device)
    discount = torch.ones_like(returns)
    state = None
    ns_memory = [None for _ in range(env_batch * samples)]
    for t in range(horizon):
        action = flat_sequences[:, t]
        predicted_z, reward, done_logit, state = rnn.step(current_z, action, state, deterministic=True)
        if ns_variant in ("projection", "residual"):
            current_z, ns_memory, _ = apply_ns_mawm_to_latent_step(
                vae=vae,
                current_z=current_z,
                predicted_z=predicted_z,
                action=action,
                ns_memory=ns_memory,
                ns_variant=ns_variant,
                ns_coverage=ns_coverage,
                num_agents=num_agents,
                device=z.device,
            )
        else:
            current_z = predicted_z
        done_prob = torch.sigmoid(done_logit).clamp(0.0, 1.0)
        returns += discount * reward
        discount *= float(args.gamma if hasattr(args, "gamma") else 0.99) * (1.0 - done_prob)
    return returns.reshape(env_batch, samples, num_agents).mean(dim=-1)


def infer_ns_settings(baseline_id: str) -> tuple[str, float]:
    text = str(baseline_id)
    variant = "neural"
    for candidate in NS_VARIANTS:
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


if __name__ == "__main__":
    main()
