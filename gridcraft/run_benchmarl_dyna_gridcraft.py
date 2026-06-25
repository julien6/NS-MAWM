from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vGridcraft"))
sys.path.insert(0, str(ROOT / "gridcraft"))

from experiment_logging import add_wandb_args, logger_from_args, should_log_wandb_videos
from ns_symbolic import NS_VARIANTS, apply_symbolic_projection, tabular_to_vector
from wandb_schema import GENERAL, MARL_EVALUATION, MARL_TRAINING
from torch_world_model import TorchGridcraftRNN, TorchGridcraftVAE
from torch_world_model.models import ACTION_SIZE
from vgridcraft import VGridcraftConfig, VectorizedGridcraftEnv


class SharedActorCritic(nn.Module):
    def __init__(self, obs_size: int = 64, hidden_size: int = 256, action_size: int = ACTION_SIZE):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )
        self.actor = nn.Linear(hidden_size, action_size)
        self.critic = nn.Linear(hidden_size, 1)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.net(obs.float())
        return self.actor(h), self.critic(h).squeeze(-1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-id", required=True)
    parser.add_argument("--wm-run-dir", required=True)
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--num-agents", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--max-iters", type=int, default=1000)
    parser.add_argument("--imagined-horizon", type=int, default=32)
    parser.add_argument("--imagined-start-noise", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--mappo-hidden-size", type=int, default=256)
    parser.add_argument("--mappo-eval-every-iters", type=int, default=25)
    parser.add_argument("--mappo-eval-episodes", type=int, default=4)
    parser.add_argument("--mappo-video-every-iters", type=int, default=250)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--wandb-step-offset", type=int, default=int(os.environ.get("WANDB_STEP_OFFSET", "0")))
    add_wandb_args(parser)
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    config = VGridcraftConfig(num_agents=args.num_agents, max_steps=args.max_steps, seed=args.seed)
    run_dir = Path(args.wm_run_dir)
    checkpoint_dir = run_dir / "checkpoints"
    ns_variant, ns_coverage = infer_ns_settings(args.baseline_id)

    logger = logger_from_args(
        args,
        config={
            **vars(args),
            "downstream_policy_backend": "dyna_mappo_world_model_only",
            "downstream_uses_real_environment": False,
            "downstream_real_interaction_ratio": 0.0,
            "downstream_imagined_ratio": 1.0,
            "ns_variant": ns_variant,
            "ns_coverage": ns_coverage,
            "world_model_checkpoint_dir": str(checkpoint_dir),
            "gridcraft_config": config.__dict__,
        },
        default_group=args.baseline_id,
        default_name=f"{args.baseline_id}_a{args.num_agents}_dyna_seed{args.seed}",
        tags=["gridcraft", "dyna-mappo", "world-model-policy", args.baseline_id],
        info_sections=[GENERAL, MARL_TRAINING, MARL_EVALUATION],
        out_dir=str(run_dir / "dyna_policy"),
    )

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
    for param in vae.parameters():
        param.requires_grad_(False)
    for param in rnn.parameters():
        param.requires_grad_(False)

    policy = SharedActorCritic(hidden_size=args.mappo_hidden_size).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.learning_rate)
    batch = args.num_envs * args.num_agents
    z = torch.randn(batch, vae.z_size, device=device) * float(args.imagined_start_noise)
    rnn_state = None
    ns_memory = [None for _ in range(args.num_envs)]

    print(f"=== Dyna MAPPO downstream training in world model ({args.baseline_id}) ===", flush=True)
    print(
        f"Dyna config: envs={args.num_envs}, agents={args.num_agents}, "
        f"iters={args.max_iters}, imagined_horizon={args.imagined_horizon}, "
        f"ns_variant={ns_variant}, ns_coverage={ns_coverage}, device={device}",
        flush=True,
    )
    for iteration in range(1, args.max_iters + 1):
        z, rnn_state, ns_memory, metrics = dyna_update(
            vae=vae,
            rnn=rnn,
            policy=policy,
            optimizer=optimizer,
            z=z,
            rnn_state=rnn_state,
            ns_memory=ns_memory,
            ns_variant=ns_variant,
            ns_coverage=ns_coverage,
            args=args,
            device=device,
        )
        step = int(args.wandb_step_offset) + iteration
        logger.log(metrics, step=step, namespace="marl_training")
        if iteration == 1 or iteration % max(1, int(args.mappo_eval_every_iters)) == 0:
            eval_metrics = evaluate_imagined_policy(vae, rnn, policy, args, device, ns_variant, ns_coverage)
            eval_metrics["policy_backend"] = "dyna_mappo_world_model_only"
            logger.log(eval_metrics, step=step, namespace="marl_evaluation")
            print(
                "[dyna] "
                f"iter={iteration}/{args.max_iters} "
                f"train_imagined_reward={metrics['training_imagined_reward']:.3f} "
                f"eval_imagined_reward={eval_metrics['eval_imagined_reward']:.3f} "
                f"policy_loss={metrics['policy_loss']:.4f}",
                flush=True,
            )
        if should_log_wandb_videos(args) and int(args.mappo_video_every_iters) > 0 and iteration % int(args.mappo_video_every_iters) == 0:
            log_imagined_policy_video(logger, vae, rnn, policy, args, config, device, ns_variant, ns_coverage, step=step)

    final_step = int(args.wandb_step_offset) + int(args.max_iters) + 1
    final_metrics = evaluate_imagined_policy(vae, rnn, policy, args, device, ns_variant, ns_coverage)
    final_metrics["policy_backend"] = "dyna_mappo_world_model_only"
    logger.log(final_metrics, step=final_step, namespace="marl_evaluation")
    log_imagined_policy_video(logger, vae, rnn, policy, args, config, device, ns_variant, ns_coverage, step=final_step)
    torch.save(policy.state_dict(), run_dir / "dyna_policy" / "policy.pt")
    logger.log_summary(final_metrics, namespace="marl_evaluation")
    logger.finish()


def dyna_update(vae, rnn, policy, optimizer, z, rnn_state, ns_memory, ns_variant, ns_coverage, args, device):
    log_probs = []
    values = []
    rewards = []
    entropies = []
    current_z = z.detach()
    state = detach_rnn_state(rnn_state)
    for _ in range(max(1, int(args.imagined_horizon))):
        logits, value = policy(current_z)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        with torch.no_grad():
            next_z, reward, done_logit, state = rnn.step(current_z, action, state, deterministic=True)
            next_z, ns_memory, ns_metrics = apply_ns_mawm_to_latent_step(
                vae=vae,
                current_z=current_z,
                predicted_z=next_z,
                action=action,
                ns_memory=ns_memory,
                ns_variant=ns_variant,
                ns_coverage=ns_coverage,
                num_agents=args.num_agents,
                device=device,
            )
            done_prob = torch.sigmoid(done_logit).clamp(0.0, 1.0)
            reward = reward * (1.0 - done_prob)
        log_probs.append(dist.log_prob(action))
        values.append(value)
        rewards.append(reward.detach())
        entropies.append(dist.entropy())
        current_z = next_z.detach()

    returns = discounted_returns(torch.stack(rewards), gamma=args.gamma)
    values_t = torch.stack(values)
    log_probs_t = torch.stack(log_probs)
    entropy_t = torch.stack(entropies)
    advantage = returns - values_t.detach()
    policy_loss = -(log_probs_t * advantage).mean()
    value_loss = F.mse_loss(values_t, returns)
    entropy = entropy_t.mean()
    total_loss = policy_loss + float(args.value_coef) * value_loss - float(args.entropy_coef) * entropy
    optimizer.zero_grad(set_to_none=True)
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
    optimizer.step()
    return current_z.detach(), detach_rnn_state(state), ns_memory, {
        "training_imagined_reward": float(torch.stack(rewards).sum(dim=0).mean().detach().cpu()),
        "policy_loss": float(policy_loss.detach().cpu()),
        "value_loss": float(value_loss.detach().cpu()),
        "entropy": float(entropy.detach().cpu()),
        "policy_total_loss": float(total_loss.detach().cpu()),
        "imagined_ratio": 1.0,
        "real_interaction_ratio": 0.0,
        "ns_correction_count": float(ns_metrics.get("correction_count", 0.0)),
    }


def discounted_returns(rewards: torch.Tensor, gamma: float) -> torch.Tensor:
    returns = torch.zeros_like(rewards)
    running = torch.zeros_like(rewards[-1])
    for index in range(rewards.shape[0] - 1, -1, -1):
        running = rewards[index] + float(gamma) * running
        returns[index] = running
    return returns


def detach_rnn_state(state):
    if state is None:
        return None
    return tuple(item.detach() for item in state)


@torch.no_grad()
def evaluate_imagined_policy(vae, rnn, policy, args, device, ns_variant, ns_coverage):
    episodes = max(1, int(args.mappo_eval_episodes))
    batch = episodes * int(args.num_agents)
    z = torch.randn(batch, vae.z_size, device=device) * float(args.imagined_start_noise)
    state = None
    ns_memory = [None for _ in range(episodes)]
    returns = torch.zeros(batch, device=device)
    lengths = torch.zeros(batch, device=device)
    for _ in range(max(1, int(args.video_max_steps))):
        logits, _ = policy(z)
        action = logits.argmax(dim=-1)
        predicted_z, reward, done_logit, state = rnn.step(z, action, state, deterministic=True)
        z, ns_memory, _ = apply_ns_mawm_to_latent_step(
            vae=vae,
            current_z=z,
            predicted_z=predicted_z,
            action=action,
            ns_memory=ns_memory,
            ns_variant=ns_variant,
            ns_coverage=ns_coverage,
            num_agents=args.num_agents,
            device=device,
        )
        done_prob = torch.sigmoid(done_logit)
        alive = done_prob < 0.5
        returns += reward * alive.float()
        lengths += alive.float()
    return {
        "eval_imagined_reward": float(returns.mean().cpu()),
        "eval_imagined_reward_std": float(returns.std(unbiased=False).cpu()),
        "eval_imagined_episode_length": float(lengths.mean().cpu()),
        "imagined_ratio": 1.0,
        "real_interaction_ratio": 0.0,
    }


@torch.no_grad()
def log_imagined_policy_video(logger, vae, rnn, policy, args, config, device, ns_variant, ns_coverage, step=None):
    if not should_log_wandb_videos(args):
        return
    try:
        frames = record_imagined_policy_video(vae, rnn, policy, args, config, device, ns_variant, ns_coverage)
        logged = logger.log_video("video_policy_rollout", frames, fps=args.video_fps, step=step, namespace="marl_evaluation")
        logger.log(
            {
                "video_policy_rollout_logged": int(bool(logged)),
                "video_policy_rollout_frame_count": int(len(frames)),
                "video_policy_rollout_source": "world_model",
            },
            step=step,
            namespace="marl_evaluation",
        )
    except Exception as exc:
        logger.log(
            {
                "video_policy_rollout_logged": 0,
                "video_policy_rollout_generation_failed": 1,
                "video_policy_rollout_error": str(exc),
            },
            step=step,
            namespace="marl_evaluation",
        )


@torch.no_grad()
def record_imagined_policy_video(vae, rnn, policy, args, config, device, ns_variant, ns_coverage):
    env = VectorizedGridcraftEnv(num_envs=1, num_agents=config.num_agents, device=device, seed=args.seed + 9900, config=config)
    z = torch.randn(config.num_agents, vae.z_size, device=device) * float(args.imagined_start_noise)
    state = None
    ns_memory = [None]
    frames = []
    env.reset()
    for _ in range(max(1, int(args.video_max_steps))):
        logits, _ = policy(z)
        action = logits.argmax(dim=-1)
        predicted_z, _, _, state = rnn.step(z, action, state, deterministic=True)
        z, ns_memory, _ = apply_ns_mawm_to_latent_step(
            vae=vae,
            current_z=z,
            predicted_z=predicted_z,
            action=action,
            ns_memory=ns_memory,
            ns_variant=ns_variant,
            ns_coverage=ns_coverage,
            num_agents=config.num_agents,
            device=device,
        )
        imagined = vae.decode_tabular(z)
        tabular = {}
        for agent_idx in range(config.num_agents):
            tabular[f"agent_{agent_idx}"] = {
                "grid": imagined["grid"][agent_idx].detach().cpu().numpy().astype("int8"),
                "self": imagined["self"][agent_idx].detach().cpu().numpy().astype("int16"),
            }
        frame = env.render(env_index=0, mode="rgb_array", tabular_observations=tabular)
        frames.append(frame[:, :, :3] if frame.shape[-1] == 4 else frame)
    env.close()
    return np.asarray(frames, dtype=np.uint8)


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


@torch.no_grad()
def apply_ns_mawm_to_latent_step(vae, current_z, predicted_z, action, ns_memory, ns_variant, ns_coverage, num_agents, device):
    if ns_variant in ("neural", "regularization"):
        return predicted_z, ns_memory, {"correction_count": 0.0}
    if predicted_z.shape[0] % int(num_agents) != 0:
        raise ValueError(f"latent batch {predicted_z.shape[0]} is not divisible by num_agents={num_agents}")
    num_envs = predicted_z.shape[0] // int(num_agents)
    if ns_memory is None or len(ns_memory) != num_envs:
        ns_memory = [None for _ in range(num_envs)]
    current_tabular = decode_flat_tabular(vae, current_z, num_envs, num_agents)
    predicted_tabular = decode_flat_tabular(vae, predicted_z, num_envs, num_agents)
    action_np = action.detach().cpu().long().reshape(num_envs, num_agents).numpy()
    projected_vectors = []
    correction_count = 0
    for env_idx in range(num_envs):
        current_joint = {
            f"agent_{agent_idx}": current_tabular[env_idx][agent_idx]
            for agent_idx in range(num_agents)
        }
        predicted_joint = {
            f"agent_{agent_idx}": predicted_tabular[env_idx][agent_idx]
            for agent_idx in range(num_agents)
        }
        joint_action = {
            f"agent_{agent_idx}": int(action_np[env_idx, agent_idx])
            for agent_idx in range(num_agents)
        }
        projected_joint, symbolic_info = apply_symbolic_projection(
            predicted_joint,
            current_joint,
            joint_action,
            ns_variant,
            coverage=ns_coverage,
            memory=ns_memory[env_idx],
        )
        if symbolic_info is not None and len(symbolic_info) >= 3:
            ns_memory[env_idx] = symbolic_info[2]
        for agent_idx in range(num_agents):
            agent_id = f"agent_{agent_idx}"
            projected_vectors.append(tabular_to_vector(projected_joint[agent_id]))
            if not tabular_equal(projected_joint[agent_id], predicted_joint[agent_id]):
                correction_count += 1
    projected = torch.as_tensor(np.asarray(projected_vectors, dtype=np.float32), device=device)
    corrected_z = vae.encode(projected, sample=False)
    return corrected_z, ns_memory, {"correction_count": float(correction_count)}


@torch.no_grad()
def decode_flat_tabular(vae, z, num_envs, num_agents):
    decoded = vae.decode_tabular(z)
    grids = decoded["grid"].detach().cpu().numpy().astype("int8")
    selves = decoded["self"].detach().cpu().numpy().astype("int16")
    result = []
    cursor = 0
    for _ in range(num_envs):
        row = []
        for _ in range(num_agents):
            row.append({"grid": grids[cursor], "self": selves[cursor]})
            cursor += 1
        result.append(row)
    return result


def tabular_equal(left, right):
    return bool(np.array_equal(left["grid"], right["grid"]) and np.array_equal(left["self"], right["self"]))


if __name__ == "__main__":
    main()
