from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vGridcraft"))
sys.path.insert(0, str(ROOT / "gridcraft"))

from experiment_logging import add_wandb_args, logger_from_args, should_log_wandb_videos
from wandb_schema import GENERAL, MARL_EVALUATION, MARL_TRAINING, WORLD_MODEL_EVALUATION, WORLD_MODEL_TRAINING
from torch_world_model import TorchGridcraftRNN, TorchGridcraftVAE
from vgridcraft import VGridcraftConfig, VectorizedGridcraftEnv
from vgridcraft.dataset import RolloutDataset, SequenceDataset, collect_or_load_dataset, observation_shape, observation_vectors


BASELINES = {
    "B00_model-free-control": {"id": "B00", "variant": "none", "coverage": 0.0, "model_based": False},
    "B25_residual_k0.3": {"id": "B25", "variant": "residual", "coverage": 0.3, "model_based": True},
    "B25_projection_k0.3": {"id": "B25", "variant": "projection", "coverage": 0.3, "model_based": True},
    "B25_regularization_k0.3": {"id": "B25", "variant": "regularization", "coverage": 0.3, "model_based": True},
    "B26_residual_k0.6": {"id": "B26", "variant": "residual", "coverage": 0.6, "model_based": True},
    "B26_projection_k0.6": {"id": "B26", "variant": "projection", "coverage": 0.6, "model_based": True},
    "B26_regularization_k0.6": {"id": "B26", "variant": "regularization", "coverage": 0.6, "model_based": True},
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-id", required=True, choices=sorted(BASELINES))
    parser.add_argument("--phase", choices=("world_model", "policy", "all"), default="all")
    parser.add_argument("--run-dir", default="runs_benchmarl")
    parser.add_argument("--dataset-dir", default="datasets/gridcraft")
    parser.add_argument("--reuse-dataset", action="store_true", default=True)
    parser.add_argument("--force-recollect", action="store_true")
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--num-agents", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--episodes", type=int, default=1024)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--wm-batch-size", type=int, default=512)
    parser.add_argument("--wm-num-workers", type=int, default=2)
    parser.add_argument("--vae-steps", type=int, default=5000)
    parser.add_argument("--rnn-steps", type=int, default=5000)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--video-every", type=int, default=1000)
    parser.add_argument("--horizons", nargs="+", type=int, default=[1, 5, 10])
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--dry-run", action="store_true")
    add_wandb_args(parser)
    args = parser.parse_args()

    baseline = BASELINES[args.baseline_id]
    config = VGridcraftConfig(num_agents=args.num_agents, max_steps=args.max_steps, seed=args.seed)
    run_name = f"{args.baseline_id}_a{args.num_agents}_seed{args.seed}"
    run_dir = Path(args.run_dir) / run_name
    checkpoint_dir = run_dir / "checkpoints"
    eval_dir = run_dir / "eval"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    logger = logger_from_args(
        args,
        config={
            **vars(args),
            "baseline": baseline,
            "gridcraft_config": config.__dict__,
            "torch_cuda_available": torch.cuda.is_available(),
        },
        default_group=args.baseline_id,
        default_name=run_name,
        tags=["gridcraft", "benchmarl", baseline["id"], baseline["variant"], f"k{baseline['coverage']}"],
        info_sections=[GENERAL, WORLD_MODEL_TRAINING, WORLD_MODEL_EVALUATION, MARL_TRAINING, MARL_EVALUATION],
        out_dir=str(run_dir),
    )
    logger.save_json(str(run_dir / "baseline_config.json"), {"baseline": baseline, "args": vars(args), "gridcraft_config": config.__dict__})

    if args.dry_run:
        print(json.dumps({"run_dir": str(run_dir), "baseline": baseline, "device": args.device}, indent=2))
        logger.finish()
        return

    data = None
    vae = None
    rnn = None
    device = torch.device(args.device)

    if baseline["model_based"] and args.phase in ("world_model", "all"):
        data, dataset_file, reused = collect_or_load_dataset(
            dataset_dir=args.dataset_dir,
            episodes=args.episodes,
            max_steps=args.max_steps,
            num_envs=args.num_envs,
            device=args.device,
            seed=args.seed,
            config=config,
            reuse=args.reuse_dataset,
            force_recollect=args.force_recollect,
        )
        logger.log(
            {
                "dataset_reused": float(reused),
                "dataset_collected": float(not reused),
                "dataset_episodes": observation_shape(data)[0],
                "dataset_steps": observation_shape(data)[1],
                "dataset_agents": observation_shape(data)[2],
                "dataset_agent_steps": int(observation_shape(data)[0] * observation_shape(data)[1] * observation_shape(data)[2]),
                "dataset_path": str(dataset_file),
            },
            step=1,
            namespace="wm_training",
        )
        vae = train_vae(data, args, device, checkpoint_dir, logger)
        z = encode_dataset(vae, data, device, args.wm_batch_size)
        rnn = train_rnn(z, data, args, device, checkpoint_dir, eval_dir, logger, vae, step_offset=args.vae_steps)
        metrics = evaluate_world_model(vae, rnn, data, device, horizons=args.horizons)
        (eval_dir / "world_model_summary.json").write_text(json.dumps(metrics, indent=2))
        final_eval_step = args.vae_steps + args.rnn_steps + 1
        logger.log(metrics, step=final_eval_step, namespace="wm_evaluation")
        log_world_model_video(logger, args, config, vae, rnn, device, step=final_eval_step)
        logger.log_summary(metrics, namespace="wm_evaluation")

    if args.phase in ("policy", "all"):
        policy_metrics = run_vectorized_policy_smoke(args, config, device, model_based=baseline["model_based"])
        policy_step = args.vae_steps + args.rnn_steps + 2 if baseline["model_based"] else 1
        logger.log(policy_metrics, step=policy_step, namespace="marl_evaluation")
        log_policy_video(logger, args, config, device, step=policy_step)
        logger.log_summary(policy_metrics, namespace="marl_evaluation")

    logger.finish()


def train_vae(data, args, device, checkpoint_dir: Path, logger):
    model = TorchGridcraftVAE().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    effective_batch_size = min(args.wm_batch_size, len(RolloutDataset(data)))
    loader = DataLoader(
        RolloutDataset(data),
        batch_size=effective_batch_size,
        shuffle=True,
        num_workers=args.wm_num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    iterator = iter(loader)
    start = time.time()
    for step in range(1, args.vae_steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        batch = batch.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        loss, metrics = model.loss(batch)
        loss.backward()
        optimizer.step()
        if step == 1 or step % 100 == 0:
            metrics["vae_samples_per_second"] = step * effective_batch_size / max(time.time() - start, 1e-6)
            logger.log(metrics, step=step, namespace="wm_training")
        if args.eval_every and step % args.eval_every == 0:
            torch.save(model.state_dict(), checkpoint_dir / f"vae_step_{step}.pt")
    torch.save(model.state_dict(), checkpoint_dir / "vae.pt")
    return model


@torch.no_grad()
def encode_dataset(vae, data, device, batch_size):
    episodes, steps, agents = observation_shape(data)
    dataset = RolloutDataset(data)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    chunks = []
    for batch in loader:
        chunks.append(vae.encode(batch.to(device, non_blocking=True), sample=False).cpu())
    z = torch.cat(chunks, dim=0)
    return z.reshape(episodes, steps, agents, -1)


def train_rnn(z, data, args, device, checkpoint_dir: Path, eval_dir: Path, logger, vae, step_offset: int = 0):
    model = TorchGridcraftRNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    z_seq, actions, rewards, dones = flatten_agent_sequences(z, data["action"], data["reward"], data["done"])
    dataset = SequenceDataset(z_seq, actions, rewards, dones, seq_len=min(args.seq_len, z_seq.shape[1] - 1))
    effective_batch_size = min(args.wm_batch_size, len(dataset))
    loader = DataLoader(
        dataset,
        batch_size=effective_batch_size,
        shuffle=True,
        num_workers=args.wm_num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    iterator = iter(loader)
    start = time.time()
    for step in range(1, args.rnn_steps + 1):
        try:
            z_seq, action, reward, done = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            z_seq, action, reward, done = next(iterator)
        z_seq = z_seq.to(device, non_blocking=True)
        action = action.to(device, non_blocking=True)
        reward = reward.to(device, non_blocking=True)
        done = done.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        loss, metrics = model.loss(z_seq, action, reward, done)
        loss.backward()
        optimizer.step()
        global_step = step_offset + step
        if step == 1 or step % 100 == 0:
            metrics["rnn_sequences_per_second"] = step * effective_batch_size / max(time.time() - start, 1e-6)
            logger.log(metrics, step=global_step, namespace="wm_training")
        if args.eval_every and step % args.eval_every == 0:
            checkpoint = checkpoint_dir / f"rnn_step_{step}.pt"
            torch.save(model.state_dict(), checkpoint)
            metrics = evaluate_world_model(vae, model, data, device, horizons=args.horizons)
            metrics["checkpoint_step"] = step
            (eval_dir / f"world_model_step_{step}.json").write_text(json.dumps(metrics, indent=2))
            logger.log(metrics, step=global_step, namespace="wm_evaluation")
            if args.video_every and step % args.video_every == 0:
                log_world_model_video(logger, args, args_to_config(args), vae, model, device, step=global_step)
    torch.save(model.state_dict(), checkpoint_dir / "rnn.pt")
    return model


@torch.no_grad()
def evaluate_world_model(vae, rnn, data, device, horizons):
    episodes, _, _ = observation_shape(data)
    obs = observation_vectors(data, episode_limit=min(128, episodes))
    action_data = data["action"][: obs.shape[0]]
    obs = flatten_obs_agents(obs).to(device)
    actions = flatten_action_agents(action_data).to(device)
    z = vae.encode(obs.reshape(-1, obs.shape[-1]), sample=False).reshape(obs.shape[0], obs.shape[1], -1)
    result = {}
    for horizon in horizons:
        h = min(int(horizon), z.shape[1] - 1)
        current_z = z[:, 0]
        state = None
        pred_z = current_z
        for t in range(h):
            pred_z, _, _, state = rnn.step(pred_z, actions[:, t], state, deterministic=True)
        decoded = vae.decode(pred_z)
        target = obs[:, h]
        result[f"compounding_grid_mismatch_h{h}"] = float(grid_mismatch(decoded, target).cpu())
        result[f"compounding_self_mse_h{h}"] = float(torch.mean((decoded[:, -11:] - target[:, -11:]) ** 2).cpu())
    decoded_one = vae.decode(z[:, 1])
    result["grid_mismatch"] = float(grid_mismatch(decoded_one, obs[:, 1]).cpu())
    result["self_mse"] = float(torch.mean((decoded_one[:, -11:] - obs[:, 1, -11:]) ** 2).cpu())
    return result


def all_agent_obs(obs: torch.Tensor) -> torch.Tensor:
    if obs.ndim == 4:
        return obs
    if obs.ndim == 3:
        return obs.unsqueeze(2)
    return obs


def flatten_obs_agents(obs: torch.Tensor) -> torch.Tensor:
    obs = all_agent_obs(obs)
    episodes, steps, agents, obs_size = obs.shape
    return obs.permute(0, 2, 1, 3).reshape(episodes * agents, steps, obs_size)


def flatten_action_agents(actions: torch.Tensor) -> torch.Tensor:
    if actions.ndim == 2:
        return actions
    if actions.ndim == 3:
        episodes, steps, agents = actions.shape
        return actions.permute(0, 2, 1).reshape(episodes * agents, steps)
    if actions.ndim == 4 and actions.shape[-1] == 1:
        return flatten_action_agents(actions.squeeze(-1))
    raise ValueError(f"unsupported action tensor shape: {tuple(actions.shape)}")


def flatten_reward_agents(rewards: torch.Tensor) -> torch.Tensor:
    if rewards.ndim == 2:
        return rewards
    if rewards.ndim == 3:
        episodes, steps, agents = rewards.shape
        return rewards.permute(0, 2, 1).reshape(episodes * agents, steps)
    if rewards.ndim == 4 and rewards.shape[-1] == 1:
        return flatten_reward_agents(rewards.squeeze(-1))
    raise ValueError(f"unsupported reward tensor shape: {tuple(rewards.shape)}")


def flatten_done_agents(dones: torch.Tensor, num_agents: int) -> torch.Tensor:
    if dones.ndim == 2:
        episodes, steps = dones.shape
        return dones[:, None, :].expand(episodes, num_agents, steps).reshape(episodes * num_agents, steps)
    if dones.ndim == 3:
        episodes, steps, agents = dones.shape
        return dones.permute(0, 2, 1).reshape(episodes * agents, steps)
    if dones.ndim == 4 and dones.shape[-1] == 1:
        return flatten_done_agents(dones.squeeze(-1), num_agents)
    raise ValueError(f"unsupported done tensor shape: {tuple(dones.shape)}")


def flatten_agent_sequences(z: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor, dones: torch.Tensor):
    if z.ndim == 3:
        z_seq = z
        num_agents = 1
    elif z.ndim == 4:
        episodes, steps, agents, z_size = z.shape
        num_agents = agents
        z_seq = z.permute(0, 2, 1, 3).reshape(episodes * agents, steps, z_size)
    else:
        raise ValueError(f"unsupported latent tensor shape: {tuple(z.shape)}")
    return (
        z_seq,
        flatten_action_agents(actions),
        flatten_reward_agents(rewards),
        flatten_done_agents(dones, num_agents).float(),
    )


def grid_mismatch(decoded, target):
    cursor = 0
    mismatches = []
    for depth in (3, 4, 4):
        pred = decoded[:, cursor:cursor + 49 * depth].reshape(-1, 49, depth).argmax(-1)
        label = target[:, cursor:cursor + 49 * depth].reshape(-1, 49, depth).argmax(-1)
        mismatches.append((pred != label).float().mean())
        cursor += 49 * depth
    return torch.stack(mismatches).mean()


def run_vectorized_policy_smoke(args, config, device, model_based: bool):
    env = VectorizedGridcraftEnv(num_envs=args.num_envs, num_agents=config.num_agents, device=device, seed=args.seed + 123, config=config)
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed + 456)
    obs = env.reset()
    returns = torch.zeros((args.num_envs, config.num_agents), device=device)
    lengths = torch.zeros((args.num_envs,), device=device)
    for _ in range(args.max_steps):
        actions = torch.randint(0, config.action_size, (args.num_envs, config.num_agents), generator=generator, device=device)
        obs, reward, done, truncated, _ = env.step(actions)
        returns += reward
        lengths += (~(done | truncated)).float()
        if bool((done | truncated).all()):
            break
    prefix = "imagined" if model_based else "real"
    return {
        f"eval_{prefix}_reward": float(returns.mean().cpu()),
        "eval_real_reward": float(returns.mean().cpu()) if not model_based else 0.0,
        "episode_length": float(lengths.mean().cpu()),
        "policy_backend": "vectorized_random_smoke",
    }


def args_to_config(args):
    return VGridcraftConfig(num_agents=args.num_agents, max_steps=args.max_steps, seed=args.seed)


@torch.no_grad()
def log_world_model_video(logger, args, config, vae, rnn, device, step=None):
    if not should_log_wandb_videos(args):
        return
    try:
        frames = record_world_model_video(args, config, vae, rnn, device)
        logged = logger.log_video(
            "video_real_vs_imagined",
            frames,
            fps=args.video_fps,
            step=step,
            namespace="wm_evaluation",
        )
        logger.log(
            {
                "video_real_vs_imagined_logged": int(bool(logged)),
                "video_real_vs_imagined_frame_count": int(len(frames)),
            },
            step=step,
            namespace="wm_evaluation",
        )
    except Exception as exc:
        logger.log(
            {
                "video_real_vs_imagined_logged": 0,
                "video_real_vs_imagined_generation_failed": 1,
                "video_real_vs_imagined_error": str(exc),
            },
            step=step,
            namespace="wm_evaluation",
        )


@torch.no_grad()
def record_world_model_video(args, config, vae, rnn, device):
    env = VectorizedGridcraftEnv(num_envs=1, num_agents=config.num_agents, device=device, seed=args.seed + 9000, config=config)
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed + 9001)
    frames = []
    obs = env.reset()
    z = vae.encode(obs["vector"][0].to(device), sample=False)
    state = None
    max_steps = max(1, int(args.video_max_steps))
    for _ in range(max_steps):
        action = torch.randint(0, config.action_size, (1, config.num_agents), generator=generator, device=device)
        obs, reward, done, truncated, _ = env.step(action)
        pred_z, _, _, state = rnn.step(z, action[0], state, deterministic=True)
        imagined = vae.decode_tabular(pred_z)
        tabular = {}
        for agent_idx in range(config.num_agents):
            tabular[f"agent_{agent_idx}"] = {
                "grid": imagined["grid"][agent_idx].detach().cpu().numpy().astype("int8"),
                "self": imagined["self"][agent_idx].detach().cpu().numpy().astype("int16"),
            }
        frame = env.render(env_index=0, mode="rgb_array", tabular_observations=tabular)
        frames.append(frame[:, :, :3] if frame.shape[-1] == 4 else frame)
        z = vae.encode(obs["vector"][0].to(device), sample=False)
        if bool((done | truncated).all()):
            break
    env.close()
    return frames


@torch.no_grad()
def log_policy_video(logger, args, config, device, step=None):
    if not should_log_wandb_videos(args):
        return
    try:
        frames = record_real_policy_video(args, config, device)
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
                "video_policy_rollout_frame_count": int(len(frames)),
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
def record_real_policy_video(args, config, device):
    env = VectorizedGridcraftEnv(num_envs=1, num_agents=config.num_agents, device=device, seed=args.seed + 9100, config=config)
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed + 9101)
    frames = []
    env.reset()
    max_steps = max(1, int(args.video_max_steps))
    for _ in range(max_steps):
        action = torch.randint(0, config.action_size, (1, config.num_agents), generator=generator, device=device)
        _, _, done, truncated, _ = env.step(action)
        frame = env.render(env_index=0, mode="rgb_array")
        frames.append(frame[:, :, :3] if frame.shape[-1] == 4 else frame)
        if bool((done | truncated).all()):
            break
    env.close()
    return frames


if __name__ == "__main__":
    main()
