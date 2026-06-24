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

from experiment_logging import add_wandb_args, logger_from_args
from wandb_schema import GENERAL, MARL_EVALUATION, MARL_TRAINING, WORLD_MODEL_EVALUATION, WORLD_MODEL_TRAINING
from torch_world_model import TorchGridcraftRNN, TorchGridcraftVAE
from vgridcraft import VGridcraftConfig, VectorizedGridcraftEnv
from vgridcraft.dataset import RolloutDataset, SequenceDataset, collect_or_load_dataset


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
    run_name = f"{args.baseline_id}_seed{args.seed}"
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
                "dataset_episodes": int(data["obs"].shape[0]),
                "dataset_steps": int(data["obs"].shape[1]),
                "dataset_path": str(dataset_file),
            },
            step=0,
            namespace="wm_training",
        )
        vae = train_vae(data, args, device, checkpoint_dir, logger)
        z = encode_dataset(vae, data, device, args.wm_batch_size)
        rnn = train_rnn(z, data, args, device, checkpoint_dir, eval_dir, logger, vae)
        metrics = evaluate_world_model(vae, rnn, data, device, horizons=args.horizons)
        (eval_dir / "world_model_summary.json").write_text(json.dumps(metrics, indent=2))
        logger.log(metrics, step=args.rnn_steps, namespace="wm_evaluation")
        logger.log_summary(metrics, namespace="wm_evaluation")

    if args.phase in ("policy", "all"):
        policy_metrics = run_vectorized_policy_smoke(args, config, device, model_based=baseline["model_based"])
        logger.log(policy_metrics, step=0, namespace="marl_evaluation")
        logger.log_summary(policy_metrics, namespace="marl_evaluation")

    logger.finish()


def train_vae(data, args, device, checkpoint_dir: Path, logger):
    model = TorchGridcraftVAE().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loader = DataLoader(
        RolloutDataset(data),
        batch_size=args.wm_batch_size,
        shuffle=True,
        num_workers=args.wm_num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
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
            metrics["vae_samples_per_second"] = step * args.wm_batch_size / max(time.time() - start, 1e-6)
            logger.log(metrics, step=step, namespace="wm_training")
        if args.eval_every and step % args.eval_every == 0:
            torch.save(model.state_dict(), checkpoint_dir / f"vae_step_{step}.pt")
    torch.save(model.state_dict(), checkpoint_dir / "vae.pt")
    return model


@torch.no_grad()
def encode_dataset(vae, data, device, batch_size):
    flat = data["obs"].reshape(-1, data["obs"].shape[-1]).float()
    chunks = []
    for start in range(0, flat.shape[0], batch_size):
        chunks.append(vae.encode(flat[start:start + batch_size].to(device), sample=False).cpu())
    z = torch.cat(chunks, dim=0)
    return z.reshape(data["obs"].shape[0], data["obs"].shape[1], -1)


def train_rnn(z, data, args, device, checkpoint_dir: Path, eval_dir: Path, logger, vae):
    model = TorchGridcraftRNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    actions = data["action"].squeeze(-1)
    rewards = data["reward"].squeeze(-1)
    dones = data["done"].float()
    dataset = SequenceDataset(z, actions, rewards, dones, seq_len=min(args.seq_len, z.shape[1] - 1))
    loader = DataLoader(
        dataset,
        batch_size=args.wm_batch_size,
        shuffle=True,
        num_workers=args.wm_num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
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
        if step == 1 or step % 100 == 0:
            metrics["rnn_sequences_per_second"] = step * args.wm_batch_size / max(time.time() - start, 1e-6)
            logger.log(metrics, step=step, namespace="wm_training")
        if args.eval_every and step % args.eval_every == 0:
            checkpoint = checkpoint_dir / f"rnn_step_{step}.pt"
            torch.save(model.state_dict(), checkpoint)
            metrics = evaluate_world_model(vae, model, data, device, horizons=args.horizons)
            metrics["checkpoint_step"] = step
            (eval_dir / f"world_model_step_{step}.json").write_text(json.dumps(metrics, indent=2))
            logger.log(metrics, step=step, namespace="wm_evaluation")
    torch.save(model.state_dict(), checkpoint_dir / "rnn.pt")
    return model


@torch.no_grad()
def evaluate_world_model(vae, rnn, data, device, horizons):
    obs = data["obs"][: min(128, data["obs"].shape[0])].to(device)
    actions = data["action"][: obs.shape[0]].squeeze(-1).to(device)
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


if __name__ == "__main__":
    main()
