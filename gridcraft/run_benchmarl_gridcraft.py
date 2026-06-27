from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vGridcraft"))
sys.path.insert(0, str(ROOT / "gridcraft"))

from experiment_logging import add_wandb_args, logger_from_args, should_log_wandb_videos
from ns_symbolic import (
    apply_symbolic_projection,
    compare_joint_with_symbolic,
    symbolic_batch_targets,
    symbolic_joint_transition,
    tabular_mask_to_vector_mask,
    tabular_to_vector,
    vector_to_tabular,
)
from wandb_schema import GENERAL, MARL_EVALUATION, MARL_TRAINING, PSTR_RULES, WORLD_MODEL_EVALUATION, WORLD_MODEL_TRAINING
from torch_world_model import TorchGridcraftRNN, TorchGridcraftVAE
from vgridcraft import VGridcraftConfig, VectorizedGridcraftEnv
from vgridcraft.dataset import RolloutDataset, SequenceDataset, collect_or_load_dataset, observation_shape, observation_vectors, has_compact_observations, vector_from_tabular


BASELINES = {
    "B00_model-free-control": {"id": "B00", "variant": "none", "coverage": 0.0, "model_based": False},
    "B10_neural_k0.0": {"id": "B10", "variant": "neural", "coverage": 0.0, "model_based": True},
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
    parser.add_argument("--lambda-sym", type=float, default=1.0)
    parser.add_argument("--lambda-residual", type=float, default=0.25)
    parser.add_argument("--symbolic-train-samples", type=int, default=512)
    parser.add_argument("--joint-symbolic-train-episodes", type=int, default=8)
    parser.add_argument("--joint-symbolic-train-steps", type=int, default=8)
    parser.add_argument("--rvr-eval-steps", type=int, default=50)
    parser.add_argument("--shared-model-dir", default="shared_models")
    vae_cache_group = parser.add_mutually_exclusive_group()
    vae_cache_group.add_argument("--reuse-vae-cache", dest="reuse_vae_cache", action="store_true", default=True)
    vae_cache_group.add_argument("--no-reuse-vae-cache", dest="reuse_vae_cache", action="store_false")
    parser.add_argument("--force-vae-retrain", action="store_true")
    latent_cache_group = parser.add_mutually_exclusive_group()
    latent_cache_group.add_argument("--reuse-latent-cache", dest="reuse_latent_cache", action="store_true", default=True)
    latent_cache_group.add_argument("--no-reuse-latent-cache", dest="reuse_latent_cache", action="store_false")
    parser.add_argument("--force-latent-reencode", action="store_true")
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
        print(f"=== World Model Training/Evaluation ({args.baseline_id}) ===", flush=True)
        print(
            "World Model config: "
            f"episodes={args.episodes}, max_steps={args.max_steps}, agents={args.num_agents}, "
            f"vae_steps={args.vae_steps}, rnn_steps={args.rnn_steps}, "
            f"batch={args.wm_batch_size}, eval_every={args.eval_every}, video_every={args.video_every}",
            flush=True,
        )
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
        maybe_attach_symbolic_dataset_targets(data, baseline, args)
        logger.log(
            {
                "dataset_reused": float(reused),
                "dataset_collected": float(not reused),
                "dataset_episodes": observation_shape(data)[0],
                "dataset_steps": observation_shape(data)[1],
                "dataset_agents": observation_shape(data)[2],
                "dataset_agent_steps": int(observation_shape(data)[0] * observation_shape(data)[1] * observation_shape(data)[2]),
                "dataset_valid_transition_count": int(data.get("metadata", {}).get("valid_transition_count", 0)),
                "dataset_invalid_padded_transition_count": int(data.get("metadata", {}).get("invalid_padded_transition_count", 0)),
                "dataset_mean_episode_length": float(data.get("metadata", {}).get("mean_episode_length", 0.0)),
                "dataset_truncation_or_done_rate": float(data.get("metadata", {}).get("truncation_or_done_rate", 0.0)),
                "symbolic_target_cached": int("symbolic_target" in data),
                "symbolic_memory_mean_map_coverage": float(data.get("symbolic_metadata", {}).get("mean_map_coverage_ratio", 0.0)),
                "dataset_path": str(dataset_file),
            },
            step=1,
            namespace="wm_training",
        )
        print("=== World Model phase 1/5: VAE training/cache ===", flush=True)
        vae, vae_cache_dir, vae_cache_key = load_or_train_vae(data, args, dataset_file, device, checkpoint_dir, logger)
        print("=== World Model phase 2/5: latent encoding/cache ===", flush=True)
        z = load_or_encode_latents(vae, data, args, vae_cache_dir, vae_cache_key, device, logger)
        print("=== World Model phase 3/5: MDN-RNN training and periodic evaluation ===", flush=True)
        rnn = train_rnn(z, data, args, device, checkpoint_dir, eval_dir, logger, vae, baseline, step_offset=args.vae_steps)
        print("=== World Model phase 4/5: final evaluation ===", flush=True)
        metrics, pstr_rows = evaluate_world_model(vae, rnn, data, device, horizons=args.horizons, baseline=baseline, args=args)
        (eval_dir / "world_model_summary.json").write_text(json.dumps(metrics, indent=2))
        write_pstr_rvr_table(eval_dir / "pstr_rvr_table_final.json", pstr_rows)
        final_eval_step = args.vae_steps + args.rnn_steps + 1
        logger.log(metrics, step=final_eval_step, namespace="wm_evaluation")
        log_pstr_rvr_table(logger, pstr_rows, step=final_eval_step)
        print("=== World Model phase 5/5: final video ===", flush=True)
        log_world_model_video(logger, args, config, vae, rnn, device, step=final_eval_step)
        logger.log_summary(metrics, namespace="wm_evaluation")

    if args.phase in ("policy", "all") and not baseline["model_based"]:
        print("=== MARL Evaluation: lightweight policy smoke ===", flush=True)
        policy_metrics = run_vectorized_policy_smoke(args, config, device, model_based=baseline["model_based"])
        policy_step = 1
        logger.log(policy_metrics, step=policy_step, namespace="marl_evaluation")
        log_policy_video(logger, args, config, device, step=policy_step)
        logger.log_summary(policy_metrics, namespace="marl_evaluation")
    elif args.phase in ("policy", "all") and baseline["model_based"]:
        print(
            "=== Skipping real-environment policy smoke for model-based baseline; "
            "downstream MARL must run through Dyna world-model rollouts. ===",
            flush=True,
        )

    logger.finish()


def shared_model_root(args) -> Path:
    root = Path(args.shared_model_dir)
    if not root.is_absolute():
        root = ROOT / "gridcraft" / root
    return root


def vae_cache_payload(args, dataset_file: Path, data) -> dict:
    probe = TorchGridcraftVAE()
    return {
        "cache_version": "vae_cache_v1",
        "dataset_file": str(Path(dataset_file).resolve()),
        "dataset_metadata": data.get("metadata", {}),
        "observation_shape": list(observation_shape(data)),
        "vae_arch": {
            "class": "TorchGridcraftVAE",
            "obs_size": int(probe.obs_size),
            "z_size": int(probe.z_size),
            "hidden_size": int(probe.hidden_size),
            "kl_tolerance": float(probe.kl_tolerance),
        },
        "vae_training": {
            "vae_steps": int(args.vae_steps),
            "wm_batch_size": int(args.wm_batch_size),
            "learning_rate": float(args.learning_rate),
            "seed": int(args.seed),
        },
    }


def vae_cache_key(payload: dict) -> str:
    raw = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:20]


def load_json_if_exists(path: Path):
    if not path.exists():
        return None
    with path.open() as f:
        return json.load(f)


def cache_metadata_matches(path: Path, payload: dict, key: str) -> bool:
    metadata = load_json_if_exists(path)
    return bool(metadata and metadata.get("cache_key") == key and metadata.get("payload") == payload)


def write_cache_metadata(path: Path, payload: dict, key: str, cache_dir: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(
            {
                "cache_key": key,
                "payload": payload,
                "vae_path": str(cache_dir / "vae.pt"),
                "latents_path": str(cache_dir / "latents.pt"),
                "created_at": int(time.time()),
            },
            f,
            indent=2,
            sort_keys=True,
        )


def load_or_train_vae(data, args, dataset_file: Path, device, checkpoint_dir: Path, logger):
    payload = vae_cache_payload(args, dataset_file, data)
    key = vae_cache_key(payload)
    cache_dir = shared_model_root(args) / "vae" / key
    vae_path = cache_dir / "vae.pt"
    metadata_path = cache_dir / "metadata.json"
    print(f"[vae-cache] requested key={key} path={cache_dir}", flush=True)

    cache_hit = bool(args.reuse_vae_cache and not args.force_vae_retrain and vae_path.exists() and cache_metadata_matches(metadata_path, payload, key))
    if cache_hit:
        print(f"[vae-cache] hit; loading VAE from {vae_path}", flush=True)
        model = TorchGridcraftVAE().to(device)
        model.load_state_dict(torch.load(vae_path, map_location=device))
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(vae_path, checkpoint_dir / "vae.pt")
        logger.log(
            {
                "vae_reused": 1,
                "vae_trained": 0,
                "vae_cache_key": key,
                "vae_cache_path": str(cache_dir),
            },
            step=1,
            namespace="wm_training",
        )
        return model, cache_dir, key

    reason = "force retrain" if args.force_vae_retrain else "disabled reuse" if not args.reuse_vae_cache else "cache miss"
    print(f"[vae-cache] miss; training VAE ({reason})", flush=True)
    model = train_vae(data, args, device, checkpoint_dir, logger)
    cache_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), vae_path)
    write_cache_metadata(metadata_path, payload, key, cache_dir)
    print(f"[vae-cache] saved VAE to {vae_path}", flush=True)
    logger.log(
        {
            "vae_reused": 0,
            "vae_trained": 1,
            "vae_cache_key": key,
            "vae_cache_path": str(cache_dir),
        },
        step=max(1, int(args.vae_steps)),
        namespace="wm_training",
    )
    return model, cache_dir, key


@torch.no_grad()
def load_or_encode_latents(vae, data, args, cache_dir: Path, cache_key: str, device, logger):
    latent_path = cache_dir / "latents.pt"
    can_reuse = bool(args.reuse_latent_cache and not args.force_latent_reencode and not args.force_vae_retrain and latent_path.exists())
    if can_reuse:
        print(f"[latent-cache] hit; loading latents from {latent_path}", flush=True)
        z = torch.load(latent_path, map_location="cpu")
        logger.log(
            {
                "latents_reused": 1,
                "latents_encoded": 0,
                "vae_cache_key": cache_key,
                "latent_cache_path": str(latent_path),
            },
            step=max(1, int(args.vae_steps)),
            namespace="wm_training",
        )
        return z

    reason = "force reencode" if args.force_latent_reencode or args.force_vae_retrain else "disabled reuse" if not args.reuse_latent_cache else "cache miss"
    print(f"[latent-cache] miss; encoding dataset ({reason})", flush=True)
    z = encode_dataset(vae, data, device, args.wm_batch_size)
    cache_dir.mkdir(parents=True, exist_ok=True)
    torch.save(z, latent_path)
    print(f"[latent-cache] saved latents to {latent_path}", flush=True)
    logger.log(
        {
            "latents_reused": 0,
            "latents_encoded": 1,
            "vae_cache_key": cache_key,
            "latent_cache_path": str(latent_path),
        },
        step=max(1, int(args.vae_steps)),
        namespace="wm_training",
    )
    return z


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
    progress = tqdm(
        range(1, args.vae_steps + 1),
        total=args.vae_steps,
        desc="World Model VAE",
        unit="step",
        dynamic_ncols=True,
    )
    for step in progress:
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
        if step == 1 or step % 10 == 0:
            progress.set_postfix({
                "loss": f"{float(loss.detach().cpu()):.4f}",
                "samples/s": f"{step * effective_batch_size / max(time.time() - start, 1e-6):.0f}",
            })
        if step == 1 or step % 100 == 0:
            metrics["vae_samples_per_second"] = step * effective_batch_size / max(time.time() - start, 1e-6)
            logger.log(metrics, step=step, namespace="wm_training")
        if args.eval_every and step % args.eval_every == 0:
            progress.write(f"[world model] VAE checkpoint step={step} path={checkpoint_dir / f'vae_step_{step}.pt'}")
            torch.save(model.state_dict(), checkpoint_dir / f"vae_step_{step}.pt")
    torch.save(model.state_dict(), checkpoint_dir / "vae.pt")
    print(f"[world model] saved final VAE checkpoint to {checkpoint_dir / 'vae.pt'}", flush=True)
    return model


@torch.no_grad()
def encode_dataset(vae, data, device, batch_size):
    episodes, steps, agents = observation_shape(data)
    z_size = int(getattr(vae, "z_size", 64))
    z = torch.empty((episodes * steps * agents, z_size), dtype=torch.float32)
    if has_compact_observations(data):
        flat_grid = data["obs_grid"].reshape(episodes * steps * agents, *data["obs_grid"].shape[3:])
        flat_self = data["obs_self"].reshape(episodes * steps * agents, data["obs_self"].shape[-1])
        total = flat_grid.shape[0]
        progress = tqdm(
            range(0, total, batch_size),
            total=(total + batch_size - 1) // batch_size,
            desc="World Model encode",
            unit="batch",
            dynamic_ncols=True,
        )
        for start in progress:
            end = min(start + batch_size, total)
            batch = vector_from_tabular(flat_grid[start:end], flat_self[start:end])
            z[start:end] = vae.encode(batch.to(device, non_blocking=True), sample=False).cpu()
    else:
        obs = data["obs"]
        if obs.ndim == 3:
            obs = obs.unsqueeze(2)
        flat_obs = obs.reshape(episodes * steps * agents, obs.shape[-1]).float()
        total = flat_obs.shape[0]
        progress = tqdm(
            range(0, total, batch_size),
            total=(total + batch_size - 1) // batch_size,
            desc="World Model encode",
            unit="batch",
            dynamic_ncols=True,
        )
        for start in progress:
            end = min(start + batch_size, total)
            z[start:end] = vae.encode(flat_obs[start:end].to(device, non_blocking=True), sample=False).cpu()
    encoded = z.reshape(episodes, steps, agents, z_size)
    print(f"[world model] encoded latent tensor shape={tuple(encoded.shape)}", flush=True)
    return encoded


def maybe_attach_symbolic_dataset_targets(data, baseline, args) -> None:
    variant = str(baseline.get("variant", "neural"))
    coverage = float(baseline.get("coverage", 0.0))
    if variant not in ("regularization", "residual") or coverage <= 0.0:
        print(
            f"[ns-mawm] symbolic training target cache skipped for variant={variant} coverage={coverage}",
            flush=True,
        )
        return
    attach_symbolic_dataset_targets(
        data,
        coverage=coverage,
        episode_limit=max(0, int(args.joint_symbolic_train_episodes)),
        step_limit=max(0, int(args.joint_symbolic_train_steps)),
    )


def attach_symbolic_dataset_targets(data, coverage: float, episode_limit: int | None = None, step_limit: int | None = None) -> None:
    if "symbolic_target" in data and "symbolic_mask" in data:
        return
    obs = all_agent_obs(observation_vectors(data, episode_limit=episode_limit))
    actions = normalize_action_tensor(data["action"])
    episodes, steps, agents, obs_size = obs.shape
    target_episodes = min(episodes, int(episode_limit)) if episode_limit else episodes
    target_steps = min(steps, int(step_limit)) if step_limit else steps
    if target_episodes <= 0 or target_steps <= 1:
        print("[ns-mawm] symbolic training target cache skipped because target limit is empty", flush=True)
        return
    targets = torch.zeros((target_episodes, target_steps, agents, obs_size), dtype=torch.float32)
    masks = torch.zeros_like(targets, dtype=torch.float32)
    map_coverages = []
    alignment_counts = []
    obs_np = obs.detach().cpu().numpy()
    actions_np = actions.detach().cpu().long().numpy()
    print(
        f"[ns-mawm] precomputing joint symbolic targets episodes={target_episodes}/{episodes} steps={target_steps}/{steps} agents={agents} coverage={coverage}",
        flush=True,
    )
    for episode_idx in range(target_episodes):
        memory = None
        for t in range(max(0, target_steps - 1)):
            current_joint = {
                f"agent_{agent_idx}": vector_to_tabular(obs_np[episode_idx, t, agent_idx])
                for agent_idx in range(agents)
            }
            joint_action = {
                f"agent_{agent_idx}": int(actions_np[episode_idx, t, agent_idx])
                for agent_idx in range(agents)
            }
            symbolic, mask, memory, report = symbolic_joint_transition(
                current_joint,
                joint_action,
                memory=memory,
                coverage=coverage,
            )
            map_coverages.append(float(report.get("map_coverage_ratio", 0.0)))
            alignment_counts.append(float(report.get("relative_alignment_count", 0.0)))
            for agent_idx in range(agents):
                agent_id = f"agent_{agent_idx}"
                targets[episode_idx, t, agent_idx] = torch.as_tensor(tabular_to_vector(symbolic[agent_id]))
                masks[episode_idx, t, agent_idx] = torch.as_tensor(tabular_mask_to_vector_mask(mask[agent_id]).astype(np.float32))
        if episode_idx == 0 or (episode_idx + 1) % max(1, target_episodes // 10) == 0:
            print(f"[ns-mawm] symbolic targets {episode_idx + 1}/{target_episodes}", flush=True)
    data["symbolic_target"] = targets
    data["symbolic_mask"] = masks
    data["symbolic_metadata"] = {
        "coverage": float(coverage),
        "mean_map_coverage_ratio": float(np.mean(map_coverages)) if map_coverages else 0.0,
        "mean_relative_alignment_count": float(np.mean(alignment_counts)) if alignment_counts else 0.0,
    }


def train_rnn(z, data, args, device, checkpoint_dir: Path, eval_dir: Path, logger, vae, baseline, step_offset: int = 0):
    model = TorchGridcraftRNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    z_seq, actions, rewards, dones = flatten_agent_sequences(z, data["action"], data["reward"], data["done"])
    valid = flatten_valid_agents(data.get("transition_valid"), z.shape[2] if z.ndim == 4 else 1)
    symbolic_variant = str(baseline.get("variant", "neural")) in ("regularization", "residual") and float(baseline.get("coverage", 0.0)) > 0.0
    raw_obs_joint = None
    raw_actions_joint = None
    if symbolic_variant:
        symbolic_episode_limit = min(observation_shape(data)[0], max(1, int(args.joint_symbolic_train_episodes)))
        raw_obs_joint = all_agent_obs(observation_vectors(data, episode_limit=symbolic_episode_limit))
        raw_actions_joint = normalize_action_tensor(data["action"][:symbolic_episode_limit])
    dataset = SequenceDataset(
        z_seq,
        actions,
        rewards,
        dones,
        seq_len=min(args.seq_len, z_seq.shape[1] - 1),
        valid=valid,
        obs=None,
    )
    if len(dataset) <= 0:
        raise RuntimeError("no valid RNN training sequences were available after terminal masking")
    for param in vae.parameters():
        param.requires_grad_(False)
    vae.eval()
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
    progress = tqdm(
        range(1, args.rnn_steps + 1),
        total=args.rnn_steps,
        desc="World Model MDN-RNN",
        unit="step",
        dynamic_ncols=True,
    )
    for step in progress:
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        if len(batch) == 5:
            z_seq, action, reward, done, obs_window = batch
        else:
            z_seq, action, reward, done = batch
            obs_window = None
        z_seq = z_seq.to(device, non_blocking=True)
        action = action.to(device, non_blocking=True)
        reward = reward.to(device, non_blocking=True)
        done = done.to(device, non_blocking=True)
        if obs_window is not None:
            obs_window = obs_window.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        loss, metrics = model.loss(z_seq, action, reward, done)
        ns_loss, ns_metrics = symbolic_training_loss(
            vae=vae,
            rnn=model,
            z_seq=z_seq,
            action=action,
            obs_window=obs_window,
            baseline=baseline,
            args=args,
            device=device,
        )
        joint_ns_loss, joint_ns_metrics = joint_symbolic_training_loss(
            vae=vae,
            rnn=model,
            z=z,
            raw_actions=raw_actions_joint,
            raw_obs=raw_obs_joint,
            symbolic_target=data.get("symbolic_target"),
            symbolic_mask=data.get("symbolic_mask"),
            baseline=baseline,
            args=args,
            device=device,
            step=step,
        )
        if ns_loss is not None:
            loss = loss + ns_loss
            metrics.update(ns_metrics)
        if joint_ns_loss is not None:
            loss = loss + joint_ns_loss
            metrics.update(joint_ns_metrics)
        if ns_loss is not None or joint_ns_loss is not None:
            metrics["training_symbolic_loss"] = (
                float(metrics.get("training_symbolic_loss_individual", 0.0))
                + float(metrics.get("training_symbolic_loss_joint", 0.0))
            )
            metrics["training_residual_loss"] = (
                float(metrics.get("training_residual_loss_individual", 0.0))
                + float(metrics.get("training_residual_loss_joint", 0.0))
            )
            metrics["training_wm_total_loss"] = float(loss.detach().cpu())
        loss.backward()
        optimizer.step()
        global_step = step_offset + step
        if step == 1 or step % 10 == 0:
            progress.set_postfix({
                "loss": f"{float(loss.detach().cpu()):.4f}",
                "seq/s": f"{step * effective_batch_size / max(time.time() - start, 1e-6):.0f}",
            })
        if step == 1 or step % 100 == 0:
            metrics["rnn_sequences_per_second"] = step * effective_batch_size / max(time.time() - start, 1e-6)
            logger.log(metrics, step=global_step, namespace="wm_training")
        if args.eval_every and step % args.eval_every == 0:
            checkpoint = checkpoint_dir / f"rnn_step_{step}.pt"
            torch.save(model.state_dict(), checkpoint)
            progress.write(f"[world model] evaluating RNN checkpoint step={step} global_step={global_step}")
            metrics, pstr_rows = evaluate_world_model(vae, model, data, device, horizons=args.horizons, baseline=baseline, args=args)
            metrics["checkpoint_step"] = step
            (eval_dir / f"world_model_step_{step}.json").write_text(json.dumps(metrics, indent=2))
            write_pstr_rvr_table(eval_dir / f"pstr_rvr_table_step_{step}.json", pstr_rows)
            logger.log(metrics, step=global_step, namespace="wm_evaluation")
            log_pstr_rvr_table(logger, pstr_rows, step=global_step)
            progress.write(
                "[world model] eval "
                f"step={step} grid_mismatch={metrics.get('grid_mismatch', float('nan')):.4f} "
                f"self_mse={metrics.get('self_mse', float('nan')):.4f} "
                f"rvr={metrics.get('rvr_global', float('nan')):.4f}"
            )
            if args.video_every and step % args.video_every == 0:
                progress.write(f"[world model] logging comparison video step={step} global_step={global_step}")
                log_world_model_video(logger, args, args_to_config(args), vae, model, device, step=global_step)
    torch.save(model.state_dict(), checkpoint_dir / "rnn.pt")
    print(f"[world model] saved final RNN checkpoint to {checkpoint_dir / 'rnn.pt'}", flush=True)
    return model


@torch.no_grad()
def evaluate_world_model(vae, rnn, data, device, horizons, baseline=None, args=None):
    episodes, _, _ = observation_shape(data)
    raw_obs = all_agent_obs(observation_vectors(data, episode_limit=min(128, episodes))).to(device)
    action_data = data["action"][: raw_obs.shape[0]]
    raw_actions = normalize_action_tensor(action_data).to(device)
    flat_obs = flatten_obs_agents(raw_obs)
    actions = flatten_action_agents(raw_actions)
    z = vae.encode(flat_obs.reshape(-1, flat_obs.shape[-1]), sample=False).reshape(flat_obs.shape[0], flat_obs.shape[1], -1)
    result = {}
    for horizon in horizons:
        h = min(int(horizon), z.shape[1] - 1)
        current_z = z[:, 0]
        state = None
        pred_z = current_z
        for t in range(h):
            pred_z, _, _, state = rnn.step(pred_z, actions[:, t], state, deterministic=True)
        decoded = vae.decode(pred_z)
        target = flat_obs[:, h]
        result[f"compounding_grid_mismatch_h{h}"] = float(grid_mismatch(decoded, target).cpu())
        result[f"compounding_self_mse_h{h}"] = float(torch.mean((decoded[:, -11:] - target[:, -11:]) ** 2).cpu())
    pred_one, _, _, _ = rnn.step(z[:, 0], actions[:, 0], None, deterministic=True)
    decoded_one = vae.decode(pred_one)
    result["grid_mismatch"] = float(grid_mismatch(decoded_one, flat_obs[:, 1]).cpu())
    result["self_mse"] = float(torch.mean((decoded_one[:, -11:] - flat_obs[:, 1, -11:]) ** 2).cpu())
    symbolic_metrics, pstr_rows = evaluate_symbolic_rvr(
        vae=vae,
        rnn=rnn,
        z=z,
        raw_actions=raw_actions,
        raw_obs=raw_obs,
        baseline=baseline or {},
        device=device,
        max_eval_steps=getattr(args, "rvr_eval_steps", 50) if args is not None else 50,
    )
    result.update(symbolic_metrics)
    return result, pstr_rows


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


def normalize_action_tensor(actions: torch.Tensor) -> torch.Tensor:
    if actions.ndim == 2:
        return actions.unsqueeze(2)
    if actions.ndim == 3:
        return actions
    if actions.ndim == 4 and actions.shape[-1] == 1:
        return normalize_action_tensor(actions.squeeze(-1))
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


def flatten_valid_agents(valid: torch.Tensor | None, num_agents: int) -> torch.Tensor | None:
    if valid is None:
        return None
    if valid.ndim == 2:
        episodes, steps = valid.shape
        return valid[:, None, :].expand(episodes, num_agents, steps).reshape(episodes * num_agents, steps)
    if valid.ndim == 3:
        episodes, steps, agents = valid.shape
        return valid.permute(0, 2, 1).reshape(episodes * agents, steps)
    if valid.ndim == 4 and valid.shape[-1] == 1:
        return flatten_valid_agents(valid.squeeze(-1), num_agents)
    raise ValueError(f"unsupported valid tensor shape: {tuple(valid.shape)}")


def symbolic_training_loss(vae, rnn, z_seq, action, obs_window, baseline, args, device):
    variant = str(baseline.get("variant", "neural"))
    if variant not in ("regularization", "residual"):
        return None, {}
    if obs_window is None:
        return None, {}
    coverage = float(baseline.get("coverage", 0.0))
    if coverage <= 0.0:
        return None, {
            "training_symbolic_loss": 0.0,
            "training_residual_loss": 0.0,
            "training_symbolic_loss_individual": 0.0,
            "training_residual_loss_individual": 0.0,
            "training_symbolic_mask_count_individual": 0.0,
        }
    limit = min(int(args.symbolic_train_samples), z_seq.shape[0])
    if limit <= 0:
        return None, {}
    pred_input = z_seq[:limit, :-1]
    actions = action[:limit]
    (logmix, mean, _logstd, _reward_pred, _done_logit), _ = rnn.forward(pred_input, actions)
    expected_z = (logmix.exp() * mean).sum(dim=-1)
    decoded = vae.decode(expected_z.reshape(-1, expected_z.shape[-1])).reshape(limit, actions.shape[1], -1)
    obs_np = obs_window[:limit, :-1].detach().cpu().numpy()
    action_np = actions.detach().cpu().numpy()
    target_np, mask_np = symbolic_batch_targets(obs_np, action_np, coverage=coverage)
    target = torch.as_tensor(target_np, dtype=torch.float32, device=device)
    mask = torch.as_tensor(mask_np, dtype=torch.float32, device=device)
    mask_count = mask.sum().clamp_min(1.0)
    symbolic_loss = (((decoded - target) ** 2) * mask).sum() / mask_count
    residual_mask = 1.0 - mask
    residual_target = obs_window[:limit, 1:].detach()
    residual_count = residual_mask.sum().clamp_min(1.0)
    residual_loss = (((decoded - residual_target) ** 2) * residual_mask).sum() / residual_count
    if variant == "regularization":
        total = float(args.lambda_sym) * symbolic_loss
    else:
        total = float(args.lambda_residual) * residual_loss
    return total, {
        "training_symbolic_loss_individual": float(symbolic_loss.detach().cpu()),
        "training_residual_loss_individual": float(residual_loss.detach().cpu()),
        "training_symbolic_mask_count_individual": float(mask.sum().detach().cpu()),
    }


def joint_symbolic_training_loss(vae, rnn, z, raw_actions, raw_obs, symbolic_target, symbolic_mask, baseline, args, device, step):
    variant = str(baseline.get("variant", "neural"))
    if variant not in ("regularization", "residual"):
        return None, {}
    coverage = float(baseline.get("coverage", 0.0))
    if coverage <= 0.0:
        return None, {
            "training_symbolic_loss_joint": 0.0,
            "training_residual_loss_joint": 0.0,
            "training_symbolic_mask_count_joint": 0.0,
        }
    episodes, steps, agents, obs_size = raw_obs.shape
    seq_len = min(int(args.joint_symbolic_train_steps), steps - 1)
    batch_episodes = min(int(args.joint_symbolic_train_episodes), episodes)
    if seq_len <= 0 or batch_episodes <= 0:
        return None, {}

    max_start = max(1, episodes - batch_episodes + 1)
    start = ((int(step) - 1) * batch_episodes) % max_start
    idx = torch.arange(start, start + batch_episodes, dtype=torch.long) % episodes
    z_window = z[idx, :seq_len].to(device, non_blocking=True)
    action_window = raw_actions[idx, :seq_len].to(device, non_blocking=True)
    next_obs_window = raw_obs[idx, 1 : seq_len + 1].to(device, non_blocking=True)
    if symbolic_target is not None and symbolic_mask is not None:
        target_window = symbolic_target[idx, :seq_len].to(device, non_blocking=True)
        mask_window = symbolic_mask[idx, :seq_len].to(device, non_blocking=True)
    else:
        target_window = None
        mask_window = None
        obs_np = raw_obs[idx, :seq_len].detach().cpu().numpy()
        action_np = raw_actions[idx, :seq_len].detach().cpu().long().numpy()

    memories = [None for _ in range(batch_episodes)]
    state = None
    symbolic_losses = []
    residual_losses = []
    mask_counts = []
    for t in range(seq_len):
        z_t = z_window[:, t].reshape(batch_episodes * agents, -1)
        actions_t = action_window[:, t].reshape(batch_episodes * agents)
        pred_z, _, _, state = rnn.step(z_t, actions_t, state, deterministic=True)
        decoded = vae.decode(pred_z).reshape(batch_episodes, agents, obs_size)
        if target_window is not None and mask_window is not None:
            target = target_window[:, t]
            mask = mask_window[:, t]
        else:
            target_rows = []
            mask_rows = []
            for episode_idx in range(batch_episodes):
                current_joint = {
                    f"agent_{agent_idx}": vector_to_tabular(obs_np[episode_idx, t, agent_idx])
                    for agent_idx in range(agents)
                }
                joint_action = {
                    f"agent_{agent_idx}": int(action_np[episode_idx, t, agent_idx])
                    for agent_idx in range(agents)
                }
                symbolic, mask, memories[episode_idx], _report = symbolic_joint_transition(
                    current_joint,
                    joint_action,
                    memory=memories[episode_idx],
                    coverage=coverage,
                )
                for agent_idx in range(agents):
                    agent_id = f"agent_{agent_idx}"
                    target_rows.append(tabular_to_vector(symbolic[agent_id]))
                    mask_rows.append(tabular_mask_to_vector_mask(mask[agent_id]).astype(np.float32))
            target = torch.as_tensor(np.stack(target_rows), dtype=torch.float32, device=device).reshape(batch_episodes, agents, obs_size)
            mask = torch.as_tensor(np.stack(mask_rows), dtype=torch.float32, device=device).reshape(batch_episodes, agents, obs_size)
        mask_count = mask.sum().clamp_min(1.0)
        symbolic_loss = (((decoded - target) ** 2) * mask).sum() / mask_count
        residual_mask = 1.0 - mask
        residual_target = next_obs_window[:, t]
        residual_count = residual_mask.sum().clamp_min(1.0)
        residual_loss = (((decoded - residual_target) ** 2) * residual_mask).sum() / residual_count
        symbolic_losses.append(symbolic_loss)
        residual_losses.append(residual_loss)
        mask_counts.append(mask.sum())

    symbolic_loss = torch.stack(symbolic_losses).mean()
    residual_loss = torch.stack(residual_losses).mean()
    if variant == "regularization":
        total = float(args.lambda_sym) * symbolic_loss
    else:
        total = float(args.lambda_residual) * residual_loss
    return total, {
        "training_symbolic_loss_joint": float(symbolic_loss.detach().cpu()),
        "training_residual_loss_joint": float(residual_loss.detach().cpu()),
        "training_symbolic_mask_count_joint": float(torch.stack(mask_counts).sum().detach().cpu()),
    }


@torch.no_grad()
def evaluate_symbolic_rvr(vae, rnn, z, raw_actions, raw_obs, baseline, device, max_eval_steps=50):
    episodes, steps, agents, obs_size = raw_obs.shape
    if steps < 2:
        return empty_symbolic_metrics(), pstr_rvr_rows({}, baseline)
    flat_actions = flatten_action_agents(raw_actions)
    eval_steps = min(int(max_eval_steps), steps - 1)
    memories = [None for _ in range(episodes)]
    post_memories = [None for _ in range(episodes)]
    row_metrics = []
    post_rows = []
    state = None
    variant = str(baseline.get("variant", "neural"))
    coverage = float(baseline.get("coverage", 0.0))
    for t in range(eval_steps):
        pred_z, _, _, state = rnn.step(z[:, t], flat_actions[:, t], state, deterministic=True)
        pred_tabular = decode_flat_tabular(vae, pred_z, episodes, agents)
        current_tabular = vectors_to_joint_tabular(raw_obs[:, t])
        joint_actions = actions_to_joint(raw_actions[:, t])
        for episode_idx in range(episodes):
            symbolic, mask, updated_memory, report = symbolic_joint_transition(
                current_tabular[episode_idx],
                joint_actions[episode_idx],
                memory=memories[episode_idx],
                coverage=1.0,
            )
            memories[episode_idx] = updated_memory
            row_metrics.append(compare_joint_with_symbolic(pred_tabular[episode_idx], symbolic, mask, report))
            if variant in ("projection", "residual"):
                projected, symbolic_info = apply_symbolic_projection(
                    pred_tabular[episode_idx],
                    current_tabular[episode_idx],
                    joint_actions[episode_idx],
                    variant,
                    coverage=coverage,
                    memory=post_memories[episode_idx],
                )
                if symbolic_info is not None and len(symbolic_info) >= 3:
                    post_memories[episode_idx] = symbolic_info[2]
                if symbolic_info is not None and len(symbolic_info) >= 4:
                    symbolic, mask, _updated_memory, report = symbolic_info
                    post_rows.append(compare_joint_with_symbolic(projected, symbolic, mask, report))
    metrics = average_numeric_metrics(row_metrics)
    metrics.setdefault("rvr_global", metrics.get("rvr", 0.0))
    metrics.setdefault("rvr_individual", 0.0)
    metrics.setdefault("rvr_joint", 0.0)
    metrics["rvr_eval_steps"] = float(eval_steps)
    pstr_values = {key[len("rvr/"):]: value for key, value in metrics.items() if key.startswith("rvr/")}
    if post_rows:
        post = average_numeric_metrics(post_rows)
        if "rvr_global" in post:
            metrics["rvr_post_global"] = post["rvr_global"]
        elif "rvr" in post:
            metrics["rvr_post_global"] = post["rvr"]
    return metrics, pstr_rvr_rows(pstr_values, baseline)


def empty_symbolic_metrics():
    return {
        "rvr_global": 0.0,
        "rvr_individual": 0.0,
        "rvr_joint": 0.0,
        "determinable_count": 0.0,
        "determinable_count_individual": 0.0,
        "determinable_count_joint": 0.0,
        "map_coverage_ratio": 0.0,
        "relative_alignment_count": 0.0,
    }


def average_numeric_metrics(rows):
    if not rows:
        return empty_symbolic_metrics()
    keys = sorted({key for row in rows for key in row})
    result = {}
    for key in keys:
        values = [float(row[key]) for row in rows if key in row and isinstance(row[key], (int, float, np.integer, np.floating))]
        if values:
            result[key] = float(np.mean(values))
    return result


@torch.no_grad()
def decode_flat_tabular(vae, z, episodes, agents):
    decoded = vae.decode_tabular(z)
    grids = decoded["grid"].detach().cpu().numpy().astype("int8")
    selves = decoded["self"].detach().cpu().numpy().astype("int16")
    result = []
    cursor = 0
    for _ in range(episodes):
        row = {}
        for agent_idx in range(agents):
            row[f"agent_{agent_idx}"] = {"grid": grids[cursor], "self": selves[cursor]}
            cursor += 1
        result.append(row)
    return result


def vectors_to_joint_tabular(obs):
    obs_np = obs.detach().cpu().numpy()
    episodes, agents = obs_np.shape[:2]
    result = []
    for episode_idx in range(episodes):
        row = {}
        for agent_idx in range(agents):
            row[f"agent_{agent_idx}"] = vector_to_tabular(obs_np[episode_idx, agent_idx])
        result.append(row)
    return result


def actions_to_joint(actions):
    actions_np = actions.detach().cpu().long().numpy()
    episodes, agents = actions_np.shape[:2]
    result = []
    for episode_idx in range(episodes):
        result.append({f"agent_{agent_idx}": int(actions_np[episode_idx, agent_idx]) for agent_idx in range(agents)})
    return result


def pstr_rvr_rows(pstr_values, baseline=None):
    variant = str((baseline or {}).get("variant", "neural"))
    force_na = variant in ("projection", "residual")
    rows = []
    for rule in PSTR_RULES:
        rule_id = rule["id"]
        value = pstr_values.get(rule_id)
        rows.append(
            {
                "PSTR id": rule_id,
                "RVR": "n/a" if force_na or value is None else float(value),
            }
        )
    return rows


def write_pstr_rvr_table(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(rows, f, indent=2)


def log_pstr_rvr_table(logger, rows, step=None):
    table_rows = [[row["PSTR id"], format_pstr_rvr_value(row["RVR"])] for row in rows]
    logger.log_table("PSTR RVR table", ["PSTR id", "RVR"], table_rows, step=step, namespace="wm_evaluation")


def format_pstr_rvr_value(value):
    if value == "n/a" or value is None:
        return "n/a"
    return f"{float(value):.6f}"


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
    metrics = {
        f"eval_{prefix}_reward": float(returns.mean().cpu()),
        "episode_length": float(lengths.mean().cpu()),
        "policy_backend": "vectorized_random_smoke",
    }
    if not model_based:
        metrics["eval_real_reward"] = float(returns.mean().cpu())
    return metrics


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
    baseline = BASELINES.get(getattr(args, "baseline_id", ""), {})
    ns_variant = str(baseline.get("variant", "neural"))
    ns_coverage = float(baseline.get("coverage", 0.0))
    ns_memory = None
    max_steps = max(1, int(args.video_max_steps))
    for _ in range(max_steps):
        current_tabular = observation_to_joint_tabular(obs["vector"][0])
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
        if ns_variant in ("projection", "residual"):
            joint_action = {f"agent_{agent_idx}": int(action[0, agent_idx].detach().cpu()) for agent_idx in range(config.num_agents)}
            tabular, symbolic_info = apply_symbolic_projection(
                tabular,
                current_tabular,
                joint_action,
                ns_variant,
                coverage=ns_coverage,
                memory=ns_memory,
            )
            if symbolic_info is not None and len(symbolic_info) >= 3:
                ns_memory = symbolic_info[2]
        frame = env.render(env_index=0, mode="rgb_array", tabular_observations=tabular)
        frames.append(frame[:, :, :3] if frame.shape[-1] == 4 else frame)
        if ns_variant in ("projection", "residual"):
            projected = torch.as_tensor(
                np.asarray([tabular_to_vector(tabular[f"agent_{idx}"]) for idx in range(config.num_agents)], dtype=np.float32),
                device=device,
            )
            z = vae.encode(projected, sample=False)
        else:
            z = vae.encode(obs["vector"][0].to(device), sample=False)
        if bool((done | truncated).all()):
            break
    env.close()
    return frames


def observation_to_joint_tabular(agent_vectors: torch.Tensor):
    vectors = agent_vectors.detach().cpu().numpy()
    return {f"agent_{idx}": vector_to_tabular(vectors[idx]) for idx in range(vectors.shape[0])}


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
