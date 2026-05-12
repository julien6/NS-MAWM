"""Training utilities for WM and policy smoke/full runs."""

from __future__ import annotations

import time
import platform
import resource
import subprocess

import torch
import torch.nn.functional as F

from marl_lib import ReplayBuffer, collect_transitions, train_policy


def seed_everything(seed: int) -> None:
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def resource_snapshot() -> dict[str, float | str]:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    ram_mb = float(usage.ru_maxrss) / (1024.0 if platform.system() != "Darwin" else 1024.0 * 1024.0)
    device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    return {"cpu_percent": 0.0, "ram_used_mb": ram_mb, "device": device}


def git_commit_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def collect_replay(env, policy, steps: int, seed: int, variant=None) -> ReplayBuffer:
    replay = ReplayBuffer(capacity=max(steps * 4, 64), seed=seed)
    batch = collect_transitions(env, policy, steps=steps, seed=seed, variant=variant)
    replay.extend(batch, variant_id=getattr(env, "variant_id", "default"), seed_id=seed)
    return replay


def _sample_world_model_batch(replay: ReplayBuffer, batch_size: int, sequence_length: int):
    if sequence_length > 1 and len(replay) >= sequence_length:
        max_sequences = max(1, len(replay) - sequence_length + 1)
        return replay.sample_sequence(min(batch_size, max_sequences), sequence_length)
    return replay.sample(min(batch_size, len(replay)))


def train_world_model(
    model,
    replay: ReplayBuffer,
    updates: int,
    batch_size: int,
    lr: float = 1e-3,
    sequence_length: int = 1,
) -> list[dict[str, float]]:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    rows: list[dict[str, float]] = []
    for step in range(updates):
        batch = _sample_world_model_batch(replay, batch_size, sequence_length)
        start = time.perf_counter()
        loss, metrics = model.loss(batch.obs, batch.action, batch.next_obs, batch.reward, batch.done)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        rows.append(
            {
                "step": step,
                "sequence_length": sequence_length,
                "wm_total_loss": float(loss.detach()),
                "convergence_speed": float(1.0 / (1.0 + float(loss.detach()))),
                **{("kl_loss" if key == "kl" else key): float(value.detach()) for key, value in metrics.items()},
                "wm_update_time_ms": (time.perf_counter() - start) * 1000.0,
                **resource_snapshot(),
            }
        )
    return rows


def train_model_free_policy(policy, replay: ReplayBuffer, updates: int, batch_size: int, learning_rate: float = 1e-3) -> list[dict[str, float]]:
    return train_policy(policy, replay, steps=updates, batch_size=batch_size, learning_rate=learning_rate)
