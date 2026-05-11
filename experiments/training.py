"""Training utilities for WM and policy smoke/full runs."""

from __future__ import annotations

import time

import torch
import torch.nn.functional as F

from marl_lib import ReplayBuffer, collect_transitions, train_policy


def seed_everything(seed: int) -> None:
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def collect_replay(env, policy, steps: int, seed: int) -> ReplayBuffer:
    replay = ReplayBuffer(capacity=max(steps * 4, 64), seed=seed)
    batch = collect_transitions(env, policy, steps=steps, seed=seed)
    replay.extend(batch, variant_id=getattr(env, "variant_id", "default"), seed_id=seed)
    return replay


def train_world_model(model, replay: ReplayBuffer, updates: int, batch_size: int, lr: float = 1e-3) -> list[dict[str, float]]:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    rows: list[dict[str, float]] = []
    for step in range(updates):
        batch = replay.sample(min(batch_size, len(replay)))
        start = time.perf_counter()
        loss, metrics = model.loss(batch.obs, batch.action, batch.next_obs, batch.reward, batch.done)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        rows.append(
            {
                "step": step,
                "wm_total_loss": float(loss.detach()),
                **{("kl_loss" if key == "kl" else key): float(value.detach()) for key, value in metrics.items()},
                "wm_update_time_ms": (time.perf_counter() - start) * 1000.0,
                "cpu_percent": 0.0,
                "ram_used_mb": 0.0,
            }
        )
    return rows


def train_model_free_policy(policy, replay: ReplayBuffer, updates: int, batch_size: int) -> list[dict[str, float]]:
    return train_policy(policy, replay, steps=updates, batch_size=batch_size)
