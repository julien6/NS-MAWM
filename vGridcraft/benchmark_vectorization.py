from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vGridcraft"))

from vgridcraft import VGridcraftConfig, VectorizedGridcraftEnv
from vgridcraft.dataset import collect_dataset


PROFILES = {
    "small": {"num_envs": 256, "num_agents": 3, "steps": 100},
    "spark": {"num_envs": 1024, "num_agents": 3, "steps": 200},
    "stress": {"num_envs": 4096, "num_agents": 3, "steps": 200},
}


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed(device: torch.device, fn):
    synchronize(device)
    start = time.time()
    result = fn()
    synchronize(device)
    return time.time() - start, result


def benchmark(args) -> dict[str, float | int | str]:
    profile = PROFILES[args.profile].copy()
    num_envs = int(args.num_envs or profile["num_envs"])
    num_agents = int(args.num_agents or profile["num_agents"])
    steps = int(args.steps or profile["steps"])
    device = torch.device(args.device)
    config = VGridcraftConfig(
        num_agents=num_agents,
        max_steps=args.max_steps,
        seed=args.seed,
        max_mobs=args.max_mobs,
        max_items=args.max_items,
        view_size=args.view_size,
    )
    env = VectorizedGridcraftEnv(num_envs=num_envs, num_agents=num_agents, device=device, seed=args.seed, config=config)
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed + 13)
    try:
        env.reset()

        def run_steps():
            obs = None
            for _ in range(steps):
                actions = torch.randint(0, config.action_size, (num_envs, num_agents), generator=generator, device=device)
                obs, *_ = env.step(actions)
            return obs

        step_time, _ = timed(device, run_steps)
        observation_time, _ = timed(device, lambda: env.observation())
        local_grid_time, _ = timed(device, lambda: env.local_grid())
    finally:
        env.close()

    dataset_episodes = int(args.dataset_episodes)
    dataset_time, data = timed(
        device,
        lambda: collect_dataset(
            episodes=dataset_episodes,
            max_steps=args.dataset_max_steps,
            num_envs=min(num_envs, dataset_episodes),
            device=str(device),
            seed=args.seed + 100,
            config=VGridcraftConfig(
                num_agents=num_agents,
                max_steps=args.dataset_max_steps,
                seed=args.seed + 100,
                max_mobs=args.max_mobs,
                max_items=args.max_items,
                view_size=args.view_size,
            ),
        ),
    )

    return {
        "device": str(device),
        "profile": args.profile,
        "num_envs": num_envs,
        "num_agents": num_agents,
        "max_mobs": args.max_mobs,
        "max_items": args.max_items,
        "view_size": args.view_size,
        "steps": steps,
        "env_steps_per_second": float(num_envs * steps / max(step_time, 1e-9)),
        "agent_steps_per_second": float(num_envs * num_agents * steps / max(step_time, 1e-9)),
        "step_time_ms": float(1000.0 * step_time / max(steps, 1)),
        "observation_time_ms": float(1000.0 * observation_time),
        "local_grid_time_ms": float(1000.0 * local_grid_time),
        "dataset_collection_seconds": float(dataset_time),
        "dataset_collection_env_steps_per_second": float(data["metadata"].get("collection_env_steps_per_second", 0.0)),
        "dataset_collection_agent_steps_per_second": float(data["metadata"].get("collection_agent_steps_per_second", 0.0)),
        "dataset_cpu_copy_time": float(data["metadata"].get("collection_cpu_copy_time", 0.0)),
        "dataset_valid_transition_count": int(data["metadata"].get("valid_transition_count", 0)),
        "dataset_invalid_padded_transition_count": int(data["metadata"].get("invalid_padded_transition_count", 0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=sorted(PROFILES), default="small")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--num-agents", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--view-size", type=int, default=7)
    parser.add_argument("--max-mobs", type=int, default=6)
    parser.add_argument("--max-items", type=int, default=32)
    parser.add_argument("--dataset-episodes", type=int, default=64)
    parser.add_argument("--dataset-max-steps", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    print(json.dumps(benchmark(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
