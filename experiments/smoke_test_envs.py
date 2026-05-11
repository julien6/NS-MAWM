"""Instantiate and step all available environment adapters."""

from __future__ import annotations

import torch

from env_adapters import make_environment


def main() -> None:
    for name in ("predator_prey", "gridcraft", "overcooked"):
        env = make_environment(name)
        obs = env.reset(seed=0)
        action = torch.zeros(env.n_agents, env.action_dim)
        action[:, 0] = 1.0
        next_obs, reward, done, _info = env.step(action.reshape(-1))
        print(name, {"obs": tuple(obs.shape), "next_obs": tuple(next_obs.shape), "reward": tuple(reward.shape), "done": bool(done.item())})
    try:
        env = make_environment("smacv2")
        obs = env.reset(seed=0)
        action = torch.zeros(env.n_agents, env.action_dim)
        action[:, 0] = 1.0
        next_obs, reward, done, _info = env.step(action.reshape(-1))
        print("smacv2", {"obs": tuple(obs.shape), "next_obs": tuple(next_obs.shape), "reward": tuple(reward.shape), "done": bool(done.item())})
    except Exception as exc:
        print("smacv2 skipped", type(exc).__name__, str(exc))


if __name__ == "__main__":
    main()
