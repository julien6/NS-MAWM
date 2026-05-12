"""Adapter for SMAC PettingZoo environments."""

from __future__ import annotations

import contextlib

import torch

from env_adapters.flat import ActionSpec, VariantSpec, default_variants, flatten_multiagent_obs, resolve_variant_id, semantic_schema
from ns_mawm.features import FeatureSchema


class SMACAdapter:
    def __init__(self, map_name: str = "8m", max_steps: int | None = 5):
        from smac.env.pettingzoo import StarCraft2PZEnv as sc2

        self.env = sc2.parallel_env(map_name=map_name, max_cycles=max_steps)
        obs, _infos = self.env.reset()
        self.agent_order = list(self.env.possible_agents)
        self.n_agents = len(self.agent_order)
        self.action_dim = max(self.env.action_spaces[agent].n for agent in self.agent_order)
        self.variant_id = f"smac:{map_name}"
        self.obs_dim = int(flatten_multiagent_obs(obs, self.agent_order).numel())
        self._schema = semantic_schema(
            self.obs_dim,
            "smac",
            ("position", "movement_feasibility", "range", "action_mask", "health", "unit_activity"),
        )
        self._action_spec = ActionSpec(self.n_agents, self.action_dim, tuple(str(i) for i in range(self.action_dim)))

    @property
    def schema(self) -> FeatureSchema:
        return self._schema

    @property
    def action_spec(self) -> ActionSpec:
        return self._action_spec

    def make_variants(self, split: str) -> tuple[VariantSpec, ...]:
        return default_variants("smac", split)

    def reset(self, seed: int | None = None, variant: str | VariantSpec | None = None) -> torch.Tensor:
        variant_id, seed_offset = resolve_variant_id(f"smac:{self.variant_id.split(':', 1)[-1]}", variant)
        self.variant_id = variant_id
        obs, _infos = self.env.reset(seed=None if seed is None else seed + seed_offset)
        return flatten_multiagent_obs(obs, self.agent_order)

    def step(self, action: torch.Tensor):
        idx = action.reshape(self.n_agents, self.action_dim).argmax(dim=-1).tolist()
        obs_preview = self.env._observe_all()
        action_dict = {
            agent: self._safe_action(agent, int(idx[i]), obs_preview)
            for i, agent in enumerate(self.agent_order)
            if agent in self.env.agents
        }
        obs, rewards, terms, truncs, infos = self.env.step(action_dict)
        done = all(terms.values()) or all(truncs.values())
        reward = sum(float(v) for v in rewards.values()) / max(len(rewards), 1)
        return flatten_multiagent_obs(obs, self.agent_order), torch.tensor([reward]), torch.tensor([done]), infos

    def _safe_action(self, agent: str, proposed: int, obs_preview: dict) -> int:
        n = self.env.action_spaces[agent].n
        action = proposed % n
        mask = obs_preview.get(agent, {}).get("action_mask")
        if mask is None or int(mask[action]) == 1:
            return action
        for fallback, available in enumerate(mask):
            if int(available) == 1:
                return fallback
        return 0

    def decode(self, obs: torch.Tensor) -> dict[str, object]:
        return self.schema.decode_tensor(obs)

    def encode_action(self, actions) -> torch.Tensor:
        if isinstance(actions, torch.Tensor):
            return actions.float().reshape(-1)
        idx = torch.as_tensor(actions, dtype=torch.long).reshape(self.n_agents)
        return torch.nn.functional.one_hot(idx, self.action_dim).float().reshape(-1)

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self.env.close()

    def __del__(self) -> None:
        self.close()
