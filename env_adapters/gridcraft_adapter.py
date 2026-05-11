"""Adapter for the installed GridCraft package."""

from __future__ import annotations

import torch

from env_adapters.flat import ActionSpec, flatten_multiagent_obs, generic_schema
from ns_mawm.features import FeatureSchema, FeatureSpec, FeatureType


class GridCraftAdapter:
    def __init__(self, n_agents: int = 2, seed: int = 0, max_steps: int = 50):
        from gridcraft import GridcraftConfig, GridcraftEnv
        from gridcraft.constants import ACTION_NAMES

        self.config = GridcraftConfig(width=12, height=12, num_agents=n_agents, max_steps=max_steps, seed=seed)
        self.env = GridcraftEnv(config=self.config)
        self.n_agents = n_agents
        self.action_dim = len(ACTION_NAMES)
        self.variant_id = "gridcraft_sv"
        self.agent_order = [f"agent_{i}" for i in range(n_agents)]
        first, _ = self.env.reset(seed=seed)
        self.obs_dim = int(flatten_multiagent_obs(first, self.agent_order).numel())
        self._schema = self._build_schema(first)
        self._action_spec = ActionSpec(n_agents, self.action_dim, tuple(ACTION_NAMES))

    @property
    def schema(self) -> FeatureSchema:
        return self._schema

    @property
    def action_spec(self) -> ActionSpec:
        return self._action_spec

    def _build_schema(self, raw_obs) -> FeatureSchema:
        specs: list[FeatureSpec] = []
        offset = 0
        for agent in self.agent_order:
            flat = flatten_multiagent_obs({agent: raw_obs[agent]}, [agent])
            for i in range(flat.numel()):
                family = "agent_state" if i >= flat.numel() - 11 else "local_grid"
                specs.append(FeatureSpec(f"{agent}.feature_{i}", FeatureType.CONTINUOUS, owner=agent, family=family))
            offset += int(flat.numel())
        return FeatureSchema.from_specs(specs)

    def reset(self, seed: int | None = None) -> torch.Tensor:
        obs, _infos = self.env.reset(seed=seed)
        return flatten_multiagent_obs(obs, self.agent_order)

    def step(self, action: torch.Tensor):
        idx = action.reshape(self.n_agents, self.action_dim).argmax(dim=-1).tolist()
        action_dict = {agent: int(idx[i]) for i, agent in enumerate(self.agent_order) if agent in self.env.agents}
        obs, rewards, terms, truncs, infos = self.env.step(action_dict)
        done = all(terms.values()) or all(truncs.values())
        reward = sum(float(v) for v in rewards.values()) / max(len(rewards), 1)
        return flatten_multiagent_obs(obs, self.agent_order), torch.tensor([reward]), torch.tensor([done]), infos

    def decode(self, obs: torch.Tensor) -> dict[str, object]:
        return self.schema.decode_tensor(obs)

    def encode_action(self, actions) -> torch.Tensor:
        if isinstance(actions, torch.Tensor):
            return actions.float().reshape(-1)
        idx = torch.as_tensor(actions, dtype=torch.long).reshape(self.n_agents)
        return torch.nn.functional.one_hot(idx, self.action_dim).float().reshape(-1)
