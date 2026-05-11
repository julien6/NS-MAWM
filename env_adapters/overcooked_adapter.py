"""Adapter for installed Overcooked-AI."""

from __future__ import annotations

import torch
import numpy as np

from env_adapters.flat import ActionSpec, generic_schema
from ns_mawm.features import FeatureSchema


class OvercookedAdapter:
    def __init__(self, layout: str = "cramped_room", horizon: int = 50):
        from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
        from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
        from overcooked_ai_py.mdp.actions import Action

        mdp = OvercookedGridworld.from_layout_name(layout)
        self.env = OvercookedEnv.from_mdp(mdp, horizon=horizon)
        self.n_agents = 2
        self.action_dim = len(Action.ALL_ACTIONS)
        self.variant_id = f"overcooked:{layout}"
        self._actions = Action.ALL_ACTIONS
        first = self.reset()
        self.obs_dim = int(first.numel())
        self._schema = generic_schema(self.obs_dim, "overcooked_feature", family="overcooked")
        self._action_spec = ActionSpec(self.n_agents, self.action_dim, tuple(str(a) for a in self._actions))

    @property
    def schema(self) -> FeatureSchema:
        return self._schema

    @property
    def action_spec(self) -> ActionSpec:
        return self._action_spec

    def _flat_state(self) -> torch.Tensor:
        features = self.env.featurize_state_mdp(self.env.state)
        return torch.as_tensor(np.asarray(features), dtype=torch.float32).reshape(-1)

    def reset(self, seed: int | None = None) -> torch.Tensor:
        self.env.reset()
        return self._flat_state()

    def step(self, action: torch.Tensor):
        idx = action.reshape(self.n_agents, self.action_dim).argmax(dim=-1).tolist()
        joint_action = tuple(self._actions[i] for i in idx)
        _state, reward, done, info = self.env.step(joint_action)
        return self._flat_state(), torch.tensor([float(reward)]), torch.tensor([bool(done)]), info

    def decode(self, obs: torch.Tensor) -> dict[str, object]:
        return self.schema.decode_tensor(obs)

    def encode_action(self, actions) -> torch.Tensor:
        if isinstance(actions, torch.Tensor):
            return actions.float().reshape(-1)
        idx = torch.as_tensor(actions, dtype=torch.long).reshape(self.n_agents)
        return torch.nn.functional.one_hot(idx, self.action_dim).float().reshape(-1)
