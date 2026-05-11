"""Small local predator-prey adapter used when no external package is installed."""

from __future__ import annotations

import torch

from env_adapters.flat import ActionSpec
from ns_mawm.features import FeatureSchema, FeatureSpec, FeatureType


class PredatorPreyAdapter:
    def __init__(self, n_predators: int = 2, width: int = 7, height: int = 7, max_steps: int = 50):
        self.n_agents = n_predators
        self.action_dim = 5
        self.variant_id = "predator_prey_sv"
        self.width = width
        self.height = height
        self.max_steps = max_steps
        self.obs_dim = 2 * n_predators + 2
        self.steps = 0
        self.generator = torch.Generator()
        self.predators = torch.zeros(n_predators, 2)
        self.prey = torch.zeros(2)
        self._schema = FeatureSchema.from_specs(
            [
                *[
                    FeatureSpec(f"predator_{i}.{axis}", FeatureType.INTEGER, owner=f"predator_{i}", family="position", tolerance=0.5)
                    for i in range(n_predators)
                    for axis in ("x", "y")
                ],
                FeatureSpec("prey.x", FeatureType.INTEGER, owner="prey", family="position", tolerance=0.5),
                FeatureSpec("prey.y", FeatureType.INTEGER, owner="prey", family="position", tolerance=0.5),
            ]
        )
        self._action_spec = ActionSpec(n_predators, self.action_dim, ("stay", "up", "down", "left", "right"))

    @property
    def schema(self) -> FeatureSchema:
        return self._schema

    @property
    def action_spec(self) -> ActionSpec:
        return self._action_spec

    def reset(self, seed: int | None = None) -> torch.Tensor:
        if seed is not None:
            self.generator.manual_seed(seed)
        self.steps = 0
        self.predators = torch.randint(self.width, (self.n_agents, 2), generator=self.generator).float()
        self.prey = torch.randint(self.width, (2,), generator=self.generator).float()
        return self._obs()

    def _obs(self) -> torch.Tensor:
        return torch.cat([self.predators.reshape(-1), self.prey])

    def step(self, action: torch.Tensor):
        idx = action.reshape(self.n_agents, self.action_dim).argmax(dim=-1)
        delta = torch.zeros_like(self.predators)
        delta[idx == 1, 1] = -1
        delta[idx == 2, 1] = 1
        delta[idx == 3, 0] = -1
        delta[idx == 4, 0] = 1
        self.predators = (self.predators + delta).clamp(0, self.width - 1)
        self.steps += 1
        dist = torch.linalg.vector_norm(self.predators - self.prey, dim=-1).min()
        caught = bool(dist.item() <= 1.0)
        done = caught or self.steps >= self.max_steps
        reward = 1.0 if caught else -0.01 * float(dist.item())
        return self._obs(), torch.tensor([reward]), torch.tensor([done]), {}

    def decode(self, obs: torch.Tensor) -> dict[str, object]:
        return self.schema.decode_tensor(obs)

    def encode_action(self, actions) -> torch.Tensor:
        if isinstance(actions, torch.Tensor):
            return actions.float().reshape(-1)
        idx = torch.as_tensor(actions, dtype=torch.long).reshape(self.n_agents)
        return torch.nn.functional.one_hot(idx, self.action_dim).float().reshape(-1)
