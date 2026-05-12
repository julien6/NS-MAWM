"""Small local predator-prey adapter used when no external package is installed."""

from __future__ import annotations

import torch

from env_adapters.flat import ActionSpec, VariantSpec, default_variants, resolve_variant_id
from ns_mawm.features import FeatureSchema, FeatureSpec, FeatureType


class PredatorPreyAdapter:
    def __init__(self, n_predators: int = 2, width: int = 7, height: int = 7, max_steps: int = 50):
        self.n_agents = n_predators
        self.action_dim = 5
        self.variant_id = "predator_prey_sv"
        self.width = width
        self.height = height
        self.max_steps = max_steps
        self.obs_dim = 4 * n_predators + 5
        self.steps = 0
        self.generator = torch.Generator()
        self.predators = torch.zeros(n_predators, 2)
        self.predator_delta = torch.zeros(n_predators, 2)
        self.prey = torch.zeros(2)
        self.prey_active = torch.tensor(1.0)
        self.capture_in_range = torch.tensor(0.0)
        self.boundary_contact = torch.tensor(0.0)
        self._schema = FeatureSchema.from_specs(
            [
                *[
                    FeatureSpec(f"predator_{i}.{axis}", FeatureType.INTEGER, owner=f"predator_{i}", family="position", tolerance=0.5)
                    for i in range(n_predators)
                    for axis in ("x", "y")
                ],
                *[
                    FeatureSpec(f"predator_{i}.d{axis}", FeatureType.INTEGER, owner=f"predator_{i}", family="velocity", tolerance=0.5)
                    for i in range(n_predators)
                    for axis in ("x", "y")
                ],
                FeatureSpec("prey.x", FeatureType.INTEGER, owner="prey", family="position", tolerance=0.5),
                FeatureSpec("prey.y", FeatureType.INTEGER, owner="prey", family="position", tolerance=0.5),
                FeatureSpec("prey.active", FeatureType.BINARY, owner="prey", family="activity"),
                FeatureSpec("capture.in_range", FeatureType.BINARY, owner="global", family="capture"),
                FeatureSpec("boundary.contact", FeatureType.BINARY, owner="global", family="boundary"),
            ]
        )
        self._action_spec = ActionSpec(n_predators, self.action_dim, ("stay", "up", "down", "left", "right"))

    @property
    def schema(self) -> FeatureSchema:
        return self._schema

    @property
    def action_spec(self) -> ActionSpec:
        return self._action_spec

    def make_variants(self, split: str) -> tuple[VariantSpec, ...]:
        return default_variants("predator_prey", split)

    def reset(self, seed: int | None = None, variant: str | VariantSpec | None = None) -> torch.Tensor:
        variant_id, seed_offset = resolve_variant_id("predator_prey_sv", variant)
        self.variant_id = variant_id
        if seed is not None:
            self.generator.manual_seed(seed + seed_offset)
        self.steps = 0
        self.predators = torch.randint(self.width, (self.n_agents, 2), generator=self.generator).float()
        self.predator_delta = torch.zeros(self.n_agents, 2)
        self.prey = torch.randint(self.width, (2,), generator=self.generator).float()
        self.prey_active = torch.tensor(1.0)
        self.boundary_contact = torch.tensor(0.0)
        self._update_capture_features()
        return self._obs()

    def _obs(self) -> torch.Tensor:
        return torch.cat(
            [
                self.predators.reshape(-1),
                self.predator_delta.reshape(-1),
                self.prey,
                self.prey_active.reshape(1),
                self.capture_in_range.reshape(1),
                self.boundary_contact.reshape(1),
            ]
        )

    def _update_capture_features(self) -> bool:
        dist = torch.linalg.vector_norm(self.predators - self.prey, dim=-1).min()
        caught = bool(dist.item() <= 1.0 and bool(self.prey_active.item()))
        self.capture_in_range = torch.tensor(float(dist.item() <= 1.0))
        if caught:
            self.prey_active = torch.tensor(0.0)
        return caught

    def step(self, action: torch.Tensor):
        idx = action.reshape(self.n_agents, self.action_dim).argmax(dim=-1)
        delta = torch.zeros_like(self.predators)
        delta[idx == 1, 1] = -1
        delta[idx == 2, 1] = 1
        delta[idx == 3, 0] = -1
        delta[idx == 4, 0] = 1
        unclipped = self.predators + delta
        next_predators = unclipped.clone()
        next_predators[:, 0] = next_predators[:, 0].clamp(0, self.width - 1)
        next_predators[:, 1] = next_predators[:, 1].clamp(0, self.height - 1)
        self.boundary_contact = torch.tensor(float(not torch.equal(unclipped, next_predators)))
        self.predator_delta = next_predators - self.predators
        self.predators = next_predators
        self.steps += 1
        dist = torch.linalg.vector_norm(self.predators - self.prey, dim=-1).min()
        caught = self._update_capture_features()
        done = caught or self.steps >= self.max_steps
        reward = 1.0 if caught else -0.01 * float(dist.item())
        info = {
            "distance": float(dist.item()),
            "caught": caught,
            "prey_active": bool(self.prey_active.item()),
            "boundary_contact": bool(self.boundary_contact.item()),
        }
        return self._obs(), torch.tensor([reward]), torch.tensor([done]), info

    def decode(self, obs: torch.Tensor) -> dict[str, object]:
        return self.schema.decode_tensor(obs)

    def encode_action(self, actions) -> torch.Tensor:
        if isinstance(actions, torch.Tensor):
            return actions.float().reshape(-1)
        idx = torch.as_tensor(actions, dtype=torch.long).reshape(self.n_agents)
        return torch.nn.functional.one_hot(idx, self.action_dim).float().reshape(-1)
