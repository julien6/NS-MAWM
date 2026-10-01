"""Typed public adapters for dynamically registered implementations."""
from typing import Any, Callable, Mapping, Sequence, cast
from torch import Tensor
from .contracts import Backbone, Environment


class RegisteredBackbone:
    """Checks a plugin's outputs at the dynamic-to-typed boundary."""
    def __init__(self, implementation: Backbone) -> None:
        self.implementation = implementation

    def init_state(self, prefix: Sequence[object] = ()) -> object:
        return self.implementation.init_state(prefix)

    def step(self, state: object, obs: Tensor, joint_action: Tensor,
             extra: tuple[Tensor, Tensor] | None = None) -> tuple[Tensor, object]:
        output, next_state = self.implementation.step(state, obs, joint_action, extra)
        if output.shape != obs.shape:
            raise ValueError('Backbone output does not match the observation schema')
        return output, next_state


class RegisteredEnvironment:
    def __init__(self, implementation: Environment) -> None:
        self.implementation = implementation
        self.agents = implementation.agents
        self.action_size = implementation.action_size
        self.metadata = implementation.metadata

    def reset(self, seed: int) -> Tensor:
        return self.implementation.reset(seed)

    def step(self, actions: Sequence[int]) -> tuple[Tensor, float, bool, bool]:
        if len(actions) != self.agents or any(a < 0 or a >= self.action_size for a in actions):
            raise ValueError('Invalid joint action')
        return self.implementation.step(actions)

    def close(self) -> None:
        self.implementation.close()
