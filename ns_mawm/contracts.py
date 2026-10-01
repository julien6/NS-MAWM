"""Public extension contracts, independently checked with mypy --strict."""
from typing import Mapping, Protocol, Sequence, runtime_checkable
from torch import Tensor

@runtime_checkable
class Backbone(Protocol):
    def init_state(self, prefix: Sequence[object] = ()) -> object: ...
    def step(self, state: object, obs: Tensor, joint_action: Tensor,
             extra: tuple[Tensor, Tensor] | None = None) -> tuple[Tensor, object]: ...

@runtime_checkable
class ControlHeads(Protocol):
    def heads(self, state: object) -> tuple[Tensor, Tensor]: ...

class Environment(Protocol):
    agents: int
    action_size: int
    metadata: Mapping[str, object]
    def reset(self, seed: int) -> Tensor: ...
    def step(self, actions: Sequence[int]) -> tuple[Tensor, float, bool, bool]: ...
    def close(self) -> None: ...
