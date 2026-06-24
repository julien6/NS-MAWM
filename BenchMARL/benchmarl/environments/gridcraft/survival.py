from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TaskConfig:
    width: int = 16
    height: int = 16
    num_agents: int = 1
    view_size: int = 7
    max_steps: int = 500
    seed: int | None = None
