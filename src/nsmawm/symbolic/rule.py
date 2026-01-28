"""Rule interfaces for symbolic engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch


@dataclass
class RuleContext:
    obs_t: torch.Tensor
    act_t: torch.Tensor
    prev_hidden: Optional[torch.Tensor] = None
    meta: Optional[Dict[str, Any]] = None


@dataclass
class RuleResult:
    values: torch.Tensor
    mask: torch.Tensor


class Rule:
    """Base class for symbolic rules."""

    def apply(self, context: RuleContext) -> RuleResult:  # pragma: no cover - interface
        raise NotImplementedError
