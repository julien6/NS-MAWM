"""World-model factory."""

from __future__ import annotations

from wm_lib.models import DeterministicWorldModel, RSSMWorldModel, TransformerWorldModel


def make_world_model(
    name: str,
    obs_dim: int,
    action_dim: int,
    hidden_dim: int = 64,
    output_dim: int | None = None,
    kl_free_nats: float = 0.0,
    kl_balance: float = 0.5,
):
    key = name.lower()
    if key in {"deterministic", "det"}:
        return DeterministicWorldModel(obs_dim, action_dim, hidden_dim=hidden_dim, output_dim=output_dim)
    if key == "rssm":
        return RSSMWorldModel(
            obs_dim,
            action_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            kl_free_nats=kl_free_nats,
            kl_balance=kl_balance,
        )
    if key == "transformer":
        return TransformerWorldModel(obs_dim, action_dim, hidden_dim=hidden_dim, output_dim=output_dim)
    raise ValueError(f"Unknown world model: {name}")
