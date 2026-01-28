"""Minimal Overcooked-AI demo for NS-MAWM."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from nsmawm.data.datasets import TransitionsDataset
from nsmawm.envs.overcooked import OvercookedAdapter, OvercookedFeatureConfig, collect_random_transitions
from nsmawm.metrics.rvr import compute_rvr
from nsmawm.models.mawm_backbone import BackboneConfig
from nsmawm.models.nsmawm import NSMAWM
from nsmawm.symbolic.engine import RuleEngine
from nsmawm.symbolic.overcooked_rules import make_overcooked_rules
from nsmawm.training.trainer import fit


def make_env(layout_name: str = "cramped_room", horizon: int = 200):
    try:
        from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
        from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
    except Exception as exc:
        raise ImportError(
            "overcooked_ai_py is required. Install Overcooked-AI and try again."
        ) from exc

    mdp = OvercookedGridworld.from_layout_name(layout_name)
    env = OvercookedEnv.from_mdp(mdp, horizon=horizon)
    return env, mdp


def main() -> None:
    torch.manual_seed(7)

    env, mdp = make_env()
    layout = getattr(mdp, "layout", None)
    grid_shape = tuple(layout.shape[:2]) if layout is not None and hasattr(layout, "shape") else None
    feature_cfg = OvercookedFeatureConfig(
        include_positions=True,
        include_orientation=True,
        include_holding=True,
    )
    adapter = OvercookedAdapter(env, feature_config=feature_cfg, grid_shape=grid_shape)
    print("Feature index table:")
    print(adapter.feature_index_table())

    grid_shape = adapter.infer_grid_shape() or (10, 10)
    stay_idx = adapter.get_stay_action_index()
    feature_indices = adapter.feature_indices()
    rule_engine = RuleEngine(
        make_overcooked_rules(
            stay_action_index=stay_idx,
            grid_shape=grid_shape,
            feature_indices=feature_indices,
        )
    )

    obs, act, next_obs = collect_random_transitions(adapter, n_steps=512)
    dataset = TransitionsDataset(obs, act, next_obs)
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True)

    backbone_cfg = BackboneConfig(
        n_agents=adapter.n_agents,
        n_features=adapter.n_features,
        action_dim=adapter.action_dim,
        latent_dim=64,
        hidden_dim=64,
        encoder_hidden=128,
        decoder_hidden=128,
    )

    model = NSMAWM.from_config(backbone_cfg, rule_engine=rule_engine, strategy="reg+proj", lambda_symb=1.0)
    fit(model, train_loader, max_epochs=3, learning_rate=1e-3)

    model.eval()
    with torch.no_grad():
        output = model.forward(obs[:32], act[:32], apply_projection=True)
        rvr = compute_rvr(output.prediction, output.omega_d, output.mask)
        print(f"RVR (reg+proj) = {rvr.item():.4f}")


if __name__ == "__main__":
    main()
