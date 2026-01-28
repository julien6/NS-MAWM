from __future__ import annotations

import torch

from nsmawm.envs.overcooked import OvercookedAdapter, OvercookedFeatureConfig
from nsmawm.symbolic.overcooked_rules import StayPutRule


class MockHeld:
    def __init__(self, name: str):
        self.name = name


class MockPlayer:
    def __init__(self, position, orientation, held_object=None):
        self.position = position
        self.orientation = orientation
        self.held_object = held_object


class MockState:
    def __init__(self, players):
        self.players = players


class MockEnv:
    def __init__(self, state):
        self.state = state

    def reset(self):
        return self.state


def test_overcooked_featurizer_positions_orientation_holding():
    players = [
        MockPlayer(position=(1, 2), orientation="NORTH", held_object=MockHeld("onion")),
        MockPlayer(position=(3, 4), orientation="WEST", held_object=None),
    ]
    state = MockState(players)
    env = MockEnv(state)

    feature_cfg = OvercookedFeatureConfig(
        include_positions=True,
        include_orientation=True,
        include_holding=True,
        orientation_vocab=("NORTH", "SOUTH", "EAST", "WEST"),
        holding_vocab=("onion", "tomato", "dish", "soup"),
        include_unknown_holding=True,
    )
    adapter = OvercookedAdapter(env, feature_config=feature_cfg)

    feats = adapter.featurize_state(state)
    assert feats.shape == (2, adapter.n_features)

    # Layout: [x, y] + orientation(4) + holding(none + 4 + unknown)
    # Agent 0: position (1,2), orientation NORTH, holding onion.
    assert torch.allclose(feats[0, :2], torch.tensor([1.0, 2.0]))
    assert torch.allclose(feats[0, 2:6], torch.tensor([1.0, 0.0, 0.0, 0.0]))
    # Holding: none=0, onion=1
    assert feats[0, 6] == 0.0
    assert feats[0, 7] == 1.0

    # Agent 1: position (3,4), orientation WEST, holding none.
    assert torch.allclose(feats[1, :2], torch.tensor([3.0, 4.0]))
    assert torch.allclose(feats[1, 2:6], torch.tensor([0.0, 0.0, 0.0, 1.0]))
    assert feats[1, 6] == 1.0


def test_overcooked_feature_indices():
    players = [
        MockPlayer(position=(0, 0), orientation="EAST", held_object=MockHeld("dish")),
    ]
    state = MockState(players)
    env = MockEnv(state)

    feature_cfg = OvercookedFeatureConfig(
        include_positions=True,
        include_orientation=True,
        include_holding=True,
        orientation_vocab=("NORTH", "SOUTH", "EAST", "WEST"),
        holding_vocab=("onion", "tomato", "dish", "soup"),
        include_unknown_holding=True,
    )
    adapter = OvercookedAdapter(env, feature_config=feature_cfg)
    indices = adapter.feature_indices()

    assert indices["positions"]["x"] == 0
    assert indices["positions"]["y"] == 1
    assert indices["orientation"]["EAST"] == 4
    assert indices["holding"]["none"] == 6
    assert indices["holding"]["dish"] == 9
    assert indices["holding"]["unknown"] == 11


def test_overcooked_rule_position_indices_resolution():
    players = [MockPlayer(position=(0, 0), orientation="NORTH", held_object=None)]
    state = MockState(players)
    env = MockEnv(state)
    feature_cfg = OvercookedFeatureConfig(
        include_positions=True,
        include_orientation=True,
        include_holding=True,
        orientation_vocab=("NORTH", "SOUTH", "EAST", "WEST"),
        holding_vocab=("onion", "tomato", "dish", "soup"),
        include_unknown_holding=True,
    )
    adapter = OvercookedAdapter(env, feature_config=feature_cfg)
    indices = adapter.feature_indices()
    stay_idx = 0

    rule = StayPutRule(stay_action_index=stay_idx, feature_indices=indices)

    obs = torch.zeros(1, 1, adapter.n_features)
    obs[..., indices["positions"]["x"]] = 5.0
    obs[..., indices["positions"]["y"]] = 6.0
    act = torch.zeros(1, 1, adapter.action_dim)
    act[..., stay_idx] = 1.0

    context = type("C", (), {"obs_t": obs, "act_t": act})()
    result = rule.apply(context)
    assert result.mask[..., indices["positions"]["x"]].all()
    assert result.mask[..., indices["positions"]["y"]].all()
    assert result.values[..., indices["positions"]["x"]].item() == 5.0
    assert result.values[..., indices["positions"]["y"]].item() == 6.0


def test_overcooked_position_bounds_rule_indices_resolution():
    from nsmawm.symbolic.overcooked_rules import PositionBoundsRule

    players = [MockPlayer(position=(0, 0), orientation="NORTH", held_object=None)]
    state = MockState(players)
    env = MockEnv(state)
    feature_cfg = OvercookedFeatureConfig(
        include_positions=True,
        include_orientation=True,
        include_holding=True,
        orientation_vocab=("NORTH", "SOUTH", "EAST", "WEST"),
        holding_vocab=("onion", "tomato", "dish", "soup"),
        include_unknown_holding=True,
    )
    adapter = OvercookedAdapter(env, feature_config=feature_cfg)
    indices = adapter.feature_indices()

    rule = PositionBoundsRule(grid_shape=(3, 3), feature_indices=indices)

    obs = torch.zeros(1, 1, adapter.n_features)
    obs[..., indices["positions"]["x"]] = 5.0
    obs[..., indices["positions"]["y"]] = -2.0
    act = torch.zeros(1, 1, adapter.action_dim)

    context = type("C", (), {"obs_t": obs, "act_t": act})()
    result = rule.apply(context)
    assert result.values[..., indices["positions"]["x"]].item() == 2.0
    assert result.values[..., indices["positions"]["y"]].item() == 0.0
    assert result.mask[..., indices["positions"]["x"]].all()
    assert result.mask[..., indices["positions"]["y"]].all()
