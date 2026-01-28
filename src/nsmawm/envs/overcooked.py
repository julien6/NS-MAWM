"""Overcooked-AI integration utilities (minimal)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Tuple

import random

import torch


@dataclass
class OvercookedFeatureConfig:
    include_positions: bool = True
    position_scale: float = 1.0
    include_orientation: bool = False
    orientation_one_hot: bool = True
    orientation_vocab: Tuple[str, ...] = ("NORTH", "SOUTH", "EAST", "WEST")
    include_holding: bool = False
    holding_vocab: Tuple[str, ...] = ("onion", "tomato", "dish", "soup")
    include_unknown_holding: bool = True


class OvercookedAdapter:
    """Minimal adapter that extracts low-dimensional features from Overcooked state."""

    def __init__(
        self,
        env,
        feature_config: Optional[OvercookedFeatureConfig] = None,
        grid_shape: Optional[Tuple[int, int]] = None,
    ):
        self.env = env
        self.feature_config = feature_config or OvercookedFeatureConfig()
        self.grid_shape = grid_shape
        self._action_list = self._infer_action_list()
        self._feature_indices = self._build_feature_indices()

    @property
    def n_agents(self) -> int:
        return len(self._extract_positions(self._get_state()))

    @property
    def n_features(self) -> int:
        dim = 0
        if self.feature_config.include_positions:
            dim += 2
        if self.feature_config.include_orientation:
            dim += len(self.feature_config.orientation_vocab) if self.feature_config.orientation_one_hot else 1
        if self.feature_config.include_holding:
            dim += len(self.feature_config.holding_vocab)
            if self.feature_config.include_unknown_holding:
                dim += 1
            dim += 1  # "none" slot
        return dim

    def feature_indices(self) -> dict:
        """Return a mapping from feature names to indices for rule authoring."""
        return dict(self._feature_indices)

    def feature_slices(self) -> dict:
        """Return contiguous slices for feature blocks (positions/orientation/holding)."""
        indices = self._feature_indices
        slices = {}
        if "positions" in indices:
            x_idx = indices["positions"]["x"]
            slices["positions"] = slice(x_idx, x_idx + 2)
        if "orientation" in indices:
            orient = indices["orientation"]
            if isinstance(orient, dict):
                start = min(orient.values())
                slices["orientation"] = slice(start, start + len(orient))
            else:
                slices["orientation"] = slice(orient, orient + 1)
        if "holding" in indices:
            hold = indices["holding"]
            start = min(hold.values())
            slices["holding"] = slice(start, start + len(hold))
        return slices

    def feature_index_table(self) -> str:
        """Return a human-readable table of feature indices."""
        indices = self._feature_indices
        lines = []
        if "positions" in indices:
            pos = indices["positions"]
            lines.append(f"positions:   x={pos.get('x')}, y={pos.get('y')}")
        if "orientation" in indices:
            orient = indices["orientation"]
            if isinstance(orient, dict):
                parts = ", ".join(f"{k}={v}" for k, v in orient.items())
                lines.append(f"orientation: {parts}")
            else:
                lines.append(f"orientation: idx={orient}")
        if "holding" in indices:
            hold = indices["holding"]
            parts = ", ".join(f"{k}={v}" for k, v in hold.items())
            lines.append(f"holding:     {parts}")
        return "\n".join(lines)

    @property
    def action_dim(self) -> int:
        return len(self._action_list)

    def reset(self):
        state = self.env.reset()
        return self.featurize_state(state), state

    def step(self, joint_action):
        next_state, reward, done, info = self.env.step(joint_action)
        return self.featurize_state(next_state), next_state, reward, done, info

    def featurize_state(self, state) -> torch.Tensor:
        if not (
            self.feature_config.include_positions
            or self.feature_config.include_orientation
            or self.feature_config.include_holding
        ):
            raise ValueError("No features enabled in OvercookedFeatureConfig")
        positions = self._extract_positions(state)
        orientations = self._extract_orientations(state)
        holdings = self._extract_holdings(state)

        feats = []
        for idx, (x, y) in enumerate(positions):
            row = []
            if self.feature_config.include_positions:
                row.extend(
                    [
                        float(x) * self.feature_config.position_scale,
                        float(y) * self.feature_config.position_scale,
                    ]
                )
            if self.feature_config.include_orientation:
                orient = orientations[idx] if idx < len(orientations) else None
                row.extend(self._orientation_features(orient))
            if self.feature_config.include_holding:
                holding = holdings[idx] if idx < len(holdings) else None
                row.extend(self._holding_features(holding))
            feats.append(row)
        return torch.tensor(feats, dtype=torch.float32)

    def encode_action(self, joint_action) -> torch.Tensor:
        action_vecs = []
        for act in joint_action:
            idx = self._action_to_index(act)
            one_hot = torch.zeros(self.action_dim, dtype=torch.float32)
            one_hot[idx] = 1.0
            action_vecs.append(one_hot)
        return torch.stack(action_vecs, dim=0)

    def sample_joint_action(self) -> Tuple:
        return tuple(random.choice(self._action_list) for _ in range(self.n_agents))

    def get_stay_action_index(self, default: int = 4) -> int:
        try:
            from overcooked_ai_py.mdp.actions import Action

            if hasattr(Action, "STAY"):
                return self._action_to_index(Action.STAY)
        except Exception:
            pass
        return default

    def infer_grid_shape(self) -> Optional[Tuple[int, int]]:
        if self.grid_shape is not None:
            return self.grid_shape
        mdp = getattr(self.env, "mdp", None)
        layout = getattr(mdp, "layout", None)
        if layout is not None and hasattr(layout, "shape"):
            return tuple(layout.shape[:2])
        return None

    def _get_state(self):
        if hasattr(self.env, "state"):
            return self.env.state
        return self.env.reset()

    def _extract_positions(self, state) -> List[Tuple[int, int]]:
        if hasattr(state, "players"):
            positions = []
            for player in state.players:
                if hasattr(player, "position"):
                    positions.append(tuple(player.position))
                elif hasattr(player, "pos"):
                    positions.append(tuple(player.pos))
                else:
                    raise ValueError("Unsupported player position attribute on Overcooked state")
            return positions
        if hasattr(state, "player_positions"):
            return [tuple(pos) for pos in state.player_positions]
        raise ValueError("Unable to extract player positions from Overcooked state")

    def _extract_orientations(self, state) -> List[Optional[object]]:
        if hasattr(state, "players"):
            orientations = []
            for player in state.players:
                if hasattr(player, "orientation"):
                    orientations.append(player.orientation)
                elif hasattr(player, "dir"):
                    orientations.append(player.dir)
                else:
                    orientations.append(None)
            return orientations
        return []

    def _extract_holdings(self, state) -> List[Optional[str]]:
        if hasattr(state, "players"):
            holdings = []
            for player in state.players:
                held = None
                if hasattr(player, "held_object"):
                    held = player.held_object
                elif hasattr(player, "held"):
                    held = player.held
                if held is None:
                    holdings.append(None)
                    continue
                name = None
                for attr in ("name", "obj_type", "type"):
                    if hasattr(held, attr):
                        name = getattr(held, attr)
                        break
                if name is None:
                    holdings.append(None)
                else:
                    holdings.append(str(name))
            return holdings
        return []

    def _orientation_features(self, orientation) -> List[float]:
        if self.feature_config.orientation_one_hot:
            vec = [0.0 for _ in self.feature_config.orientation_vocab]
            idx = self._orientation_to_index(orientation)
            if idx is not None:
                vec[idx] = 1.0
            return vec
        idx = self._orientation_to_index(orientation)
        return [float(idx) if idx is not None else 0.0]

    def _orientation_to_index(self, orientation) -> Optional[int]:
        if orientation is None:
            return None
        if isinstance(orientation, int):
            if 0 <= orientation < len(self.feature_config.orientation_vocab):
                return orientation
            return None
        if isinstance(orientation, str):
            upper = orientation.upper()
            try:
                return self.feature_config.orientation_vocab.index(upper)
            except ValueError:
                return None
        if isinstance(orientation, (tuple, list)) and len(orientation) == 2:
            vec_map = {(0, -1): "NORTH", (0, 1): "SOUTH", (1, 0): "EAST", (-1, 0): "WEST"}
            name = vec_map.get(tuple(orientation))
            if name is None:
                return None
            try:
                return self.feature_config.orientation_vocab.index(name)
            except ValueError:
                return None
        if hasattr(orientation, "name"):
            try:
                return self.feature_config.orientation_vocab.index(str(orientation.name).upper())
            except ValueError:
                return None
        return None

    def _holding_features(self, holding: Optional[str]) -> List[float]:
        vocab = self.feature_config.holding_vocab
        include_unknown = self.feature_config.include_unknown_holding
        vec_len = len(vocab) + 1 + (1 if include_unknown else 0)
        vec = [0.0 for _ in range(vec_len)]
        if holding is None:
            vec[0] = 1.0
            return vec
        holding_name = str(holding).lower()
        try:
            idx = vocab.index(holding_name)
            vec[1 + idx] = 1.0
        except ValueError:
            if include_unknown:
                vec[-1] = 1.0
        return vec

    def _build_feature_indices(self) -> dict:
        indices = {}
        offset = 0
        if self.feature_config.include_positions:
            indices["positions"] = {"x": offset, "y": offset + 1}
            offset += 2
        if self.feature_config.include_orientation:
            if self.feature_config.orientation_one_hot:
                orient = {}
                for idx, name in enumerate(self.feature_config.orientation_vocab):
                    orient[name] = offset + idx
                indices["orientation"] = orient
                offset += len(self.feature_config.orientation_vocab)
            else:
                indices["orientation"] = offset
                offset += 1
        if self.feature_config.include_holding:
            hold = {"none": offset}
            offset += 1
            for name in self.feature_config.holding_vocab:
                hold[name] = offset
                offset += 1
            if self.feature_config.include_unknown_holding:
                hold["unknown"] = offset
                offset += 1
            indices["holding"] = hold
        return indices

    def _infer_action_list(self) -> List:
        if hasattr(self.env, "action_space"):
            try:
                sample = self.env.action_space.sample()
                if isinstance(sample, (list, tuple)):
                    return list(range(len(sample)))
            except Exception:
                pass
        try:
            from overcooked_ai_py.mdp.actions import Action

            if hasattr(Action, "ALL_ACTIONS"):
                return list(Action.ALL_ACTIONS)
        except Exception:
            pass
        return list(range(6))

    def _action_to_index(self, action) -> int:
        try:
            return self._action_list.index(action)
        except ValueError:
            return int(action)


def collect_random_transitions(
    adapter: OvercookedAdapter,
    n_steps: int = 256,
    seed: int = 7,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    random.seed(seed)
    obs_list = []
    act_list = []
    next_list = []

    obs_t, _ = adapter.reset()
    for _ in range(n_steps):
        joint_action = adapter.sample_joint_action()
        act_vec = adapter.encode_action(joint_action)
        next_obs, _, _, done, _ = adapter.step(joint_action)

        obs_list.append(obs_t)
        act_list.append(act_vec)
        next_list.append(next_obs)

        obs_t = next_obs
        if done:
            obs_t, _ = adapter.reset()

    obs = torch.stack(obs_list, dim=0)
    act = torch.stack(act_list, dim=0)
    next_obs = torch.stack(next_list, dim=0)
    return obs, act, next_obs
