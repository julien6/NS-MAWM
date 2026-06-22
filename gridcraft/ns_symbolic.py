import numpy as np

from exp_config import (
  BLOCK_CLASSES,
  ENTITY_CLASSES,
  GRID_CELLS,
  GRID_FEATURES,
  GRIDCRAFT_VIEW_SIZE,
  SELF_FEATURES,
  TERRAIN_CLASSES,
)
from vae.vae import vector_to_tabular


NS_VARIANTS = ("neural", "regularization", "projection", "residual")


def symbolic_transition_from_vector(obs_vector, action, coverage=1.0):
  return symbolic_transition(vector_to_tabular(obs_vector), action, coverage=coverage)


def symbolic_transition(obs, action, coverage=1.0):
  grid = np.asarray(obs["grid"], dtype=np.int8)
  symbolic = {
    "grid": grid.copy(),
    "self": np.asarray(obs["self"], dtype=np.int16).copy(),
  }
  mask = {
    "grid": np.zeros_like(grid, dtype=np.bool_),
    "self": np.zeros((SELF_FEATURES,), dtype=np.bool_),
  }

  shift = _agent_shift(grid, int(action))
  if shift is not None:
    _apply_shifted_static_planes(grid, symbolic["grid"], mask["grid"], shift)

  center = GRIDCRAFT_VIEW_SIZE // 2
  symbolic["grid"][2, center, center] = 1
  mask["grid"][2, center, center] = True
  apply_coverage(mask, coverage)
  return symbolic, mask


def apply_symbolic_projection(predicted_obs, current_obs, action, variant, coverage=1.0):
  if variant == "neural" or variant == "regularization":
    return predicted_obs, None
  symbolic, mask = symbolic_transition(current_obs, action, coverage=coverage)
  projected = {
    "grid": np.asarray(predicted_obs["grid"], dtype=np.int8).copy(),
    "self": np.asarray(predicted_obs["self"], dtype=np.int16).copy(),
  }
  projected["grid"][mask["grid"]] = symbolic["grid"][mask["grid"]]
  projected["self"][mask["self"]] = symbolic["self"][mask["self"]]
  return projected, (symbolic, mask)


def compare_with_symbolic(predicted_obs, current_obs, action, coverage=1.0):
  symbolic, mask = symbolic_transition(current_obs, action, coverage=coverage)
  grid_mask = mask["grid"]
  self_mask = mask["self"]
  total = int(np.sum(grid_mask) + np.sum(self_mask))
  if total == 0:
    return {
      "rvr": 0.0,
      "determinable_mismatch": 0.0,
      "undeterminable_mismatch": 0.0,
      "determinable_count": 0.0,
    }

  predicted_grid = np.asarray(predicted_obs["grid"], dtype=np.int16)
  predicted_self = np.asarray(predicted_obs["self"], dtype=np.float32)
  symbolic_grid = np.asarray(symbolic["grid"], dtype=np.int16)
  symbolic_self = np.asarray(symbolic["self"], dtype=np.float32)
  grid_violations = int(np.sum(predicted_grid[grid_mask] != symbolic_grid[grid_mask]))
  if np.any(self_mask):
    self_violations = int(np.sum(np.abs(predicted_self[self_mask] - symbolic_self[self_mask]) > 0.5))
  else:
    self_violations = 0
  rvr = float((grid_violations + self_violations) / total)

  predicted_vec = tabular_to_vector(predicted_obs)
  symbolic_vec = tabular_to_vector(symbolic)
  mask_vec = tabular_mask_to_vector_mask(mask)
  det = float(np.mean(np.abs(predicted_vec[mask_vec] - symbolic_vec[mask_vec]))) if np.any(mask_vec) else 0.0
  undet = float(np.mean(np.abs(predicted_vec[~mask_vec] - symbolic_vec[~mask_vec]))) if np.any(~mask_vec) else 0.0
  return {
    "rvr": rvr,
    "determinable_mismatch": det,
    "undeterminable_mismatch": undet,
    "determinable_count": float(total),
  }


def symbolic_batch_targets(obs_batch, action_batch, coverage=1.0):
  batch, seq_len = action_batch.shape
  targets = np.zeros((batch, seq_len, GRID_FEATURES + SELF_FEATURES), dtype=np.float32)
  masks = np.zeros_like(targets, dtype=np.float32)
  for b in range(batch):
    for t in range(seq_len):
      symbolic, mask = symbolic_transition_from_vector(obs_batch[b, t], int(action_batch[b, t]), coverage=coverage)
      targets[b, t] = tabular_to_vector(symbolic)
      masks[b, t] = tabular_mask_to_vector_mask(mask).astype(np.float32)
  return targets, masks


def tabular_to_vector(obs):
  grid = np.asarray(obs["grid"], dtype=np.int64)
  terrain = _one_hot(grid[0], TERRAIN_CLASSES)
  blocks = _one_hot(grid[1], BLOCK_CLASSES)
  entities = _one_hot(grid[2], ENTITY_CLASSES)
  self_vec = np.asarray(obs["self"], dtype=np.float32)
  normalized = np.zeros((SELF_FEATURES,), dtype=np.float32)
  normalized[0:2] = self_vec[0:2] / 20.0
  normalized[2:] = np.clip(self_vec[2:], 0, 10) / 10.0
  return np.concatenate([terrain, blocks, entities, normalized]).astype(np.float32)


def tabular_mask_to_vector_mask(mask):
  grid_mask = np.asarray(mask["grid"], dtype=np.bool_)
  terrain = np.repeat(grid_mask[0].reshape(-1), TERRAIN_CLASSES)
  blocks = np.repeat(grid_mask[1].reshape(-1), BLOCK_CLASSES)
  entities = np.repeat(grid_mask[2].reshape(-1), ENTITY_CLASSES)
  self_mask = np.asarray(mask["self"], dtype=np.bool_)
  return np.concatenate([terrain, blocks, entities, self_mask])


def _one_hot(values, depth):
  values = np.asarray(values, dtype=np.int64).reshape(-1)
  values = np.clip(values, 0, depth - 1)
  return np.eye(depth, dtype=np.float32)[values].reshape(-1)


def apply_coverage(mask, coverage):
  coverage = float(np.clip(coverage, 0.0, 1.0))
  if coverage >= 1.0:
    return
  if coverage <= 0.0:
    mask["grid"][:] = False
    mask["self"][:] = False
    return
  grid = mask["grid"]
  true_positions = np.argwhere(grid)
  for channel, y, x in true_positions:
    idx = int(channel) * GRID_CELLS + int(y) * GRIDCRAFT_VIEW_SIZE + int(x)
    score = ((idx * 1103515245 + 12345) % 10000) / 10000.0
    if score >= coverage:
      grid[channel, y, x] = False
  for idx, active in enumerate(mask["self"]):
    if active:
      score = (((GRID_CELLS * 3 + idx) * 1103515245 + 12345) % 10000) / 10000.0
      if score >= coverage:
        mask["self"][idx] = False


def _agent_shift(grid, action):
  if action == 0:
    return (0, 0)
  deltas = {
    1: (0, -1),
    2: (0, 1),
    3: (-1, 0),
    4: (1, 0),
  }
  if action not in deltas:
    return None
  dx, dy = deltas[action]
  center = GRIDCRAFT_VIEW_SIZE // 2
  target_x = center + dx
  target_y = center + dy
  if not (0 <= target_x < GRIDCRAFT_VIEW_SIZE and 0 <= target_y < GRIDCRAFT_VIEW_SIZE):
    return (0, 0)
  terrain = int(grid[0, target_y, target_x])
  block = int(grid[1, target_y, target_x])
  entity = int(grid[2, target_y, target_x])
  blocked = terrain == 1 or block in (1, 2) or entity in (1, 2)
  return (0, 0) if blocked else (dx, dy)


def _apply_shifted_static_planes(source, dest, mask, shift):
  dx, dy = shift
  size = GRIDCRAFT_VIEW_SIZE
  for gy in range(size):
    for gx in range(size):
      sy = gy + dy
      sx = gx + dx
      if 0 <= sy < size and 0 <= sx < size:
        dest[0, gy, gx] = source[0, sy, sx]
        dest[1, gy, gx] = source[1, sy, sx]
        mask[0, gy, gx] = True
        mask[1, gy, gx] = True
