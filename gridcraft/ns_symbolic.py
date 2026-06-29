from __future__ import annotations

from copy import deepcopy

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


NS_VARIANTS = ("neural", "regularization", "projection", "residual")

TERRAIN_WATER = 1
BLOCK_EMPTY = 0
BLOCK_TREE = 1
BLOCK_STONE = 2
ENTITY_NONE = 0
ENTITY_AGENT = 1
ENTITY_MOB = 2
ENTITY_ITEM = 3

ITEM_WOOD = 0
ITEM_PLANK = 1
ITEM_STICK = 2
ITEM_STONE = 3
ITEM_WOOD_SWORD = 4
ITEM_STONE_SWORD = 5
ITEM_WOOD_PICKAXE = 6
ITEM_STONE_PICKAXE = 7
ITEM_APPLE = 8

ACTION_STAY = 0
ACTION_MOVE_N = 1
ACTION_MOVE_S = 2
ACTION_MOVE_W = 3
ACTION_MOVE_E = 4
ACTION_HARVEST = 5
ACTION_PICKUP = 6
ACTION_ATTACK = 7
ACTION_EAT = 8

MOVE_DELTAS = {
  ACTION_STAY: (0, 0),
  ACTION_MOVE_N: (0, -1),
  ACTION_MOVE_S: (0, 1),
  ACTION_MOVE_W: (-1, 0),
  ACTION_MOVE_E: (1, 0),
}

CRAFT_RECIPES = {
  9: (((ITEM_WOOD, 1),), (ITEM_PLANK, 2), "PSTR_INDIV_CRAFT_PLANK"),
  10: (((ITEM_PLANK, 2),), (ITEM_STICK, 4), "PSTR_INDIV_CRAFT_STICK"),
  11: (((ITEM_STICK, 1), (ITEM_PLANK, 1)), (ITEM_WOOD_SWORD, 1), "PSTR_INDIV_CRAFT_TOOLS"),
  12: (((ITEM_STICK, 1), (ITEM_STONE, 1)), (ITEM_STONE_SWORD, 1), "PSTR_INDIV_CRAFT_TOOLS"),
  13: (((ITEM_STICK, 1), (ITEM_PLANK, 1)), (ITEM_WOOD_PICKAXE, 1), "PSTR_INDIV_CRAFT_TOOLS"),
  14: (((ITEM_STICK, 1), (ITEM_STONE, 1)), (ITEM_STONE_PICKAXE, 1), "PSTR_INDIV_CRAFT_TOOLS"),
}


def symbolic_transition_from_vector(obs_vector, action, coverage=1.0, enabled_pstr_rules=None):
  return symbolic_transition(vector_to_tabular(obs_vector), action, coverage=coverage, enabled_pstr_rules=enabled_pstr_rules)


def symbolic_transition(obs, action, coverage=1.0, enabled_pstr_rules=None):
  symbolic, mask, _, _ = symbolic_joint_transition(
    {"agent_0": obs},
    {"agent_0": int(action)},
    memory=None,
    coverage=coverage,
    enabled_pstr_rules=enabled_pstr_rules,
  )
  return symbolic["agent_0"], mask["agent_0"]


def symbolic_transition_with_report(obs, action, coverage=1.0, enabled_pstr_rules=None):
  symbolic, mask, memory, report = symbolic_joint_transition(
    {"agent_0": obs},
    {"agent_0": int(action)},
    memory=None,
    coverage=coverage,
    enabled_pstr_rules=enabled_pstr_rules,
  )
  return symbolic["agent_0"], mask["agent_0"], memory, report


def symbolic_joint_transition(joint_obs, joint_action, memory=None, coverage=1.0, enabled_pstr_rules=None):
  """Predict a conservative partial joint symbolic observation.

  Any feature whose mask is False must be interpreted as unknown, even though
  the value arrays carry convenient copied values for compatibility with the
  existing projection/training code.
  """
  memory = init_symbolic_memory(memory)
  agents = sorted(joint_obs)
  report = _new_report()
  _align_agents_from_observations(joint_obs, memory, report)
  _fuse_observations_into_map(joint_obs, memory, report)

  proposed_shifts = {}
  for agent_id in agents:
    action = int(joint_action.get(agent_id, ACTION_STAY))
    proposed_shifts[agent_id] = _agent_shift(np.asarray(joint_obs[agent_id]["grid"], dtype=np.int8), action, report=report)
  _apply_joint_collision_rules(proposed_shifts, memory, report)

  symbolic = {}
  mask = {}
  for agent_id in agents:
    action = int(joint_action.get(agent_id, ACTION_STAY))
    symbolic[agent_id], mask[agent_id] = _symbolic_agent_transition(
      joint_obs[agent_id],
      action,
      memory,
      agent_id,
      proposed_shifts.get(agent_id),
      report,
    )
    _apply_global_static_query(symbolic[agent_id], mask[agent_id], memory, agent_id, proposed_shifts.get(agent_id), report)

  _apply_joint_world_updates(joint_obs, joint_action, memory, report)
  _apply_joint_agent_entity_prediction(symbolic, mask, memory, proposed_shifts, report)
  for agent_id in agents:
    apply_coverage(mask[agent_id], coverage)
  _filter_rules_and_masks(mask, report, enabled_pstr_rules)
  _refresh_report_counts(report, mask)
  return symbolic, mask, memory, report


def init_symbolic_memory(memory=None):
  if memory is None:
    memory = {}
  memory = deepcopy(memory)
  memory.setdefault("agent_pos", {})
  memory.setdefault("terrain", {})
  memory.setdefault("blocks", {})
  memory.setdefault("entities", {})
  memory.setdefault("items", {})
  memory.setdefault("mobs", {})
  memory.setdefault("hunger_counters", {})
  memory.setdefault("step", 0)
  return memory


def apply_symbolic_projection(predicted_obs, current_obs, action, variant, coverage=1.0, memory=None, enabled_pstr_rules=None):
  if variant == "neural" or variant == "regularization":
    return predicted_obs, None
  if _is_joint_obs(current_obs):
    symbolic, mask, updated_memory, report = symbolic_joint_transition(
      current_obs,
      action,
      memory=memory,
      coverage=coverage,
      enabled_pstr_rules=enabled_pstr_rules,
    )
    projected = {}
    for agent_id, obs in predicted_obs.items():
      projected[agent_id] = _project_tabular(obs, symbolic[agent_id], mask[agent_id])
    return projected, (symbolic, mask, updated_memory, report)
  symbolic, mask = symbolic_transition(current_obs, action, coverage=coverage, enabled_pstr_rules=enabled_pstr_rules)
  return _project_tabular(predicted_obs, symbolic, mask), (symbolic, mask)


def compare_with_symbolic(predicted_obs, current_obs, action, coverage=1.0, memory=None, enabled_pstr_rules=None):
  if _is_joint_obs(current_obs):
    symbolic, mask, _, report = symbolic_joint_transition(
      current_obs,
      action,
      memory=memory,
      coverage=coverage,
      enabled_pstr_rules=enabled_pstr_rules,
    )
    return compare_joint_with_symbolic(predicted_obs, symbolic, mask, report)
  symbolic, mask, _, report = symbolic_transition_with_report(current_obs, action, coverage=coverage, enabled_pstr_rules=enabled_pstr_rules)
  metrics = _compare_single_with_symbolic(predicted_obs, symbolic, mask)
  metrics.update(_rule_metrics({"agent_0": predicted_obs}, {"agent_0": symbolic}, {"agent_0": mask}, report))
  metrics["rvr_global"] = metrics["rvr"]
  metrics["rvr_individual"] = metrics["rvr"]
  metrics["rvr_joint"] = 0.0
  metrics["determinable_count_individual"] = metrics["determinable_count"]
  metrics["determinable_count_joint"] = 0.0
  metrics["map_coverage_ratio"] = float(report.get("map_coverage_ratio", 0.0))
  metrics["relative_alignment_count"] = float(report.get("relative_alignment_count", 0.0))
  return metrics


def compare_joint_with_symbolic(predicted_joint_obs, symbolic_joint, joint_mask, report=None):
  report = report or _new_report()
  rows = []
  for agent_id, symbolic in symbolic_joint.items():
    if agent_id in predicted_joint_obs:
      rows.append(_compare_single_with_symbolic(predicted_joint_obs[agent_id], symbolic, joint_mask[agent_id]))
  if not rows:
    metrics = {"rvr": 0.0, "determinable_mismatch": 0.0, "undeterminable_mismatch": 0.0, "determinable_count": 0.0}
  else:
    metrics = {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}
  metrics["rvr_global"] = metrics["rvr"]
  metrics["rvr_individual"] = _scoped_rvr(predicted_joint_obs, symbolic_joint, joint_mask, report, "individual")
  metrics["rvr_joint"] = _scoped_rvr(predicted_joint_obs, symbolic_joint, joint_mask, report, "joint")
  metrics["determinable_count_individual"] = float(report.get("determinable_count_individual", 0.0))
  metrics["determinable_count_joint"] = float(report.get("determinable_count_joint", 0.0))
  metrics["map_coverage_ratio"] = float(report.get("map_coverage_ratio", 0.0))
  metrics["relative_alignment_count"] = float(report.get("relative_alignment_count", 0.0))
  metrics.update(_rule_metrics(predicted_joint_obs, symbolic_joint, joint_mask, report))
  return metrics


def symbolic_batch_targets(obs_batch, action_batch, coverage=1.0, enabled_pstr_rules=None):
  batch, seq_len = action_batch.shape
  targets = np.zeros((batch, seq_len, GRID_FEATURES + SELF_FEATURES), dtype=np.float32)
  masks = np.zeros_like(targets, dtype=np.float32)
  for b in range(batch):
    for t in range(seq_len):
      symbolic, mask = symbolic_transition_from_vector(
        obs_batch[b, t],
        int(action_batch[b, t]),
        coverage=coverage,
        enabled_pstr_rules=enabled_pstr_rules,
      )
      targets[b, t] = tabular_to_vector(symbolic)
      masks[b, t] = tabular_mask_to_vector_mask(mask).astype(np.float32)
  return targets, masks


def vector_to_tabular(vector):
  vector = np.asarray(vector, dtype=np.float32)
  cursor = 0
  terrain = vector[cursor:cursor + GRID_CELLS * TERRAIN_CLASSES].reshape(GRID_CELLS, TERRAIN_CLASSES).argmax(axis=1)
  cursor += GRID_CELLS * TERRAIN_CLASSES
  blocks = vector[cursor:cursor + GRID_CELLS * BLOCK_CLASSES].reshape(GRID_CELLS, BLOCK_CLASSES).argmax(axis=1)
  cursor += GRID_CELLS * BLOCK_CLASSES
  entities = vector[cursor:cursor + GRID_CELLS * ENTITY_CLASSES].reshape(GRID_CELLS, ENTITY_CLASSES).argmax(axis=1)
  self_vec = vector[GRID_FEATURES:GRID_FEATURES + SELF_FEATURES]
  hp_hunger = np.clip(np.rint(self_vec[:2] * 20.0), 0, 20)
  inventory = np.clip(np.rint(self_vec[2:] * 10.0), 0, 99)
  side = int(np.sqrt(GRID_CELLS))
  return {
    "grid": np.stack([
      terrain.reshape(side, side),
      blocks.reshape(side, side),
      entities.reshape(side, side),
    ]).astype(np.int8),
    "self": np.concatenate([hp_hunger, inventory]).astype(np.int16),
  }


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


def normalize_enabled_pstr_rules(enabled_pstr_rules):
  if enabled_pstr_rules is None:
    return None
  if isinstance(enabled_pstr_rules, str):
    enabled_pstr_rules = [part.strip() for part in enabled_pstr_rules.split(",")]
  expanded = []
  for rule in enabled_pstr_rules:
    expanded.extend(str(rule).split(","))
  rules = {rule.strip() for rule in expanded if rule.strip()}
  return rules or None


def _filter_rules_and_masks(joint_mask, report, enabled_pstr_rules):
  enabled = normalize_enabled_pstr_rules(enabled_pstr_rules)
  if enabled is None:
    return
  filtered_rules = {
    rule_id: entries
    for rule_id, entries in report.get("rules", {}).items()
    if rule_id in enabled
  }
  filtered_mask = {}
  for agent_id, mask in joint_mask.items():
    filtered_mask[agent_id] = {
      "grid": np.zeros_like(mask["grid"], dtype=np.bool_),
      "self": np.zeros_like(mask["self"], dtype=np.bool_),
    }
  for entries in filtered_rules.values():
    for entry in entries:
      agent_id = entry["agent"]
      if agent_id == "*" or agent_id not in joint_mask:
        continue
      for feature in entry["features"]:
        if feature[0] == "grid":
          _, channel, y, x = feature
          if joint_mask[agent_id]["grid"][channel, y, x]:
            filtered_mask[agent_id]["grid"][channel, y, x] = True
        elif feature[0] == "self":
          _, idx = feature
          if joint_mask[agent_id]["self"][idx]:
            filtered_mask[agent_id]["self"][idx] = True
  for agent_id, mask in joint_mask.items():
    mask["grid"][:] = filtered_mask[agent_id]["grid"]
    mask["self"][:] = filtered_mask[agent_id]["self"]
  report["rules"] = filtered_rules


def _symbolic_agent_transition(obs, action, memory, agent_id, forced_shift, report):
  grid = np.asarray(obs["grid"], dtype=np.int8)
  symbolic = {
    "grid": grid.copy(),
    "self": np.asarray(obs["self"], dtype=np.int16).copy(),
  }
  mask = {
    "grid": np.zeros_like(grid, dtype=np.bool_),
    "self": np.zeros((SELF_FEATURES,), dtype=np.bool_),
  }
  if forced_shift is not None:
    _apply_shifted_static_planes(grid, symbolic["grid"], mask["grid"], forced_shift)
    _record_rule(report, "PSTR_INDIV_STATIC_TERRAIN_SHIFT", agent_id, _grid_positions(mask["grid"], channel=0), "individual")
    _record_rule(report, "PSTR_INDIV_STATIC_BLOCK_SHIFT", agent_id, _grid_positions(mask["grid"], channel=1), "individual")
  _apply_center_agent(symbolic, mask, report, agent_id)
  _apply_harvest(symbolic, mask, grid, action, report, agent_id)
  _apply_pickup(symbolic, mask, grid, action, memory, agent_id, report)
  _apply_eat(symbolic, mask, action, report, agent_id)
  _apply_craft(symbolic, mask, action, report, agent_id)
  _apply_hunger_cost(symbolic, mask, action, memory, agent_id, report)
  _apply_attack_memory_update(grid, action, memory, agent_id, report)
  return symbolic, mask


def _apply_center_agent(symbolic, mask, report, agent_id):
  c = GRIDCRAFT_VIEW_SIZE // 2
  symbolic["grid"][2, c, c] = ENTITY_AGENT
  mask["grid"][2, c, c] = True
  _record_rule(report, "PSTR_INDIV_CENTER_AGENT", agent_id, [("grid", 2, c, c)], "individual")


def _apply_harvest(symbolic, mask, grid, action, report, agent_id):
  if int(action) != ACTION_HARVEST:
    return
  c = GRIDCRAFT_VIEW_SIZE // 2
  for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
    x, y = c + dx, c + dy
    if not (0 <= x < GRIDCRAFT_VIEW_SIZE and 0 <= y < GRIDCRAFT_VIEW_SIZE):
      continue
    block = int(grid[1, y, x])
    if block == BLOCK_TREE:
      symbolic["grid"][1, y, x] = BLOCK_EMPTY
      mask["grid"][1, y, x] = True
      _inc_self(symbolic, mask, ITEM_WOOD, 1)
      _record_rule(report, "PSTR_INDIV_HARVEST_TREE_WOOD", agent_id, [("grid", 1, y, x), ("self", 2 + ITEM_WOOD)], "individual")
      return
    if block == BLOCK_STONE and (_has_item(symbolic, ITEM_WOOD_PICKAXE) or _has_item(symbolic, ITEM_STONE_PICKAXE)):
      symbolic["grid"][1, y, x] = BLOCK_EMPTY
      mask["grid"][1, y, x] = True
      _inc_self(symbolic, mask, ITEM_STONE, 1)
      _record_rule(report, "PSTR_INDIV_HARVEST_STONE_PICKAXE", agent_id, [("grid", 1, y, x), ("self", 2 + ITEM_STONE)], "individual")
      return


def _apply_pickup(symbolic, mask, grid, action, memory, agent_id, report):
  if int(action) != ACTION_PICKUP:
    return
  c = GRIDCRAFT_VIEW_SIZE // 2
  pos = memory.get("agent_pos", {}).get(agent_id)
  for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
    x, y = c + dx, c + dy
    if not (0 <= x < GRIDCRAFT_VIEW_SIZE and 0 <= y < GRIDCRAFT_VIEW_SIZE):
      continue
    if int(grid[2, y, x]) != ENTITY_ITEM:
      continue
    item_pos = (pos[0] + dx, pos[1] + dy) if pos is not None else None
    item = memory.get("items", {}).get(item_pos) if item_pos is not None else None
    symbolic["grid"][2, y, x] = ENTITY_NONE
    mask["grid"][2, y, x] = True
    features = [("grid", 2, y, x)]
    if item is not None:
      item_id, count = item
      _inc_self(symbolic, mask, int(item_id), int(count))
      features.append(("self", 2 + int(item_id)))
    _record_rule(report, "PSTR_INDIV_PICKUP_ITEM", agent_id, features, "individual")
    return


def _apply_eat(symbolic, mask, action, report, agent_id):
  if int(action) != ACTION_EAT:
    return
  hunger = int(symbolic["self"][1])
  apples = int(symbolic["self"][2 + ITEM_APPLE])
  if hunger < 20 and apples > 0:
    symbolic["self"][1] = min(20, hunger + 6)
    symbolic["self"][2 + ITEM_APPLE] = apples - 1
    mask["self"][1] = True
    mask["self"][2 + ITEM_APPLE] = True
    _record_rule(report, "PSTR_INDIV_EAT_APPLE", agent_id, [("self", 1), ("self", 2 + ITEM_APPLE)], "individual")


def _apply_craft(symbolic, mask, action, report, agent_id):
  recipe = CRAFT_RECIPES.get(int(action))
  if recipe is None:
    return
  inputs, output, rule_id = recipe
  if any(int(symbolic["self"][2 + item]) < count for item, count in inputs):
    return
  features = []
  for item, count in inputs:
    symbolic["self"][2 + item] -= count
    mask["self"][2 + item] = True
    features.append(("self", 2 + item))
  out_item, out_count = output
  symbolic["self"][2 + out_item] += out_count
  mask["self"][2 + out_item] = True
  features.append(("self", 2 + out_item))
  _record_rule(report, rule_id, agent_id, features, "individual")


def _apply_hunger_cost(symbolic, mask, action, memory, agent_id, report):
  counters = memory.get("hunger_counters", {}).get(agent_id)
  if not counters:
    return
  if int(action) in (ACTION_MOVE_N, ACTION_MOVE_S, ACTION_MOVE_W, ACTION_MOVE_E):
    kind, interval = "move", 5
  elif int(action) == ACTION_HARVEST:
    kind, interval = "harvest", 3
  elif int(action) == ACTION_ATTACK:
    kind, interval = "attack", 3
  else:
    return
  next_counter = int(counters.get(kind, 0)) + 1
  counters[kind] = next_counter
  if next_counter >= interval:
    symbolic["self"][1] = max(0, int(symbolic["self"][1]) - 1)
    mask["self"][1] = True
    counters[kind] = 0
    _record_rule(report, "PSTR_INDIV_HUNGER_COST_KNOWN_COUNTER", agent_id, [("self", 1)], "individual")


def _apply_attack_memory_update(grid, action, memory, agent_id, report):
  if int(action) != ACTION_ATTACK:
    return
  pos = memory.get("agent_pos", {}).get(agent_id)
  if pos is None:
    return
  c = GRIDCRAFT_VIEW_SIZE // 2
  for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
    lx, ly = c + dx, c + dy
    if 0 <= lx < GRIDCRAFT_VIEW_SIZE and 0 <= ly < GRIDCRAFT_VIEW_SIZE and int(grid[2, ly, lx]) == ENTITY_MOB:
      mob_pos = (pos[0] + dx, pos[1] + dy)
      mob = memory.get("mobs", {}).get(mob_pos)
      if mob and "hp" in mob:
        damage = 2
        mob["hp"] = int(mob["hp"]) - damage
        if mob["hp"] <= 0:
          memory["mobs"].pop(mob_pos, None)
          memory["entities"][mob_pos] = ENTITY_NONE
      _record_rule(report, "PSTR_INDIV_ATTACK_MOB_LOCAL", agent_id, [], "individual")
      return


def _align_agents_from_observations(joint_obs, memory, report):
  agents = sorted(joint_obs)
  if agents and not memory["agent_pos"]:
    memory["agent_pos"][agents[0]] = (0, 0)
  center = GRIDCRAFT_VIEW_SIZE // 2
  candidates = []
  for agent_id, obs in joint_obs.items():
    grid = np.asarray(obs["grid"], dtype=np.int8)
    ys, xs = np.where(grid[2] == ENTITY_AGENT)
    for y, x in zip(ys.tolist(), xs.tolist()):
      dx, dy = x - center, y - center
      if dx == 0 and dy == 0:
        continue
      candidates.append((agent_id, dx, dy))
  for source, dx, dy in candidates:
    source_pos = memory["agent_pos"].get(source)
    if source_pos is None:
      continue
    for target, rdx, rdy in candidates:
      if target == source:
        continue
      if rdx == -dx and rdy == -dy and target not in memory["agent_pos"]:
        memory["agent_pos"][target] = (source_pos[0] + dx, source_pos[1] + dy)
        _record_rule(report, "PSTR_JOINT_RELATIVE_AGENT_ALIGNMENT", target, [], "joint")
  report["relative_alignment_count"] = len(memory["agent_pos"])


def _fuse_observations_into_map(joint_obs, memory, report):
  center = GRIDCRAFT_VIEW_SIZE // 2
  known = 0
  for agent_id, obs in joint_obs.items():
    pos = memory["agent_pos"].get(agent_id)
    if pos is None:
      continue
    grid = np.asarray(obs["grid"], dtype=np.int8)
    for y in range(GRIDCRAFT_VIEW_SIZE):
      for x in range(GRIDCRAFT_VIEW_SIZE):
        gx, gy = pos[0] + (x - center), pos[1] + (y - center)
        key = (gx, gy)
        memory["terrain"][key] = int(grid[0, y, x])
        memory["blocks"][key] = int(grid[1, y, x])
        ent = int(grid[2, y, x])
        if ent != ENTITY_AGENT:
          memory["entities"][key] = ent
        if ent == ENTITY_ITEM:
          memory["items"].setdefault(key, None)
        if ent == ENTITY_MOB:
          memory["mobs"].setdefault(key, {})
        known += 1
  if known:
    _record_rule(report, "PSTR_JOINT_MAP_FUSION", "*", [], "joint")
  report["map_coverage_ratio"] = float(len(memory["terrain"]) / max(1, 16 * 16))


def _apply_global_static_query(symbolic, mask, memory, agent_id, shift, report):
  pos = memory.get("agent_pos", {}).get(agent_id)
  if pos is None or shift is None:
    return
  center = GRIDCRAFT_VIEW_SIZE // 2
  next_pos = (pos[0] + shift[0], pos[1] + shift[1])
  features = []
  for y in range(GRIDCRAFT_VIEW_SIZE):
    for x in range(GRIDCRAFT_VIEW_SIZE):
      gx, gy = next_pos[0] + (x - center), next_pos[1] + (y - center)
      key = (gx, gy)
      if key in memory["terrain"]:
        symbolic["grid"][0, y, x] = memory["terrain"][key]
        mask["grid"][0, y, x] = True
        features.append(("grid", 0, y, x))
      if key in memory["blocks"]:
        symbolic["grid"][1, y, x] = memory["blocks"][key]
        mask["grid"][1, y, x] = True
        features.append(("grid", 1, y, x))
  if features:
    _record_rule(report, "PSTR_JOINT_GLOBAL_STATIC_QUERY", agent_id, features, "joint")


def _apply_joint_collision_rules(shifts, memory, report):
  desired = {}
  for agent_id, shift in shifts.items():
    pos = memory.get("agent_pos", {}).get(agent_id)
    if pos is not None and shift is not None:
      desired[agent_id] = (pos[0] + shift[0], pos[1] + shift[1])
  for a, a_target in desired.items():
    for b, b_target in desired.items():
      if a >= b:
        continue
      a_pos = memory["agent_pos"].get(a)
      b_pos = memory["agent_pos"].get(b)
      collision = a_target == b_target
      swap = a_pos is not None and b_pos is not None and a_target == b_pos and b_target == a_pos
      if collision or swap:
        shifts[a] = (0, 0)
        shifts[b] = (0, 0)
        _record_rule(report, "PSTR_JOINT_MULTI_AGENT_COLLISION", a, [], "joint")
        _record_rule(report, "PSTR_JOINT_MULTI_AGENT_COLLISION", b, [], "joint")


def _apply_joint_agent_entity_prediction(symbolic, mask, memory, shifts, report):
  center = GRIDCRAFT_VIEW_SIZE // 2
  next_positions = {}
  for agent_id, pos in memory.get("agent_pos", {}).items():
    shift = shifts.get(agent_id, (0, 0))
    if shift is not None:
      next_positions[agent_id] = (pos[0] + shift[0], pos[1] + shift[1])
  for observer, obs in symbolic.items():
    origin = next_positions.get(observer)
    if origin is None:
      continue
    features = []
    for other, pos in next_positions.items():
      if other == observer:
        continue
      lx, ly = center + (pos[0] - origin[0]), center + (pos[1] - origin[1])
      if 0 <= lx < GRIDCRAFT_VIEW_SIZE and 0 <= ly < GRIDCRAFT_VIEW_SIZE:
        obs["grid"][2, ly, lx] = ENTITY_AGENT
        mask[observer]["grid"][2, ly, lx] = True
        features.append(("grid", 2, ly, lx))
    if features:
      _record_rule(report, "PSTR_JOINT_AGENT_ENTITY_PREDICTION", observer, features, "joint")
  memory["agent_pos"].update(next_positions)


def _apply_joint_world_updates(joint_obs, joint_action, memory, report):
  center = GRIDCRAFT_VIEW_SIZE // 2
  for agent_id, obs in joint_obs.items():
    pos = memory.get("agent_pos", {}).get(agent_id)
    if pos is None:
      continue
    action = int(joint_action.get(agent_id, ACTION_STAY))
    grid = np.asarray(obs["grid"], dtype=np.int8)
    if action == ACTION_HARVEST:
      for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        lx, ly = center + dx, center + dy
        if 0 <= lx < GRIDCRAFT_VIEW_SIZE and 0 <= ly < GRIDCRAFT_VIEW_SIZE and int(grid[1, ly, lx]) in (BLOCK_TREE, BLOCK_STONE):
          memory["blocks"][(pos[0] + dx, pos[1] + dy)] = BLOCK_EMPTY
          _record_rule(report, "PSTR_JOINT_SHARED_WORLD_UPDATE", agent_id, [], "joint")
          break
    if action == ACTION_PICKUP:
      for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        lx, ly = center + dx, center + dy
        if not (0 <= lx < GRIDCRAFT_VIEW_SIZE and 0 <= ly < GRIDCRAFT_VIEW_SIZE):
          continue
        if int(grid[2, ly, lx]) != ENTITY_ITEM:
          continue
        item_pos = (pos[0] + dx, pos[1] + dy)
        memory["items"].pop(item_pos, None)
        memory["entities"].pop(item_pos, None)
        _record_rule(report, "PSTR_JOINT_SHARED_ITEM_UPDATE", agent_id, [], "joint")
        break


def _agent_shift(grid, action, report=None):
  if action == ACTION_STAY:
    return (0, 0)
  if action not in MOVE_DELTAS:
    return None
  dx, dy = MOVE_DELTAS[action]
  center = GRIDCRAFT_VIEW_SIZE // 2
  target_x = center + dx
  target_y = center + dy
  if not (0 <= target_x < GRIDCRAFT_VIEW_SIZE and 0 <= target_y < GRIDCRAFT_VIEW_SIZE):
    return (0, 0)
  terrain = int(grid[0, target_y, target_x])
  block = int(grid[1, target_y, target_x])
  entity = int(grid[2, target_y, target_x])
  if terrain == TERRAIN_WATER:
    if report is not None:
      _record_rule(report, "PSTR_INDIV_BLOCKED_WATER", "*", [], "individual")
    return (0, 0)
  if block in (BLOCK_TREE, BLOCK_STONE):
    if report is not None:
      _record_rule(report, "PSTR_INDIV_BLOCKED_TREE_STONE", "*", [], "individual")
    return (0, 0)
  if entity in (ENTITY_AGENT, ENTITY_MOB):
    if report is not None:
      _record_rule(report, "PSTR_INDIV_BLOCKED_ENTITY", "*", [], "individual")
    return (0, 0)
  return (dx, dy)


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


def _project_tabular(predicted, symbolic, mask):
  projected = {
    "grid": np.asarray(predicted["grid"], dtype=np.int8).copy(),
    "self": np.asarray(predicted["self"], dtype=np.int16).copy(),
  }
  projected["grid"][mask["grid"]] = symbolic["grid"][mask["grid"]]
  projected["self"][mask["self"]] = symbolic["self"][mask["self"]]
  return projected


def _compare_single_with_symbolic(predicted_obs, symbolic, mask):
  grid_mask = mask["grid"]
  self_mask = mask["self"]
  total = int(np.sum(grid_mask) + np.sum(self_mask))
  if total == 0:
    return {"rvr": 0.0, "determinable_mismatch": 0.0, "undeterminable_mismatch": 0.0, "determinable_count": 0.0}
  predicted_grid = np.asarray(predicted_obs["grid"], dtype=np.int16)
  predicted_self = np.asarray(predicted_obs["self"], dtype=np.float32)
  symbolic_grid = np.asarray(symbolic["grid"], dtype=np.int16)
  symbolic_self = np.asarray(symbolic["self"], dtype=np.float32)
  grid_violations = int(np.sum(predicted_grid[grid_mask] != symbolic_grid[grid_mask]))
  self_violations = int(np.sum(np.abs(predicted_self[self_mask] - symbolic_self[self_mask]) > 0.5)) if np.any(self_mask) else 0
  predicted_vec = tabular_to_vector(predicted_obs)
  symbolic_vec = tabular_to_vector(symbolic)
  mask_vec = tabular_mask_to_vector_mask(mask)
  return {
    "rvr": float((grid_violations + self_violations) / total),
    "determinable_mismatch": float(np.mean(np.abs(predicted_vec[mask_vec] - symbolic_vec[mask_vec]))) if np.any(mask_vec) else 0.0,
    "undeterminable_mismatch": float(np.mean(np.abs(predicted_vec[~mask_vec] - symbolic_vec[~mask_vec]))) if np.any(~mask_vec) else 0.0,
    "determinable_count": float(total),
  }


def _rule_metrics(predicted_joint, symbolic_joint, joint_mask, report):
  metrics = {}
  for rule_id, entries in report.get("rules", {}).items():
    total = 0
    violations = 0
    for entry in entries:
      agent_id = entry["agent"]
      if agent_id == "*" or agent_id not in predicted_joint:
        continue
      for feature in entry["features"]:
        if feature[0] == "grid":
          _, channel, y, x = feature
          if not joint_mask[agent_id]["grid"][channel, y, x]:
            continue
          total += 1
          violations += int(predicted_joint[agent_id]["grid"][channel, y, x] != symbolic_joint[agent_id]["grid"][channel, y, x])
        elif feature[0] == "self":
          _, idx = feature
          if not joint_mask[agent_id]["self"][idx]:
            continue
          total += 1
          violations += int(abs(float(predicted_joint[agent_id]["self"][idx]) - float(symbolic_joint[agent_id]["self"][idx])) > 0.5)
    if total:
      metrics[f"rvr/{rule_id}"] = float(violations / total)
      metrics[f"determinable_count/{rule_id}"] = float(total)
  return metrics


def _scoped_rvr(predicted_joint, symbolic_joint, joint_mask, report, scope):
  total = 0
  violations = 0
  for rule_id, entries in report.get("rules", {}).items():
    for entry in entries:
      if entry["scope"] != scope:
        continue
      agent_id = entry["agent"]
      if agent_id == "*" or agent_id not in predicted_joint:
        continue
      for feature in entry["features"]:
        if feature[0] == "grid":
          _, channel, y, x = feature
          if joint_mask[agent_id]["grid"][channel, y, x]:
            total += 1
            violations += int(predicted_joint[agent_id]["grid"][channel, y, x] != symbolic_joint[agent_id]["grid"][channel, y, x])
        elif feature[0] == "self":
          _, idx = feature
          if joint_mask[agent_id]["self"][idx]:
            total += 1
            violations += int(abs(float(predicted_joint[agent_id]["self"][idx]) - float(symbolic_joint[agent_id]["self"][idx])) > 0.5)
  return float(violations / total) if total else 0.0


def _new_report():
  return {
    "rules": {},
    "determinable_count_individual": 0.0,
    "determinable_count_joint": 0.0,
    "map_coverage_ratio": 0.0,
    "relative_alignment_count": 0.0,
  }


def _record_rule(report, rule_id, agent_id, features, scope):
  report.setdefault("rules", {}).setdefault(rule_id, []).append({
    "agent": agent_id,
    "features": list(features),
    "scope": scope,
  })


def _refresh_report_counts(report, joint_mask):
  individual = 0
  joint = 0
  for entries in report.get("rules", {}).values():
    for entry in entries:
      count = 0
      agent_id = entry["agent"]
      if agent_id != "*" and agent_id in joint_mask:
        for feature in entry["features"]:
          if feature[0] == "grid":
            _, channel, y, x = feature
            count += int(joint_mask[agent_id]["grid"][channel, y, x])
          elif feature[0] == "self":
            _, idx = feature
            count += int(joint_mask[agent_id]["self"][idx])
      if entry["scope"] == "joint":
        joint += count
      else:
        individual += count
  report["determinable_count_individual"] = float(individual)
  report["determinable_count_joint"] = float(joint)


def _grid_positions(grid_mask, channel):
  ys, xs = np.where(grid_mask[channel])
  return [("grid", channel, int(y), int(x)) for y, x in zip(ys.tolist(), xs.tolist())]


def _inc_self(symbolic, mask, item, count):
  idx = 2 + int(item)
  symbolic["self"][idx] = int(symbolic["self"][idx]) + int(count)
  mask["self"][idx] = True


def _has_item(symbolic, item):
  return int(symbolic["self"][2 + int(item)]) > 0


def _one_hot(values, depth):
  values = np.asarray(values, dtype=np.int64).reshape(-1)
  values = np.clip(values, 0, depth - 1)
  return np.eye(depth, dtype=np.float32)[values].reshape(-1)


def _is_joint_obs(obs):
  return isinstance(obs, dict) and bool(obs) and all(isinstance(value, dict) and "grid" in value and "self" in value for value in obs.values()) and "grid" not in obs
