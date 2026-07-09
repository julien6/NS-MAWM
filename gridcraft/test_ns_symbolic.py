import numpy as np

from ns_symbolic import (
  ACTION_EAT,
  ACTION_HARVEST,
  ACTION_MOVE_E,
  ACTION_PICKUP,
  ACTION_STAY,
  BLOCK_TREE,
  ENTITY_AGENT,
  ENTITY_ITEM,
  ENTITY_MOB,
  ENTITY_NONE,
  ITEM_PLANK,
  ITEM_STONE,
  ITEM_WOOD,
  apply_symbolic_projection,
  compare_with_symbolic,
  symbolic_joint_transition,
  symbolic_batch_targets,
  symbolic_transition,
  tabular_to_vector,
)
from pstr_profiles import active_rules_for_baseline


def make_obs():
  grid = np.zeros((3, 7, 7), dtype=np.int8)
  grid[2, 3, 3] = ENTITY_AGENT
  self_vec = np.zeros((11,), dtype=np.int16)
  self_vec[0] = 20
  self_vec[1] = 20
  return {"grid": grid, "self": self_vec}


def test_move_shift_masks_static_planes():
  obs = make_obs()
  obs["grid"][0, 3, 4] = 2
  symbolic, mask = symbolic_transition(obs, ACTION_MOVE_E, coverage=1.0)
  assert symbolic["grid"][0, 3, 3] == 2
  assert mask["grid"][0, 3, 3]
  assert symbolic["grid"][2, 3, 3] == ENTITY_AGENT
  assert mask["grid"][2, 3, 3]


def test_move_shift_does_not_mask_uncertain_entities():
  obs = make_obs()
  obs["grid"][2, 3, 4] = ENTITY_MOB
  symbolic, mask = symbolic_transition(obs, ACTION_MOVE_E, coverage=1.0)

  assert not mask["grid"][2, 3, 2]
  assert not mask["grid"][2, 3, 4]
  assert symbolic["grid"][2, 3, 3] == ENTITY_AGENT
  assert mask["grid"][2, 3, 3]


def test_non_moving_actions_preserve_static_planes():
  obs = make_obs()
  obs["grid"][0, 1, 2] = 2
  obs["grid"][1, 4, 5] = BLOCK_TREE
  symbolic, mask = symbolic_transition(obs, ACTION_EAT, coverage=1.0)

  assert symbolic["grid"][0, 1, 2] == 2
  assert symbolic["grid"][1, 4, 5] == BLOCK_TREE
  assert np.all(symbolic["grid"][0] == obs["grid"][0])
  assert np.all(symbolic["grid"][1] == obs["grid"][1])
  assert np.all(mask["grid"][0])
  assert np.all(mask["grid"][1])


def test_harvest_tree_predicts_wood_not_apple():
  obs = make_obs()
  obs["grid"][1, 3, 2] = BLOCK_TREE
  obs["self"][2 + ITEM_WOOD] = 2
  symbolic, mask = symbolic_transition(obs, ACTION_HARVEST, coverage=1.0)
  assert symbolic["grid"][1, 3, 2] == 0
  assert mask["grid"][1, 3, 2]
  assert np.all(mask["grid"][0])
  assert np.all(mask["grid"][1])
  assert symbolic["self"][2 + ITEM_WOOD] == 3
  assert mask["self"][2 + ITEM_WOOD]
  assert not mask["self"][2 + 8]


def test_craft_plank_updates_inventory():
  obs = make_obs()
  obs["self"][2 + ITEM_WOOD] = 2
  symbolic, mask = symbolic_transition(obs, 9, coverage=1.0)
  assert symbolic["self"][2 + ITEM_WOOD] == 1
  assert symbolic["self"][2 + ITEM_PLANK] == 2
  assert mask["self"][2 + ITEM_WOOD]
  assert mask["self"][2 + ITEM_PLANK]


def test_pickup_predicts_adjacent_item_collection():
  obs = make_obs()
  obs["grid"][2, 3, 4] = ENTITY_ITEM
  symbolic, mask, memory, report = symbolic_joint_transition(
    {"agent_0": obs},
    {"agent_0": ACTION_PICKUP},
    memory={"agent_pos": {"agent_0": (0, 0)}, "items": {(1, 0): (ITEM_STONE, 2)}},
    coverage=1.0,
  )
  agent_symbolic = symbolic["agent_0"]
  agent_mask = mask["agent_0"]
  assert agent_symbolic["grid"][2, 3, 3] == ENTITY_AGENT
  assert agent_symbolic["grid"][2, 3, 4] == 0
  assert agent_mask["grid"][2, 3, 4]
  assert agent_symbolic["self"][2 + ITEM_STONE] == 2
  assert agent_mask["self"][2 + ITEM_STONE]
  assert "PSTR_INDIV_PICKUP_ITEM" in report["rules"]


def test_joint_pickup_removes_adjacent_item_from_memory():
  obs = make_obs()
  obs["grid"][2, 3, 4] = ENTITY_ITEM
  _, _, memory, report = symbolic_joint_transition(
    {"agent_0": obs},
    {"agent_0": ACTION_PICKUP},
    memory={"agent_pos": {"agent_0": (0, 0)}, "items": {(1, 0): (ITEM_STONE, 2)}, "entities": {(1, 0): ENTITY_ITEM}},
    coverage=1.0,
  )
  assert (1, 0) not in memory["items"]
  assert (1, 0) not in memory["entities"]
  assert "PSTR_JOINT_SHARED_ITEM_UPDATE" in report["rules"]


def test_joint_alignment_predicts_other_agent_entity():
  obs0 = make_obs()
  obs1 = make_obs()
  obs0["grid"][2, 3, 4] = ENTITY_AGENT
  obs1["grid"][2, 3, 2] = ENTITY_AGENT
  symbolic, mask, memory, report = symbolic_joint_transition(
    {"agent_0": obs0, "agent_1": obs1},
    {"agent_0": 0, "agent_1": 0},
    memory=None,
    coverage=1.0,
  )
  assert memory["agent_pos"]["agent_1"] == (1, 0)
  assert symbolic["agent_0"]["grid"][2, 3, 4] == ENTITY_AGENT
  assert mask["agent_0"]["grid"][2, 3, 4]
  assert "PSTR_JOINT_RELATIVE_AGENT_ALIGNMENT" in report["rules"]
  assert "PSTR_JOINT_AGENT_ENTITY_PREDICTION" in report["rules"]


def test_pstr_coverage_is_forced_to_full_masks():
  obs = make_obs()
  symbolic, mask = symbolic_transition(obs, 0, coverage=0.0)
  assert np.any(mask["grid"])
  assert np.all(mask["grid"][0])
  assert np.all(mask["grid"][1])
  assert not np.any(mask["self"])


def test_projection_static_terrain_rule_has_zero_post_rvr():
  obs = make_obs()
  obs["grid"][0, 3, 4] = 2
  raw = make_obs()
  raw["grid"][0, 3, 3] = 0
  enabled = ["PSTR_INDIV_STATIC_TERRAIN_SHIFT"]

  pre = compare_with_symbolic(raw, obs, ACTION_MOVE_E, coverage=1.0, enabled_pstr_rules=enabled)
  projected, _ = apply_symbolic_projection(raw, obs, ACTION_MOVE_E, "projection", coverage=1.0, enabled_pstr_rules=enabled)
  post = compare_with_symbolic(projected, obs, ACTION_MOVE_E, coverage=1.0, enabled_pstr_rules=enabled)

  assert pre["rvr/PSTR_INDIV_STATIC_TERRAIN_SHIFT"] > 0.0
  assert post["rvr"] == 0.0
  assert post["rvr/PSTR_INDIV_STATIC_TERRAIN_SHIFT"] == 0.0


def test_projection_pickup_rule_has_zero_post_rvr():
  obs = make_obs()
  obs["grid"][2, 3, 4] = ENTITY_ITEM
  raw = make_obs()
  raw["grid"][2, 3, 4] = ENTITY_ITEM
  enabled = ["PSTR_INDIV_PICKUP_ITEM"]
  memory = {"agent_pos": {"agent_0": (0, 0)}, "items": {(1, 0): (ITEM_STONE, 2)}}
  current = {"agent_0": obs}
  predicted = {"agent_0": raw}
  action = {"agent_0": ACTION_PICKUP}

  pre = compare_with_symbolic(predicted, current, action, coverage=1.0, memory=memory, enabled_pstr_rules=enabled)
  projected, _ = apply_symbolic_projection(predicted, current, action, "projection", coverage=1.0, memory=memory, enabled_pstr_rules=enabled)
  post = compare_with_symbolic(projected, current, action, coverage=1.0, memory=memory, enabled_pstr_rules=enabled)

  assert pre["rvr/PSTR_INDIV_PICKUP_ITEM"] > 0.0
  assert projected["agent_0"]["grid"][2, 3, 4] == ENTITY_ITEM
  assert post["rvr"] == 0.0
  assert post["rvr/PSTR_INDIV_PICKUP_ITEM"] == 0.0


def test_projection_joint_agent_entity_rule_has_zero_post_rvr():
  obs0 = make_obs()
  obs1 = make_obs()
  obs0["grid"][2, 3, 4] = ENTITY_AGENT
  obs1["grid"][2, 3, 2] = ENTITY_AGENT
  raw0 = make_obs()
  raw1 = make_obs()
  raw0["grid"][2, 3, 4] = ENTITY_NONE
  current = {"agent_0": obs0, "agent_1": obs1}
  predicted = {"agent_0": raw0, "agent_1": raw1}
  action = {"agent_0": ACTION_STAY, "agent_1": ACTION_STAY}
  enabled = ["PSTR_JOINT_AGENT_ENTITY_PREDICTION"]

  pre = compare_with_symbolic(predicted, current, action, coverage=1.0, enabled_pstr_rules=enabled)
  projected, _ = apply_symbolic_projection(predicted, current, action, "projection", coverage=1.0, enabled_pstr_rules=enabled)
  post = compare_with_symbolic(projected, current, action, coverage=1.0, enabled_pstr_rules=enabled)

  assert pre["rvr/PSTR_JOINT_AGENT_ENTITY_PREDICTION"] > 0.0
  assert post["rvr"] == 0.0
  assert post["rvr/PSTR_JOINT_AGENT_ENTITY_PREDICTION"] == 0.0


def test_enabled_pstr_rules_filter_masks_and_report():
  obs = make_obs()
  obs["grid"][0, 3, 4] = 2
  obs["grid"][1, 3, 4] = BLOCK_TREE
  _, mask, _, report = symbolic_joint_transition(
    {"agent_0": obs},
    {"agent_0": ACTION_MOVE_E},
    coverage=1.0,
    enabled_pstr_rules=["PSTR_INDIV_STATIC_TERRAIN_SHIFT"],
  )

  assert set(report["rules"]) == {"PSTR_INDIV_STATIC_TERRAIN_SHIFT"}
  assert np.any(mask["agent_0"]["grid"][0])
  assert not np.any(mask["agent_0"]["grid"][1])
  assert not np.any(mask["agent_0"]["grid"][2])


def test_k03_profile_excludes_interaction_pstr_masks():
  obs = make_obs()
  obs["grid"][1, 3, 2] = BLOCK_TREE
  obs["self"][2 + ITEM_WOOD] = 2
  _, mask, _, report = symbolic_joint_transition(
    {"agent_0": obs},
    {"agent_0": ACTION_HARVEST},
    coverage=0.3,
    enabled_pstr_rules=active_rules_for_baseline("B25_residual_k0.3"),
  )

  assert "PSTR_INDIV_HARVEST_TREE_WOOD" not in report["rules"]
  assert not mask["agent_0"]["self"][2 + ITEM_WOOD]


def test_k06_profile_includes_interaction_pstr_masks():
  obs = make_obs()
  obs["grid"][1, 3, 2] = BLOCK_TREE
  obs["self"][2 + ITEM_WOOD] = 2
  _, mask, _, report = symbolic_joint_transition(
    {"agent_0": obs},
    {"agent_0": ACTION_HARVEST},
    coverage=0.6,
    enabled_pstr_rules=active_rules_for_baseline("B26_residual_k0.6"),
  )

  assert "PSTR_INDIV_HARVEST_TREE_WOOD" in report["rules"]
  assert mask["agent_0"]["self"][2 + ITEM_WOOD]


def test_symbolic_and_residual_training_masks_are_complementary():
  obs = make_obs()
  obs["grid"][0, 3, 4] = 2
  obs_vec = tabular_to_vector(obs).reshape(1, 1, -1)
  actions = np.asarray([[ACTION_MOVE_E]], dtype=np.int64)

  _, symbolic_mask = symbolic_batch_targets(
    obs_vec,
    actions,
    coverage=1.0,
    enabled_pstr_rules=["PSTR_INDIV_STATIC_TERRAIN_SHIFT"],
  )
  residual_mask = 1.0 - symbolic_mask

  assert np.all((symbolic_mask == 0.0) | (symbolic_mask == 1.0))
  assert np.all((residual_mask == 0.0) | (residual_mask == 1.0))
  assert np.all(symbolic_mask + residual_mask == 1.0)
  assert np.any(symbolic_mask == 1.0)


if __name__ == "__main__":
  test_move_shift_masks_static_planes()
  test_harvest_tree_predicts_wood_not_apple()
  test_craft_plank_updates_inventory()
  test_joint_alignment_predicts_other_agent_entity()
  test_pstr_coverage_is_forced_to_full_masks()
  print("ns_symbolic tests passed")
