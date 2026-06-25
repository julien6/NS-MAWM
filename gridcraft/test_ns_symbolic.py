import numpy as np

from ns_symbolic import (
  ACTION_HARVEST,
  ACTION_MOVE_E,
  BLOCK_TREE,
  ENTITY_AGENT,
  ITEM_PLANK,
  ITEM_WOOD,
  symbolic_joint_transition,
  symbolic_transition,
)


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


def test_harvest_tree_predicts_wood_not_apple():
  obs = make_obs()
  obs["grid"][1, 3, 2] = BLOCK_TREE
  obs["self"][2 + ITEM_WOOD] = 2
  symbolic, mask = symbolic_transition(obs, ACTION_HARVEST, coverage=1.0)
  assert symbolic["grid"][1, 3, 2] == 0
  assert mask["grid"][1, 3, 2]
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


def test_coverage_zero_masks_everything():
  obs = make_obs()
  symbolic, mask = symbolic_transition(obs, 0, coverage=0.0)
  assert not np.any(mask["grid"])
  assert not np.any(mask["self"])


if __name__ == "__main__":
  test_move_shift_masks_static_planes()
  test_harvest_tree_predicts_wood_not_apple()
  test_craft_plank_updates_inventory()
  test_joint_alignment_predicts_other_agent_entity()
  test_coverage_zero_masks_everything()
  print("ns_symbolic tests passed")
