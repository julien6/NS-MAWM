from __future__ import annotations

import numpy as np

from ns_symbolic import BLOCK_TREE, ENTITY_AGENT, ENTITY_ITEM, ENTITY_NONE, ITEM_WOOD
from pstr_scenarios import expected_rules, run_scenario, scenario_ids, scenario_specs
from render_pstr_scenarios import _scenario_world_width, diagram_panels


def test_pstr_scenarios_cover_catalog_order():
  specs = scenario_specs()
  assert list(specs) == scenario_ids()


def test_each_pstr_scenario_triggers_target_rule_and_renders():
  for rule_id in scenario_ids():
    result = run_scenario(rule_id)
    triggered = set()
    for report in result.reports_by_step[1:]:
      triggered.update(report.get("rules", {}))
    assert any(rule in triggered for rule in expected_rules(rule_id))
    assert len(result.frames) == 6
    assert result.frame.ndim == 3
    assert result.frame.shape[2] == 3
    assert result.frame.max() > 0


def test_first_pstr_frame_matches_real_observation_with_full_mask():
  result = run_scenario("PSTR_INDIV_LOCAL_GRID_TRANSLATION", steps=4)
  for agent_id, obs in result.input_observations.items():
    assert np.all(result.masks_by_step[0][agent_id]["grid"])
    assert np.all(result.masks_by_step[0][agent_id]["self"])
  assert "?" not in result.output_ascii_by_step[0]


def test_diagram_left_panel_uses_initial_observation_for_all_scenarios():
  for rule_id in scenario_ids():
    result = run_scenario(rule_id, steps=2)
    left_panel, _ = diagram_panels(result)
    input_frame = result.plain_frames[0]
    world_width = _scenario_world_width()
    panel_width = left_panel.shape[1]
    expected_left = input_frame[:left_panel.shape[0], world_width:world_width + panel_width]
    np.testing.assert_array_equal(left_panel, expected_left)


def test_pickup_pstr_uses_adjacent_item_not_agent_cell():
  result = run_scenario("PSTR_INDIV_PICKUP_ITEM", steps=2)
  initial = result.input_observations["agent_0"]
  symbolic = result.symbolic_next_observations["agent_0"]
  mask = result.symbolic_masks["agent_0"]

  center = initial["grid"].shape[1] // 2
  east = (center, center + 1)
  assert initial["grid"][2, center, center] == ENTITY_AGENT
  assert initial["grid"][2, east[0], east[1]] == ENTITY_ITEM
  assert symbolic["grid"][2, center, center] == ENTITY_AGENT
  assert symbolic["grid"][2, east[0], east[1]] == ENTITY_NONE
  assert mask["grid"][2, east[0], east[1]]
  assert symbolic["self"][2 + ITEM_WOOD] == 2


def test_joint_shared_world_update_removes_harvested_tree_for_other_agent():
  result = run_scenario("PSTR_JOINT_SHARED_WORLD_UPDATE", steps=2)
  initial_agent_1 = result.input_observations["agent_1"]["grid"][1]
  symbolic_agent_1 = result.symbolic_next_observations["agent_1"]["grid"][1]
  mask_agent_1 = result.strict_masks_by_step[-1]["agent_1"]["grid"][1]

  tree_positions = np.argwhere(initial_agent_1 == BLOCK_TREE)
  assert tree_positions.size > 0
  for y, x in tree_positions:
    assert symbolic_agent_1[y, x] != BLOCK_TREE
    assert mask_agent_1[y, x]


def test_joint_multi_agent_collision_blocks_adjacent_agents():
  result = run_scenario("PSTR_JOINT_MULTI_AGENT_COLLISION", steps=2)
  center = result.input_observations["agent_0"]["grid"].shape[1] // 2

  assert result.input_observations["agent_0"]["grid"][2, center, center + 1] == ENTITY_AGENT
  assert result.input_observations["agent_1"]["grid"][2, center, center - 1] == ENTITY_AGENT
  assert result.symbolic_next_observations["agent_0"]["grid"][2, center, center + 1] == ENTITY_AGENT
  assert result.symbolic_next_observations["agent_1"]["grid"][2, center, center - 1] == ENTITY_AGENT
  assert result.strict_masks_by_step[-1]["agent_0"]["grid"][2, center, center + 1]
  assert result.strict_masks_by_step[-1]["agent_1"]["grid"][2, center, center - 1]

  entries = result.report["rules"]["PSTR_JOINT_MULTI_AGENT_COLLISION"]
  assert entries
  assert all(entry["features"] for entry in entries)
