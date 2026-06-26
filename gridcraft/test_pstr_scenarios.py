from __future__ import annotations

import numpy as np

from pstr_scenarios import expected_rules, run_scenario, scenario_ids, scenario_specs
from render_pstr_scenarios import diagram_panels


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
    world_width = input_frame.shape[0]
    panel_width = left_panel.shape[1]
    expected_left = input_frame[:left_panel.shape[0], world_width:world_width + panel_width]
    np.testing.assert_array_equal(left_panel, expected_left)
