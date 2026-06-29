from pstr_profiles import active_rules_for_baseline, profile_name_from_baseline


def test_baseline_profiles_are_resolved_from_kappa_names():
  assert profile_name_from_baseline("B10_neural_k0.0") == "neural_k0.0"
  assert profile_name_from_baseline("B25_projection_k0.3") == "ns_k0.3"
  assert profile_name_from_baseline("B26_residual_k0.6") == "ns_k0.6"


def test_k06_profile_is_strict_superset_of_k03():
  k03 = set(active_rules_for_baseline("B25_projection_k0.3"))
  k06 = set(active_rules_for_baseline("B26_projection_k0.6"))
  assert k03
  assert k03 < k06


def test_override_rules_take_precedence_over_profile():
  rules = active_rules_for_baseline(
    "B26_projection_k0.6",
    ["PSTR_INDIV_STATIC_TERRAIN_SHIFT"],
  )
  assert rules == ("PSTR_INDIV_STATIC_TERRAIN_SHIFT",)
