from __future__ import annotations


PSTR_K03 = (
  "PSTR_INDIV_STATIC_TERRAIN_SHIFT",
  "PSTR_INDIV_STATIC_BLOCK_SHIFT",
  "PSTR_INDIV_CENTER_AGENT",
  "PSTR_INDIV_BLOCKED_WATER",
  "PSTR_INDIV_BLOCKED_TREE_STONE",
  "PSTR_INDIV_BLOCKED_ENTITY",
)

PSTR_K06 = PSTR_K03 + (
  "PSTR_INDIV_HARVEST_TREE_WOOD",
  "PSTR_INDIV_HARVEST_STONE_PICKAXE",
  "PSTR_INDIV_PICKUP_ITEM",
  "PSTR_INDIV_EAT_APPLE",
  "PSTR_INDIV_CRAFT_PLANK",
  "PSTR_INDIV_CRAFT_STICK",
  "PSTR_INDIV_CRAFT_TOOLS",
  "PSTR_INDIV_HUNGER_COST_KNOWN_COUNTER",
  "PSTR_JOINT_RELATIVE_AGENT_ALIGNMENT",
  "PSTR_JOINT_MAP_FUSION",
  "PSTR_JOINT_GLOBAL_STATIC_QUERY",
  "PSTR_JOINT_MULTI_AGENT_COLLISION",
  "PSTR_JOINT_AGENT_ENTITY_PREDICTION",
  "PSTR_JOINT_SHARED_WORLD_UPDATE",
  "PSTR_JOINT_SHARED_ITEM_UPDATE",
)

PSTR_PROFILES = {
  "neural_k0.0": (),
  "ns_k0.3": PSTR_K03,
  "ns_k0.6": PSTR_K06,
}


def profile_name_from_baseline(baseline_id: str) -> str:
  text = str(baseline_id)
  if "B10" in text or "neural_k0.0" in text or "_k0.0" in text:
    return "neural_k0.0"
  if "B26" in text or "_k0.6" in text:
    return "ns_k0.6"
  if "B25" in text or "_k0.3" in text:
    return "ns_k0.3"
  return "neural_k0.0"


def profile_rules(profile_name: str) -> tuple[str, ...]:
  return PSTR_PROFILES.get(str(profile_name), ())


def rules_for_baseline(baseline_id: str) -> tuple[str, ...]:
  return profile_rules(profile_name_from_baseline(baseline_id))


def normalize_rule_override(rules) -> tuple[str, ...] | None:
  if not rules:
    return None
  if isinstance(rules, str):
    rules = [rules]
  normalized = []
  for rule in rules:
    normalized.extend(part.strip() for part in str(rule).split(","))
  normalized = [rule for rule in normalized if rule]
  return tuple(normalized) or None


def active_rules_for_baseline(baseline_id: str, override_rules=None) -> tuple[str, ...]:
  override = normalize_rule_override(override_rules)
  if override is not None:
    return override
  return rules_for_baseline(baseline_id)


def rules_to_csv(rules) -> str:
  return ",".join(rules or ())
