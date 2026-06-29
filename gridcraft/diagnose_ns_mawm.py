from __future__ import annotations

import argparse
import json
from copy import deepcopy

import numpy as np

from ns_symbolic import (
    ACTION_EAT,
    ACTION_HARVEST,
    ACTION_MOVE_E,
    BLOCK_EMPTY,
    BLOCK_TREE,
    ENTITY_AGENT,
    ENTITY_NONE,
    ITEM_APPLE,
    ITEM_PLANK,
    ITEM_WOOD,
    TERRAIN_WATER,
    apply_symbolic_projection,
    compare_with_symbolic,
)


DEFAULT_RULES = [
    "PSTR_INDIV_STATIC_TERRAIN_SHIFT",
    "PSTR_INDIV_STATIC_BLOCK_SHIFT",
    "PSTR_INDIV_CENTER_AGENT",
    "PSTR_INDIV_BLOCKED_WATER",
    "PSTR_INDIV_HARVEST_TREE_WOOD",
    "PSTR_INDIV_EAT_APPLE",
    "PSTR_INDIV_CRAFT_PLANK",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rules", nargs="*", default=DEFAULT_RULES)
    parser.add_argument("--coverage", type=float, default=1.0)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero when any rule with determinable features has non-zero post RVR.")
    args = parser.parse_args()

    rows = [diagnose_rule(rule_id, args.coverage) for rule_id in expand_rules(args.rules)]
    if args.json:
        print(json.dumps(rows, indent=2, sort_keys=True))
    else:
        print_table(rows)
    if args.strict:
        failures = [
            row
            for row in rows
            if row["determinable_count"] > 0
            and (row["projection_rvr_post"] != 0.0 or row["residual_rvr_post"] != 0.0)
        ]
        if failures:
            raise SystemExit(1)


def expand_rules(rules):
    expanded = []
    for rule in rules:
        expanded.extend(part.strip() for part in str(rule).split(","))
    return [rule for rule in expanded if rule]


def diagnose_rule(rule_id: str, coverage: float) -> dict:
    current, action, raw_predicted, memory = scenario(rule_id)
    enabled = [rule_id]
    pre = compare_with_symbolic(
        raw_predicted,
        current,
        action,
        coverage=coverage,
        memory=memory,
        enabled_pstr_rules=enabled,
    )
    projected, _ = apply_symbolic_projection(
        raw_predicted,
        current,
        action,
        "projection",
        coverage=coverage,
        memory=memory,
        enabled_pstr_rules=enabled,
    )
    residual, _ = apply_symbolic_projection(
        raw_predicted,
        current,
        action,
        "residual",
        coverage=coverage,
        memory=memory,
        enabled_pstr_rules=enabled,
    )
    post_projection = compare_with_symbolic(
        projected,
        current,
        action,
        coverage=coverage,
        memory=memory,
        enabled_pstr_rules=enabled,
    )
    post_residual = compare_with_symbolic(
        residual,
        current,
        action,
        coverage=coverage,
        memory=memory,
        enabled_pstr_rules=enabled,
    )
    determinable_count = float(pre.get(f"determinable_count/{rule_id}", pre.get("determinable_count", 0.0)))
    return {
        "rule_id": rule_id,
        "coverage": float(coverage),
        "determinable_count": determinable_count,
        "rvr_pre": float(pre.get(f"rvr/{rule_id}", pre.get("rvr", 0.0))),
        "projection_rvr_post": float(post_projection.get(f"rvr/{rule_id}", post_projection.get("rvr", 0.0))),
        "residual_rvr_post": float(post_residual.get(f"rvr/{rule_id}", post_residual.get("rvr", 0.0))),
        "status": status_for(pre, post_projection, post_residual, rule_id),
    }


def scenario(rule_id: str):
    obs = make_obs()
    predicted = deepcopy(obs)
    action = ACTION_MOVE_E
    memory = None

    if rule_id == "PSTR_INDIV_STATIC_TERRAIN_SHIFT":
        obs["grid"][0, 3, 4] = 2
        predicted["grid"][0, 3, 3] = 0
    elif rule_id == "PSTR_INDIV_STATIC_BLOCK_SHIFT":
        obs["grid"][1, 2, 4] = BLOCK_TREE
        predicted["grid"][1, 2, 3] = BLOCK_EMPTY
    elif rule_id == "PSTR_INDIV_CENTER_AGENT":
        action = 0
        predicted["grid"][2, 3, 3] = ENTITY_NONE
    elif rule_id == "PSTR_INDIV_BLOCKED_WATER":
        obs["grid"][0, 3, 4] = TERRAIN_WATER
        predicted["grid"][0, 3, 3] = 2
    elif rule_id == "PSTR_INDIV_HARVEST_TREE_WOOD":
        action = ACTION_HARVEST
        obs["grid"][1, 3, 2] = BLOCK_TREE
        obs["self"][2 + ITEM_WOOD] = 2
        predicted = deepcopy(obs)
        predicted["grid"][1, 3, 2] = BLOCK_TREE
        predicted["self"][2 + ITEM_WOOD] = 2
    elif rule_id == "PSTR_INDIV_EAT_APPLE":
        action = ACTION_EAT
        obs["self"][1] = 10
        obs["self"][2 + ITEM_APPLE] = 1
        predicted = deepcopy(obs)
        predicted["self"][1] = 10
        predicted["self"][2 + ITEM_APPLE] = 1
    elif rule_id == "PSTR_INDIV_CRAFT_PLANK":
        action = 9
        obs["self"][2 + ITEM_WOOD] = 2
        predicted = deepcopy(obs)
        predicted["self"][2 + ITEM_WOOD] = 2
        predicted["self"][2 + ITEM_PLANK] = 0
    else:
        raise ValueError(f"no diagnostic scenario is defined for {rule_id}")
    return obs, action, predicted, memory


def make_obs():
    grid = np.zeros((3, 7, 7), dtype=np.int8)
    grid[2, 3, 3] = ENTITY_AGENT
    self_vec = np.zeros((11,), dtype=np.int16)
    self_vec[0] = 20
    self_vec[1] = 20
    return {"grid": grid, "self": self_vec}


def status_for(pre, post_projection, post_residual, rule_id):
    count = float(pre.get(f"determinable_count/{rule_id}", pre.get("determinable_count", 0.0)))
    if count <= 0:
        return "no_determinable_features"
    if (
        float(post_projection.get(f"rvr/{rule_id}", post_projection.get("rvr", 0.0))) == 0.0
        and float(post_residual.get(f"rvr/{rule_id}", post_residual.get("rvr", 0.0))) == 0.0
    ):
        return "pass"
    return "fail"


def print_table(rows):
    headers = ["PSTR", "count", "pre", "projection_post", "residual_post", "status"]
    print(f"{headers[0]:42s} {headers[1]:>8s} {headers[2]:>8s} {headers[3]:>16s} {headers[4]:>14s} {headers[5]}")
    for row in rows:
        print(
            f"{row['rule_id']:42s} "
            f"{row['determinable_count']:8.0f} "
            f"{row['rvr_pre']:8.4f} "
            f"{row['projection_rvr_post']:16.4f} "
            f"{row['residual_rvr_post']:14.4f} "
            f"{row['status']}"
        )


if __name__ == "__main__":
    main()
