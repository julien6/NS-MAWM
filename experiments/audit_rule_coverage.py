"""Audit symbolic rule coverage."""

from __future__ import annotations

import argparse

from env_adapters.rules import achieved_coverage, coverage_sweep_rules, nominal_coverage_target, rule_inventory, rules_for_coverage
from experiments.config import load_config
from experiments.factory import make_components
from ns_mawm.rules import SymbolicWorldModel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    config = load_config(args.config, args.overrides)
    env, *_ = make_components(config)
    rules = rules_for_coverage(config.environment, env.schema, config.coverage, n_agents=env.n_agents)
    model = SymbolicWorldModel(rules)
    inventory = rule_inventory(config.environment, env.schema, n_agents=env.n_agents)
    sweep = coverage_sweep_rules(config.environment, env.schema, n_agents=env.n_agents)
    print(
        {
            **model.audit(env.schema).__dict__,
            "requested_coverage": config.coverage,
            "achieved_coverage": achieved_coverage(env.schema, rules),
            "nominal_target": nominal_coverage_target(config.environment),
        }
    )
    print({"available_rules": [rule.rule_id for rule in inventory]})
    print({"coverage_sweep": {target: achieved_coverage(env.schema, selected) for target, selected in sweep.items()}})


if __name__ == "__main__":
    main()
