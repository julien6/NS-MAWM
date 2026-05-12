"""Evaluate open-loop WM rollouts."""

from __future__ import annotations

import argparse

from experiments.config import load_config
from experiments.evaluation import open_loop_rollout
from experiments.factory import make_components
from experiments.logging import RunLogger
from experiments.training import collect_replay, seed_everything, train_world_model
from experiments.train_world_model import identity
from env_adapters.rules import nominal_coverage_target, rules_for_coverage
from ns_mawm.rules import SymbolicWorldModel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    config = load_config(args.config, args.overrides)
    logger = RunLogger(config.output_dir)
    for seed in config.active_seeds:
        seed_everything(seed)
        env, policy, _wm, model = make_components(config)
        train_split = "SV" if config.train_regime == "SV" else "KV"
        train_variant = env.make_variants(train_split)[0]
        replay = collect_replay(env, policy, config.active_steps, seed, variant=train_variant)
        row_id = identity(config, seed, getattr(env, "variant_id", train_variant.variant_id))
        if model is None:
            logger.log({**row_id, "message": "model_free_baseline_has_no_rollout_metrics"})
            continue
        train_world_model(
            model,
            replay,
            config.active_train_updates,
            config.batch_size,
            lr=config.wm_learning_rate,
            sequence_length=config.sequence_length,
        )
        eval_variant = env.make_variants(config.eval_regime)[0]
        eval_replay = collect_replay(env, policy, config.active_steps, seed, variant=eval_variant)
        horizon = max(config.active_horizons)
        if len(eval_replay) >= horizon:
            batch = eval_replay.sample_sequence(1, horizon)
        else:
            batch = eval_replay.sample(min(config.batch_size, len(eval_replay)))
        eval_row_id = {**row_id, "variant_id": getattr(env, "variant_id", eval_variant.variant_id)}
        eval_coverage = config.reference_coverage if config.reference_coverage is not None else max(config.coverage, nominal_coverage_target(config.environment))
        reference_rules = rules_for_coverage(
            config.environment,
            env.schema,
            eval_coverage,
            n_agents=env.n_agents,
            width=getattr(env, "width", 7),
            height=getattr(env, "height", 7),
        )
        reference_symbolic = SymbolicWorldModel(reference_rules, conflict_policy="last")
        for row in open_loop_rollout(model, batch, env.schema, config.active_horizons, reference_symbolic_model=reference_symbolic):
            values = dict(row.__dict__)
            for family, rvr in values.pop("rvr_by_family").items():
                values[f"rvr_family/{family}"] = rvr
            logger.log({**eval_row_id, **values})
    logger.flush("eval_rollout")
    print(f"wrote {config.output_dir}/eval_rollout.csv")


if __name__ == "__main__":
    main()
