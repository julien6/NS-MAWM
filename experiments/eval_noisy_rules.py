"""Evaluate robustness to perturbed symbolic rule outputs."""

from __future__ import annotations

import argparse

from ns_mawm.rules import NoisyRule, SymbolicWorldModel

from experiments.config import load_config
from experiments.evaluation import open_loop_rollout
from experiments.factory import make_components
from experiments.logging import RunLogger
from experiments.training import collect_replay, seed_everything, train_world_model
from experiments.train_world_model import identity


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--noise-std", type=float, default=0.05)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    config = load_config(args.config, args.overrides)
    logger = RunLogger(config.output_dir)
    for seed in config.active_seeds:
        seed_everything(seed)
        env, policy, _wm, model = make_components(config)
        if model is None:
            continue
        replay = collect_replay(env, policy, config.active_steps, seed)
        train_world_model(
            model,
            replay,
            config.active_train_updates,
            config.batch_size,
            lr=config.wm_learning_rate,
            sequence_length=config.sequence_length,
        )
        model.symbolic_model = SymbolicWorldModel([NoisyRule(rule, args.noise_std, seed) for rule in model.symbolic_model.rules])
        horizon = max(config.active_horizons)
        batch = replay.sample_sequence(1, horizon) if len(replay) >= horizon else replay.sample(min(config.batch_size, len(replay)))
        for row in open_loop_rollout(model, batch, env.schema, config.active_horizons):
            values = dict(row.__dict__)
            for family, rvr in values.pop("rvr_by_family").items():
                values[f"rvr_family/{family}"] = rvr
            logger.log({**identity(config, seed, getattr(env, "variant_id", "default")), "rule_noise_std": args.noise_std, **values})
    logger.flush("eval_noisy_rules")
    print(f"wrote {config.output_dir}/eval_noisy_rules.csv")


if __name__ == "__main__":
    main()
