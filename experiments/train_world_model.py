"""Train one WM/NS-MAWM configuration."""

from __future__ import annotations

import argparse

from experiments.config import load_config
from experiments.factory import make_components
from experiments.logging import RunLogger
from experiments.training import collect_replay, seed_everything, train_model_free_policy, train_world_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    config = load_config(args.config, args.overrides)
    logger = RunLogger(config.output_dir)
    for seed in config.seeds[:1]:
        seed_everything(seed)
        env, policy, _wm, model = make_components(config)
        replay = collect_replay(env, policy, config.smoke_steps, seed)
        for row in train_world_model(model, replay, config.train_updates, config.batch_size):
            logger.log({"seed": seed, "baseline_id": config.baseline_id, **row})
        for row in train_model_free_policy(policy, replay, 1, config.batch_size):
            logger.log({"seed": seed, "baseline_id": config.baseline_id, **row})
    logger.flush("train_world_model")
    print(f"wrote {config.output_dir}/train_world_model.csv")


if __name__ == "__main__":
    main()
