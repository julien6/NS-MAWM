"""Evaluate open-loop WM rollouts."""

from __future__ import annotations

import argparse

from experiments.config import load_config
from experiments.evaluation import open_loop_rollout
from experiments.factory import make_components
from experiments.logging import RunLogger
from experiments.training import collect_replay, seed_everything, train_world_model


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
        train_world_model(model, replay, config.train_updates, config.batch_size)
        batch = replay.sample(min(config.batch_size, len(replay)))
        for row in open_loop_rollout(model, batch, env.schema, config.horizons):
            logger.log({"seed": seed, "baseline_id": config.baseline_id, **row.__dict__})
    logger.flush("eval_rollout")
    print(f"wrote {config.output_dir}/eval_rollout.csv")


if __name__ == "__main__":
    main()
