"""Evaluate random-shooting planning with NS-MAWM rollouts."""

from __future__ import annotations

import argparse

from experiments.config import load_config
from experiments.evaluation import random_shooting_plan
from experiments.factory import make_components
from experiments.training import collect_replay, seed_everything


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    config = load_config(args.config, args.overrides)
    seed_everything(config.seeds[0])
    env, policy, _wm, model = make_components(config)
    replay = collect_replay(env, policy, config.smoke_steps, config.seeds[0])
    batch = replay.sample(1)
    action = random_shooting_plan(model, batch.obs, env.n_agents, env.action_dim)
    print({"planned_action_shape": tuple(action.shape)})


if __name__ == "__main__":
    main()
