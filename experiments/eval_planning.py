"""Evaluate random-shooting planning with NS-MAWM rollouts."""

from __future__ import annotations

import argparse

from experiments.config import load_config
from experiments.evaluation import cem_plan, random_shooting_plan
from experiments.factory import make_components
from experiments.logging import RunLogger
from experiments.train_world_model import identity
from experiments.training import collect_replay, seed_everything, train_world_model


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
        if model is None:
            logger.log({"baseline_id": config.baseline_id, "seed": seed, "message": "model_free_baseline_has_no_planning_model"})
            continue
        train_split = "SV" if config.train_regime == "SV" else "KV"
        train_variant = env.make_variants(train_split)[0]
        replay = collect_replay(env, policy, config.active_steps, seed, variant=train_variant)
        train_world_model(
            model,
            replay,
            config.active_train_updates,
            config.batch_size,
            lr=config.wm_learning_rate,
            sequence_length=config.sequence_length,
        )
        batch = replay.sample(1)
        candidates = config.planner_candidates if not config.is_smoke else 16
        horizon = config.planner_horizon if not config.is_smoke else 3
        action = random_shooting_plan(model, batch.obs, env.n_agents, env.action_dim, candidates=candidates, horizon=horizon)
        cem_action = cem_plan(model, batch.obs, env.n_agents, env.action_dim, candidates=candidates, horizon=horizon)
        logger.log(
            {
                **identity(config, seed, getattr(env, "variant_id", train_variant.variant_id)),
                "planner_horizon": horizon,
                "planner_candidates": candidates,
                "random_shooting_action_dim": int(action.shape[-1]),
                "cem_action_dim": int(cem_action.shape[-1]),
                "planning_model_available": 1,
            }
        )
    logger.flush("eval_planning")
    print(f"wrote {config.output_dir}/eval_planning.csv")


if __name__ == "__main__":
    main()
