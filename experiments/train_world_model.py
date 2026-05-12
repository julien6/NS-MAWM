"""Train one WM/NS-MAWM configuration."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone

from experiments.config import load_config
from experiments.factory import make_components
from experiments.logging import RunLogger
from experiments.training import collect_replay, git_commit_hash, seed_everything, train_model_free_policy, train_world_model


def identity(config, seed: int, variant_id: str) -> dict[str, object]:
    return {
        "baseline_id": config.baseline_id,
        "seed": seed,
        "environment": config.environment,
        "variant_id": variant_id,
        "train_regime": config.train_regime,
        "eval_regime": config.eval_regime,
        "wm": config.world_model,
        "policy": config.policy,
        "strategy": config.strategy,
        "coverage": config.coverage,
        "mode": config.mode,
        "comparison_arm": config.comparison_arm,
        "reference_coverage": config.reference_coverage,
        "paired_baseline_id": config.paired_baseline_id,
        "commit_hash": git_commit_hash(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


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
        if model is not None:
            for row in train_world_model(
                model,
                replay,
                config.active_train_updates,
                config.batch_size,
                lr=config.wm_learning_rate,
                sequence_length=config.sequence_length,
            ):
                logger.log({**row_id, **row})
        for row in train_model_free_policy(policy, replay, 1, config.batch_size, learning_rate=config.policy_learning_rate):
            reward = float(replay.sample(min(config.batch_size, len(replay))).reward.mean())
            resource = max(float(row.get("policy_total_loss", 0.0)), 1e-6)
            logger.log({**row_id, "training_real_reward": reward, "reward_per_resource": reward / resource, "generalization_gap": 0.0, **row})
    logger.flush("train_world_model")
    print(f"wrote {config.output_dir}/train_world_model.csv")


if __name__ == "__main__":
    main()
