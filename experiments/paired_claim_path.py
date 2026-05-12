"""Run paired baseline comparisons for one environment."""

from __future__ import annotations

import argparse
import shutil
from dataclasses import replace
from pathlib import Path

import pandas as pd

from env_adapters.rules import nominal_coverage_target, rules_for_coverage
from ns_mawm.rules import SymbolicWorldModel

from experiments.config import ExperimentConfig, load_config
from experiments.evaluation import open_loop_rollout
from experiments.factory import make_components
from experiments.logging import RunLogger
from experiments.reproduction_check import validate_paired_comparison
from experiments.train_world_model import identity
from experiments.training import collect_replay, seed_everything, train_model_free_policy, train_world_model


def _parse_arm_baselines(value: str) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for item in value.split(","):
        if not item.strip():
            continue
        arm, baseline = item.split(":", 1)
        mapping[arm.strip()] = baseline.strip()
    return mapping


def _run_config(config: ExperimentConfig, logger: RunLogger) -> None:
    env = None
    try:
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
                    logger.log({**row_id, **row, "run_type": "train_world_model"})
            for row in train_model_free_policy(policy, replay, 1, config.batch_size, learning_rate=config.policy_learning_rate):
                reward = float(replay.sample(min(config.batch_size, len(replay))).reward.mean())
                resource = max(float(row.get("policy_total_loss", 0.0)), 1e-6)
                logger.log(
                    {
                        **row_id,
                        "run_type": "policy_update",
                        "training_real_reward": reward,
                        "reward_per_resource": reward / resource,
                        "generalization_gap": 0.0,
                        **row,
                    }
                )
            if model is None:
                continue
            eval_variant = env.make_variants(config.eval_regime)[0]
            eval_replay = collect_replay(env, policy, config.active_steps, seed, variant=eval_variant)
            horizon = max(config.active_horizons)
            batch = eval_replay.sample_sequence(1, horizon) if len(eval_replay) >= horizon else eval_replay.sample(min(config.batch_size, len(eval_replay)))
            reference_coverage = config.reference_coverage if config.reference_coverage is not None else nominal_coverage_target(config.environment)
            reference_rules = rules_for_coverage(
                config.environment,
                env.schema,
                reference_coverage,
                n_agents=env.n_agents,
                width=getattr(env, "width", 7),
                height=getattr(env, "height", 7),
            )
            reference_symbolic = SymbolicWorldModel(reference_rules, conflict_policy="last")
            eval_row_id = {**identity(config, seed, getattr(env, "variant_id", eval_variant.variant_id)), "run_type": "eval_rollout"}
            for row in open_loop_rollout(model, batch, env.schema, config.active_horizons, reference_symbolic_model=reference_symbolic):
                values = dict(row.__dict__)
                for family, rvr in values.pop("rvr_by_family").items():
                    values[f"rvr_family/{family}"] = rvr
                logger.log({**eval_row_id, **values})
            if hasattr(env, "close"):
                env.close()
                env = None
    finally:
        if env is not None and hasattr(env, "close"):
            env.close()


def run_paired_claim_path(
    base: ExperimentConfig,
    *,
    paired_baseline: str,
    arm_baselines: dict[str, str],
    fresh: bool = False,
) -> Path:
    output_dir = Path(base.output_dir)
    if fresh and output_dir.exists():
        shutil.rmtree(output_dir)
    logger = RunLogger(str(output_dir))
    reference_coverage = base.reference_coverage if base.reference_coverage is not None else nominal_coverage_target(base.environment)
    configs = [
        replace(
            base,
            baseline_id=paired_baseline,
            strategy="none",
            coverage=0.0,
            comparison_arm="neural",
            paired_baseline_id=None,
            reference_coverage=reference_coverage,
        )
    ]
    configs.extend(
        replace(
            base,
            baseline_id=baseline_id,
            strategy=arm,
            coverage=0.3,
            comparison_arm=arm,
            paired_baseline_id=paired_baseline,
            reference_coverage=reference_coverage,
        )
        for arm, baseline_id in arm_baselines.items()
    )
    for config in configs:
        config.validate()
        _run_config(config, logger)
    stem = f"{base.environment}_claim_path"
    logger.flush(stem)
    frame = pd.read_csv(output_dir / f"{stem}.csv")
    frame[frame["run_type"].isin(["train_world_model", "policy_update"])].to_csv(output_dir / "train_world_model.csv", index=False)
    frame[frame["run_type"] == "eval_rollout"].to_csv(output_dir / "eval_rollout.csv", index=False)
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--paired-baseline", required=True)
    parser.add_argument("--arm-baselines", required=True, help="comma list such as projection:B25,residual:B26")
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    base = load_config(args.config, args.overrides)
    arm_baselines = _parse_arm_baselines(args.arm_baselines)
    output_dir = run_paired_claim_path(base, paired_baseline=args.paired_baseline, arm_baselines=arm_baselines, fresh=args.fresh)
    result = validate_paired_comparison(
        output_dir,
        paired_baseline=args.paired_baseline,
        comparison_baselines=tuple(arm_baselines.values()),
        comparison_arms=tuple(arm_baselines),
        required_seeds=len(base.active_seeds),
        required_horizons=base.active_horizons,
        require_directional_rvr_improvement=tuple(arm for arm in arm_baselines if arm in {"projection", "residual"}),
    )
    if not result.ok:
        raise SystemExit(f"paired comparison failed: {result}")
    print(f"wrote paired comparison logs to {output_dir}")


if __name__ == "__main__":
    main()
