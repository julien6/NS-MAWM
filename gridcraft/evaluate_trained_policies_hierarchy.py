from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch
from torchrl.envs.utils import ExplorationType, set_exploration_type

ROOT = Path(__file__).resolve().parents[1]
BENCHMARL_DIR = Path(os.environ.get("BENCHMARL_DIR", ROOT / "BenchMARL"))
sys.path.insert(0, str(BENCHMARL_DIR))
sys.path.insert(0, str(ROOT / "vGridcraft"))
sys.path.insert(0, str(ROOT / "gridcraft"))

from benchmarl.experiment import Experiment  # noqa: E402
from run_benchmarl_mappo_gridcraft import (  # noqa: E402
    ACTION_NAMES,
    hierarchy_metrics_from_tensordicts,
)


MODE_TO_EXPLORATION = {
    "deterministic": ExplorationType.DETERMINISTIC,
    "mode": ExplorationType.MODE,
    "random": ExplorationType.RANDOM,
    "sampled": ExplorationType.RANDOM,
    "mean": ExplorationType.MEAN,
    "median": ExplorationType.MEDIAN,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a trained BenchMARL Gridcraft policy with multiple action "
            "selection modes and log task-hierarchy metrics."
        )
    )
    parser.add_argument("--checkpoint", required=True, help="BenchMARL checkpoint_*.pt")
    parser.add_argument("--baseline-id", default="B00_model-free-control")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num-agents", type=int, default=3)
    parser.add_argument("--episodes", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument(
        "--modes",
        default="deterministic,mode,sampled",
        help="Comma-separated modes: deterministic,mode,sampled,random,mean,median",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out-dir", default="policy_hierarchy_eval")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "ns-mawm-gridcraft"))
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument("--wandb-group", default=None)
    return parser.parse_args()


def tensor_sum(rollouts: list[Any], key: tuple[str, ...]) -> float | None:
    values = []
    for rollout in rollouts:
        try:
            values.append(rollout.get(key).detach().float())
        except (KeyError, AttributeError):
            pass
    if not values:
        return None
    return float(torch.cat([value.reshape(-1) for value in values]).sum().cpu())


def reward_stats(rollouts: list[Any]) -> dict[str, float]:
    episode_returns = []
    episode_lengths = []
    for rollout in rollouts:
        reward = None
        for key in (("next", "agents", "reward"), ("agents", "reward"), ("next", "reward")):
            try:
                reward = rollout.get(key).detach().float()
                break
            except (KeyError, AttributeError):
                continue
        if reward is None:
            continue
        episode_returns.append(float(reward.sum().cpu()))
        episode_lengths.append(float(reward.shape[0]))
    if not episode_returns:
        return {}
    returns = torch.tensor(episode_returns, dtype=torch.float32)
    lengths = torch.tensor(episode_lengths, dtype=torch.float32)
    return {
        "episode_return_mean": float(returns.mean().item()),
        "episode_return_min": float(returns.min().item()),
        "episode_return_max": float(returns.max().item()),
        "episode_return_std": float(returns.std(unbiased=False).item()) if len(episode_returns) > 1 else 0.0,
        "episode_length_mean": float(lengths.mean().item()),
    }


def compact_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in metrics.items():
        if key.startswith("hierarchy/evaluation_"):
            out[key.replace("hierarchy/evaluation_", "")] = value
        else:
            out[key] = value
    return out


def evaluate_mode(experiment: Experiment, mode: str, episodes: int, max_steps: int) -> dict[str, Any]:
    exploration_type = MODE_TO_EXPLORATION[mode]
    experiment.test_env.eval()
    experiment.policy.eval()
    with torch.no_grad(), set_exploration_type(exploration_type):
        rollout = experiment.test_env.rollout(
            max_steps=max_steps,
            policy=experiment.policy,
            auto_cast_to_device=True,
            break_when_any_done=False,
        )
    rollouts = list(rollout.unbind(0))[:episodes]
    metrics = compact_metrics(hierarchy_metrics_from_tensordicts(rollouts, prefix="evaluation"))
    metrics.update(reward_stats(rollouts))
    metrics["mode"] = mode
    metrics["episodes"] = len(rollouts)
    metrics["max_steps"] = max_steps
    metrics["exploration_type"] = str(exploration_type.value)
    return metrics


def write_outputs(rows: list[dict[str, Any]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "policy_hierarchy_eval_summary.json").open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, sort_keys=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with (out_dir / "policy_hierarchy_eval_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def log_wandb(args: argparse.Namespace, rows: list[dict[str, Any]], out_dir: Path) -> None:
    import wandb

    run = wandb.init(
        project=args.wandb_project,
        name=args.wandb_name or f"{args.baseline_id}_policy_hierarchy_eval_seed{args.seed}",
        group=args.wandb_group,
        config={
            "baseline_id": args.baseline_id,
            "seed": args.seed,
            "num_agents": args.num_agents,
            "checkpoint": args.checkpoint,
            "episodes": args.episodes,
            "max_steps": args.max_steps,
            "modes": args.modes,
        },
    )
    table = wandb.Table(columns=["mode", "metric", "value"])
    for row in rows:
        mode = row["mode"]
        for key, value in sorted(row.items()):
            if key == "mode":
                continue
            if isinstance(value, (int, float)):
                table.add_data(mode, key, value)
        scalar_payload = {
            f"Policy hierarchy evaluation/{mode}/{key}": value
            for key, value in row.items()
            if key != "mode" and isinstance(value, (int, float))
        }
        wandb.log(scalar_payload)
    wandb.log({"Policy hierarchy evaluation/summary": table})
    wandb.save(str(out_dir / "policy_hierarchy_eval_summary.json"))
    wandb.save(str(out_dir / "policy_hierarchy_eval_summary.csv"))
    run.finish()


def main() -> None:
    args = parse_args()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]
    unknown_modes = sorted(set(modes) - set(MODE_TO_EXPLORATION))
    if unknown_modes:
        raise ValueError(f"Unknown evaluation modes: {unknown_modes}")

    experiment_patch = {
        "sampling_device": args.device,
        "train_device": args.device,
        "buffer_device": args.device,
        "evaluation_episodes": args.episodes,
        "max_n_iters": 0,
        "render": False,
        "loggers": ["csv"],
    }
    experiment = Experiment.reload_from_file(str(checkpoint), experiment_patch=experiment_patch)

    rows = []
    for mode in modes:
        print(f"[policy-eval] checkpoint={checkpoint} mode={mode} episodes={args.episodes}", flush=True)
        row = evaluate_mode(experiment, mode=mode, episodes=args.episodes, max_steps=args.max_steps)
        row.update({
            "baseline_id": args.baseline_id,
            "seed": args.seed,
            "num_agents": args.num_agents,
            "checkpoint": str(checkpoint),
        })
        rows.append(row)
        print(json.dumps(row, indent=2, sort_keys=True), flush=True)

    out_dir = Path(args.out_dir).expanduser().resolve()
    write_outputs(rows, out_dir)
    print(f"[policy-eval] wrote {out_dir / 'policy_hierarchy_eval_summary.json'}", flush=True)
    if args.wandb:
        log_wandb(args, rows, out_dir)


if __name__ == "__main__":
    main()
