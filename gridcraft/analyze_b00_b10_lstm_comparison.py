from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


DEFAULT_METRICS = (
    "episode_return_mean",
    "task_level_max",
    "event_count_craft_stone_tool",
    "event_count_tool_equipped",
    "event_count_mob_kill_armed",
    "event_count_mob_kill_unarmed",
    "dominant_action_rate",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate B00/B10 MASAC/MAMBPO+LSTM hierarchy evaluations."
    )
    parser.add_argument(
        "--eval-root",
        action="append",
        default=[],
        help=(
            "Root containing policy_hierarchy_eval_summary.json files. Can be repeated. "
            "Defaults to common Gridcraft evaluation output directories."
        ),
    )
    parser.add_argument(
        "--marl-summary-root",
        action="append",
        default=[],
        help="Root containing *_marl_summary.json files. Can be repeated.",
    )
    parser.add_argument("--out-dir", default="analysis_b00_b10_lstm_comparison")
    parser.add_argument("--baselines", default="B00_model-free-control,B10_neural_k0.0")
    parser.add_argument("--main-mode", default="temp_0.5")
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def find_policy_eval_files(roots: list[str]) -> list[Path]:
    if not roots:
        roots = [
            "policy_hierarchy_eval",
            "policy_hierarchy_eval_consolidated",
            "runs_benchmarl_marl_hpo",
            "runs_benchmarl",
        ]
    files: list[Path] = []
    seen = set()
    for root_text in roots:
        root = Path(root_text)
        if not root.exists():
            continue
        for path in root.rglob("policy_hierarchy_eval_summary.json"):
            resolved = path.resolve()
            if resolved not in seen:
                seen.add(resolved)
                files.append(path)
    return sorted(files)


def find_marl_summary_files(roots: list[str]) -> list[Path]:
    if not roots:
        roots = ["runs_benchmarl", "runs_benchmarl_marl_hpo"]
    files: list[Path] = []
    seen = set()
    for root_text in roots:
        root = Path(root_text)
        if not root.exists():
            continue
        for path in root.rglob("*_marl_summary.json"):
            resolved = path.resolve()
            if resolved not in seen:
                seen.add(resolved)
                files.append(path)
    return sorted(files)


def as_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def collect_policy_rows(files: list[Path], baselines: set[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in files:
        payload = read_json(path)
        if isinstance(payload, dict):
            payload = [payload]
        if not isinstance(payload, list):
            continue
        for row in payload:
            if not isinstance(row, dict):
                continue
            baseline = row.get("baseline_id")
            if baseline not in baselines:
                continue
            copied = dict(row)
            copied["source_file"] = str(path)
            rows.append(copied)
    return rows


def collect_marl_rows(files: list[Path], baselines: set[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in files:
        payload = read_json(path)
        config = payload.get("config", {}) if isinstance(payload, dict) else {}
        metrics = payload.get("metrics", {}) if isinstance(payload, dict) else {}
        baseline = config.get("baseline_id") or payload.get("baseline_id")
        if baseline not in baselines:
            continue
        row = {
            "baseline_id": baseline,
            "seed": config.get("seed") or payload.get("seed"),
            "source_file": str(path),
        }
        for key in (
            "MARL Evaluation/eval_real_reward_auc",
            "MARL Evaluation/eval_real_reward_curve_mean",
            "MARL Evaluation/eval_real_reward_stability_std",
            "MARL Evaluation/eval_real_reward_point_count",
            "MARL Training/imagination_external_world_model_used",
            "MARL Training/training_imagined_reward",
            "MARL Evaluation/eval_imagined_reward",
            "MARL Evaluation/real_imagined_reward_gap",
        ):
            if key in metrics:
                row[key] = metrics[key]
        rows.append(row)
    return rows


def aggregate(rows: list[dict[str, Any]], group_keys: tuple[str, ...], metrics: tuple[str, ...]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(key) for key in group_keys)].append(row)
    out = []
    for group, group_rows in sorted(grouped.items(), key=lambda item: tuple(str(x) for x in item[0])):
        payload = {key: value for key, value in zip(group_keys, group)}
        payload["count"] = len(group_rows)
        for metric in metrics:
            values = [as_float(row.get(metric)) for row in group_rows]
            values = [value for value in values if value is not None]
            if not values:
                continue
            payload[f"{metric}/mean"] = mean(values)
            payload[f"{metric}/std"] = pstdev(values) if len(values) > 1 else 0.0
            payload[f"{metric}/min"] = min(values)
            payload[f"{metric}/max"] = max(values)
        out.append(payload)
    return out


def acceptance(policy_rows: list[dict[str, Any]], main_mode: str, baselines: set[str]) -> list[dict[str, Any]]:
    out = []
    for baseline in sorted(baselines):
        rows = [row for row in policy_rows if row.get("baseline_id") == baseline and row.get("mode") == main_mode]
        if not rows:
            out.append({"baseline_id": baseline, "status": "missing", "reason": f"no {main_mode} rows"})
            continue
        level_ok = sum((as_float(row.get("task_level_max")) or 0.0) >= 7.0 for row in rows)
        stone_ok = sum((as_float(row.get("event_count_craft_stone_tool")) or 0.0) > 0.0 for row in rows)
        collapse_ok = all((as_float(row.get("dominant_action_rate")) or 1.0) < 0.5 for row in rows)
        unarmed_kills = sum((as_float(row.get("event_count_mob_kill_unarmed")) or 0.0) for row in rows)
        mode_rows = [
            row for row in policy_rows
            if row.get("baseline_id") == baseline and row.get("mode") in {"deterministic", "mode"}
        ]
        deterministic_or_mode_ok = defaultdict(bool)
        for row in mode_rows:
            seed = row.get("seed")
            deterministic_or_mode_ok[seed] = deterministic_or_mode_ok[seed] or (
                (as_float(row.get("task_level_max")) or 0.0) >= 5.0
            )
        consolidated_count = sum(deterministic_or_mode_ok.values())
        status = "pass" if (
            level_ok >= 2
            and stone_ok >= 2
            and consolidated_count >= 2
            and collapse_ok
            and unarmed_kills == 0
        ) else "review"
        out.append(
            {
                "baseline_id": baseline,
                "status": status,
                "main_mode": main_mode,
                "seed_count": len(rows),
                "temp05_level_ge_7_seeds": level_ok,
                "temp05_stone_tool_positive_seeds": stone_ok,
                "deterministic_or_mode_level_ge_5_seeds": consolidated_count,
                "temp05_dominant_action_rate_all_lt_0_5": collapse_ok,
                "unarmed_kill_count": unarmed_kills,
            }
        )
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    baselines = {item.strip() for item in args.baselines.split(",") if item.strip()}
    out_dir = Path(args.out_dir)
    policy_files = find_policy_eval_files(args.eval_root)
    marl_files = find_marl_summary_files(args.marl_summary_root)
    policy_rows = collect_policy_rows(policy_files, baselines)
    marl_rows = collect_marl_rows(marl_files, baselines)
    policy_agg = aggregate(policy_rows, ("baseline_id", "mode"), DEFAULT_METRICS)
    marl_agg = aggregate(
        marl_rows,
        ("baseline_id",),
        (
            "MARL Evaluation/eval_real_reward_auc",
            "MARL Evaluation/eval_real_reward_curve_mean",
            "MARL Evaluation/real_imagined_reward_gap",
            "MARL Evaluation/eval_imagined_reward",
            "MARL Training/training_imagined_reward",
        ),
    )
    acceptance_rows = acceptance(policy_rows, args.main_mode, baselines)
    payload = {
        "policy_eval_files": [str(path) for path in policy_files],
        "marl_summary_files": [str(path) for path in marl_files],
        "policy_rows": policy_rows,
        "marl_rows": marl_rows,
        "policy_aggregate": policy_agg,
        "marl_aggregate": marl_agg,
        "acceptance": acceptance_rows,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    write_csv(out_dir / "policy_rows.csv", policy_rows)
    write_csv(out_dir / "policy_aggregate.csv", policy_agg)
    write_csv(out_dir / "marl_aggregate.csv", marl_agg)
    write_csv(out_dir / "acceptance.csv", acceptance_rows)
    print(json.dumps({"out_dir": str(out_dir), "acceptance": acceptance_rows}, indent=2), flush=True)


if __name__ == "__main__":
    main()
