from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


POLICY_METRICS = (
    "episode_return_mean",
    "task_level_max",
    "event_count_craft_stone_tool",
    "event_count_tool_equipped",
    "event_count_mob_kill_armed",
    "event_count_mob_kill_unarmed",
    "dominant_action_rate",
)
MARL_METRICS = (
    "MARL Evaluation/eval_real_reward_auc",
    "MARL Evaluation/eval_real_reward_curve_mean",
    "MARL Evaluation/real_imagined_reward_gap",
    "MARL Evaluation/eval_imagined_reward",
    "MARL Training/training_imagined_reward",
    "MARL Training/imagined_batch_size",
    "MARL Training/imagination_used_for_training",
)
LAMBDA_RE = re.compile(r"_lambda_([0-9]+(?:p[0-9]+)?)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate B11 MAMBPO lambda ablations.")
    parser.add_argument("--eval-root", action="append", default=[])
    parser.add_argument("--marl-summary-root", action="append", default=[])
    parser.add_argument("--out-dir", default="analysis_b11_mambpo_lambda_ablation")
    parser.add_argument("--reference-baseline", default="B00_model-free-control")
    parser.add_argument("--main-mode", default="temp_0.5")
    parser.add_argument("--secondary-mode", default="temp_0.25")
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def as_float(value: Any, default: float | None = None) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return default


def lambda_from_baseline(baseline_id: str) -> float | None:
    if baseline_id == "B11_structured_neural_k0.0_no_imagination":
        return 0.0
    match = LAMBDA_RE.search(str(baseline_id))
    if not match:
        return None
    return float(match.group(1).replace("p", "."))


def lambda_label(value: float | None, baseline_id: str) -> str:
    if value is None:
        return "reference" if baseline_id.startswith("B00") else "unknown"
    return f"{value:g}"


def find_files(roots: list[str], pattern: str, defaults: tuple[str, ...]) -> list[Path]:
    roots = roots or list(defaults)
    files: list[Path] = []
    seen = set()
    for root_text in roots:
        root = Path(root_text)
        if not root.exists():
            continue
        for path in root.rglob(pattern):
            resolved = path.resolve()
            if resolved not in seen:
                seen.add(resolved)
                files.append(path)
    return sorted(files)


def collect_policy_rows(files: list[Path], reference_baseline: str) -> list[dict[str, Any]]:
    rows = []
    for path in files:
        payload = read_json(path)
        if isinstance(payload, dict):
            payload = [payload]
        if not isinstance(payload, list):
            continue
        for row in payload:
            if not isinstance(row, dict):
                continue
            baseline = str(row.get("baseline_id", ""))
            lam = lambda_from_baseline(baseline)
            if lam is None and baseline != reference_baseline:
                continue
            copied = dict(row)
            copied["lambda"] = lambda_label(lam, baseline)
            copied["lambda_value"] = lam
            copied["source_file"] = str(path)
            rows.append(copied)
    return rows


def collect_marl_rows(files: list[Path], reference_baseline: str) -> list[dict[str, Any]]:
    rows = []
    for path in files:
        payload = read_json(path)
        if not isinstance(payload, dict):
            continue
        config = payload.get("config", {})
        metrics = payload.get("metrics", {})
        baseline = str(config.get("baseline_id") or payload.get("baseline_id") or "")
        lam = lambda_from_baseline(baseline)
        if lam is None and baseline != reference_baseline:
            continue
        row = {
            "baseline_id": baseline,
            "lambda": lambda_label(lam, baseline),
            "lambda_value": lam,
            "seed": config.get("seed") or payload.get("seed"),
            "source_file": str(path),
        }
        for key in MARL_METRICS:
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


def ranking(policy_rows: list[dict[str, Any]], main_mode: str, secondary_mode: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in policy_rows:
        grouped[str(row.get("lambda"))].append(row)
    out = []
    for lam, rows in grouped.items():
        main = [row for row in rows if row.get("mode") == main_mode]
        secondary = [row for row in rows if row.get("mode") == secondary_mode]
        if not main:
            continue
        main_return = mean([as_float(row.get("episode_return_mean"), 0.0) or 0.0 for row in main])
        secondary_return = mean([as_float(row.get("episode_return_mean"), 0.0) or 0.0 for row in secondary]) if secondary else 0.0
        task_level = mean([as_float(row.get("task_level_max"), 0.0) or 0.0 for row in main])
        dominant = mean([as_float(row.get("dominant_action_rate"), 1.0) or 1.0 for row in main])
        unarmed = sum([as_float(row.get("event_count_mob_kill_unarmed"), 0.0) or 0.0 for row in main])
        collapse_penalty = max(0.0, dominant - 0.5) * 50.0 + unarmed * 100.0
        score = 0.5 * main_return + 0.3 * secondary_return + 10.0 * task_level - collapse_penalty
        out.append(
            {
                "lambda": lam,
                "lambda_value": as_float(lam),
                "is_reference": lam == "reference",
                "seed_count": len({row.get("seed") for row in main}),
                "main_mode": main_mode,
                "secondary_mode": secondary_mode,
                "temp05_return_mean": main_return,
                "temp025_return_mean": secondary_return,
                "task_level_max_mean": task_level,
                "dominant_action_rate_mean": dominant,
                "unarmed_kill_count": unarmed,
                "lambda_score": score,
            }
        )
    out.sort(key=lambda row: (as_float(row.get("lambda_score"), -1e9) or -1e9), reverse=True)
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
    policy_files = find_files(
        args.eval_root,
        "policy_hierarchy_eval_summary.json",
        ("policy_hierarchy_eval", "runs_benchmarl", "runs_benchmarl_marl_hpo"),
    )
    marl_files = find_files(
        args.marl_summary_root,
        "*_marl_summary.json",
        ("runs_benchmarl", "runs_benchmarl_marl_hpo"),
    )
    policy_rows = collect_policy_rows(policy_files, args.reference_baseline)
    marl_rows = collect_marl_rows(marl_files, args.reference_baseline)
    policy_aggregate = aggregate(policy_rows, ("lambda", "baseline_id", "mode"), POLICY_METRICS)
    marl_aggregate = aggregate(marl_rows, ("lambda", "baseline_id"), MARL_METRICS)
    lambda_ranking = ranking(policy_rows, args.main_mode, args.secondary_mode)
    payload = {
        "policy_eval_files": [str(path) for path in policy_files],
        "marl_summary_files": [str(path) for path in marl_files],
        "policy_rows": policy_rows,
        "marl_rows": marl_rows,
        "policy_aggregate": policy_aggregate,
        "marl_aggregate": marl_aggregate,
        "lambda_ranking": lambda_ranking,
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    write_csv(out_dir / "policy_rows.csv", policy_rows)
    write_csv(out_dir / "policy_aggregate.csv", policy_aggregate)
    write_csv(out_dir / "marl_rows.csv", marl_rows)
    write_csv(out_dir / "marl_aggregate.csv", marl_aggregate)
    write_csv(out_dir / "lambda_ranking.csv", lambda_ranking)
    print(json.dumps({"out_dir": str(out_dir), "lambda_ranking": lambda_ranking}, indent=2), flush=True)


if __name__ == "__main__":
    main()
