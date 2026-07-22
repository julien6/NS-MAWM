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
    parser.add_argument("--comparison-id", default=None)
    parser.add_argument("--latest-per-baseline-seed", action="store_true")
    parser.add_argument("--require-seeds", type=int, default=0)
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
    if isinstance(value, str):
        try:
            parsed = float(value)
        except ValueError:
            return None
        if math.isfinite(parsed):
            return parsed
    return None


def comparison_matches(row: dict[str, Any], comparison_id: str | None) -> bool:
    if not comparison_id:
        return True
    return row.get("comparison_id") == comparison_id


def collect_policy_rows(files: list[Path], baselines: set[str], comparison_id: str | None) -> list[dict[str, Any]]:
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
            if not comparison_matches(row, comparison_id):
                continue
            copied = dict(row)
            copied["source_file"] = str(path)
            copied["source_mtime"] = path.stat().st_mtime
            rows.append(copied)
    return rows


def collect_marl_rows(files: list[Path], baselines: set[str], comparison_id: str | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in files:
        payload = read_json(path)
        config = payload.get("config", {}) if isinstance(payload, dict) else {}
        metrics = payload.get("metrics", {}) if isinstance(payload, dict) else {}
        baseline = config.get("baseline_id") or payload.get("baseline_id")
        if baseline not in baselines:
            continue
        payload_comparison_id = config.get("comparison_id") or payload.get("comparison_id")
        if comparison_id and payload_comparison_id != comparison_id:
            continue
        row = {
            "baseline_id": baseline,
            "seed": config.get("seed") or payload.get("seed"),
            "comparison_id": payload_comparison_id,
            "checkpoint_path": config.get("checkpoint_path") or payload.get("checkpoint_path"),
            "missing_marl_metrics": ";".join(payload.get("missing_marl_metrics", [])) if isinstance(payload.get("missing_marl_metrics"), list) else payload.get("missing_marl_metrics"),
            "source_file": str(path),
            "source_mtime": path.stat().st_mtime,
        }
        for key in (
            "MARL Evaluation/eval_real_reward_auc",
            "MARL Evaluation/eval_real_reward_curve_mean",
            "MARL Evaluation/eval_real_reward_stability_std",
            "MARL Evaluation/eval_real_reward_point_count",
            "MARL Training/imagination_external_world_model_used",
            "MARL Training/training_imagined_reward",
            "MARL Training/imagined_batch_size",
            "MARL Training/imagination_used_for_training",
            "MARL Evaluation/eval_imagined_reward",
            "MARL Evaluation/real_imagined_reward_gap",
        ):
            if key in metrics:
                row[key] = metrics[key]
        rows.append(row)
    return rows


def keep_latest(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    latest: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        group = tuple(row.get(key) for key in keys)
        previous = latest.get(group)
        if previous is None or (as_float(row.get("source_mtime")) or 0.0) > (as_float(previous.get("source_mtime")) or 0.0):
            latest[group] = row
    return sorted(latest.values(), key=lambda row: tuple(str(row.get(key)) for key in keys))


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


def validate_required_seeds(policy_rows: list[dict[str, Any]], marl_rows: list[dict[str, Any]], baselines: set[str], main_mode: str, required: int) -> list[dict[str, Any]]:
    if required <= 0:
        return []
    issues = []
    for baseline in sorted(baselines):
        policy_seeds = {
            row.get("seed")
            for row in policy_rows
            if row.get("baseline_id") == baseline and row.get("mode") == main_mode
        }
        marl_seeds = {
            row.get("seed")
            for row in marl_rows
            if row.get("baseline_id") == baseline
        }
        if len(policy_seeds) != required:
            issues.append({
                "baseline_id": baseline,
                "kind": "policy",
                "expected_seed_count": required,
                "actual_seed_count": len(policy_seeds),
                "seeds": " ".join(str(seed) for seed in sorted(policy_seeds, key=str)),
            })
        if len(marl_seeds) != required:
            issues.append({
                "baseline_id": baseline,
                "kind": "marl",
                "expected_seed_count": required,
                "actual_seed_count": len(marl_seeds),
                "seeds": " ".join(str(seed) for seed in sorted(marl_seeds, key=str)),
            })
    return issues


def deltas(policy_agg: list[dict[str, Any]], baselines: list[str]) -> list[dict[str, Any]]:
    if len(baselines) < 2:
        return []
    reference, candidate = baselines[0], baselines[1]
    by_key = {(row.get("baseline_id"), row.get("mode")): row for row in policy_agg}
    metrics = (
        "episode_return_mean/mean",
        "task_level_max/mean",
        "event_count_craft_stone_tool/mean",
        "event_count_tool_equipped/mean",
        "event_count_mob_kill_armed/mean",
        "dominant_action_rate/mean",
    )
    rows = []
    for mode in sorted({row.get("mode") for row in policy_agg}):
        ref = by_key.get((reference, mode))
        cand = by_key.get((candidate, mode))
        if not ref or not cand:
            continue
        row = {"reference_baseline": reference, "candidate_baseline": candidate, "mode": mode}
        for metric in metrics:
            ref_value = as_float(ref.get(metric))
            cand_value = as_float(cand.get(metric))
            if ref_value is None or cand_value is None:
                continue
            row[f"{metric}/reference"] = ref_value
            row[f"{metric}/candidate"] = cand_value
            row[f"{metric}/delta"] = cand_value - ref_value
        rows.append(row)
    return rows


def main() -> None:
    args = parse_args()
    baseline_list = [item.strip() for item in args.baselines.split(",") if item.strip()]
    baselines = set(baseline_list)
    out_dir = Path(args.out_dir)
    policy_files = find_policy_eval_files(args.eval_root)
    marl_files = find_marl_summary_files(args.marl_summary_root)
    policy_rows = collect_policy_rows(policy_files, baselines, args.comparison_id)
    marl_rows = collect_marl_rows(marl_files, baselines, args.comparison_id)
    if args.latest_per_baseline_seed:
        policy_rows = keep_latest(policy_rows, ("baseline_id", "seed", "mode"))
        marl_rows = keep_latest(marl_rows, ("baseline_id", "seed"))
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
            "MARL Training/imagined_batch_size",
            "MARL Training/imagination_used_for_training",
        ),
    )
    acceptance_rows = acceptance(policy_rows, args.main_mode, baselines)
    delta_rows = deltas(policy_agg, baseline_list)
    validation_issues = validate_required_seeds(policy_rows, marl_rows, baselines, args.main_mode, args.require_seeds)
    if validation_issues:
        out_dir.mkdir(parents=True, exist_ok=True)
        write_csv(out_dir / "validation_issues.csv", validation_issues)
        raise SystemExit(
            "Missing required seeds for final comparison; see "
            f"{out_dir / 'validation_issues.csv'}"
        )
    payload = {
        "comparison_id": args.comparison_id,
        "policy_eval_files": [str(path) for path in policy_files],
        "marl_summary_files": [str(path) for path in marl_files],
        "policy_rows": policy_rows,
        "marl_rows": marl_rows,
        "policy_aggregate": policy_agg,
        "marl_aggregate": marl_agg,
        "acceptance": acceptance_rows,
        "deltas": delta_rows,
        "metric_roles": {
            "primary_controlled_behavior": "temp_0.5",
            "secondary_controlled_behavior": "temp_0.25",
            "stochastic_potential": ["temp_1.0", "sampled"],
            "collapse_diagnostics": ["deterministic", "mode"],
        },
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    write_csv(out_dir / "policy_rows.csv", policy_rows)
    write_csv(out_dir / "policy_aggregate.csv", policy_agg)
    write_csv(out_dir / "marl_aggregate.csv", marl_agg)
    write_csv(out_dir / "acceptance.csv", acceptance_rows)
    write_csv(out_dir / "deltas.csv", delta_rows)
    print(json.dumps({"out_dir": str(out_dir), "acceptance": acceptance_rows}, indent=2), flush=True)


if __name__ == "__main__":
    main()
