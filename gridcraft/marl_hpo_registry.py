from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


MARL_HPO_FAMILIES = ("masac_core", "mambpo_imagination")
DEFAULT_MARL_HPO_ROOT = Path("hpo_results/marl")

CORE_ENV_KEYS = {
    "frames_per_batch": "MARL_FRAMES_PER_BATCH",
    "train_batch_size": "MARL_TRAIN_BATCH_SIZE",
    "optimizer_steps": "MARL_OPTIMIZER_STEPS",
    "hidden_size": "MARL_HIDDEN_SIZE",
    "lr": "MARL_LR",
    "gamma": "MARL_GAMMA",
    "polyak_tau": "MARL_POLYAK_TAU",
    "alpha_init": "MARL_ALPHA_INIT",
    "discrete_target_entropy_weight": "MARL_DISCRETE_TARGET_ENTROPY_WEIGHT",
    "memory_size": "MARL_MEMORY_SIZE",
}

IMAGINATION_ENV_KEYS = {
    "mb_imagined_horizon": "MB_IMAGINED_HORIZON",
    "mb_imagined_branches": "MB_IMAGINED_BRANCHES",
    "mb_lambda_imagined": "MB_LAMBDA_IMAGINED",
    "mb_world_model_batch_size": "MB_WORLD_MODEL_BATCH_SIZE",
    "mb_world_model_train_epochs": "MB_WORLD_MODEL_TRAIN_EPOCHS",
}


def normalize_family(family: str) -> str:
    family = str(family)
    if family not in MARL_HPO_FAMILIES:
        raise ValueError(f"unsupported MARL HPO family {family!r}; expected one of {', '.join(MARL_HPO_FAMILIES)}")
    return family


def is_model_based_baseline(baseline_id: str) -> bool:
    text = str(baseline_id)
    return not (text.startswith("B00") or "model-free" in text)


def families_for_baseline(baseline_id: str, downstream_algo: str = "mambpo") -> tuple[str, ...]:
    families = ["masac_core"]
    if is_model_based_baseline(baseline_id) and str(downstream_algo) == "mambpo":
        families.append("mambpo_imagination")
    return tuple(families)


def best_config_path(family: str, root: str | Path = DEFAULT_MARL_HPO_ROOT) -> Path:
    return Path(root) / normalize_family(family) / "best_config.json"


def load_best_config(family: str, root: str | Path = DEFAULT_MARL_HPO_ROOT) -> dict[str, Any] | None:
    path = best_config_path(family, root)
    if not path.exists():
        return None
    with path.open() as handle:
        return json.load(handle)


def env_exports(best_config: dict[str, Any], family: str) -> dict[str, str]:
    hyperparams = best_config.get("hyperparameters", best_config)
    keys = CORE_ENV_KEYS if normalize_family(family) == "masac_core" else IMAGINATION_ENV_KEYS
    exports = {}
    for key, env_key in keys.items():
        if key in hyperparams and hyperparams[key] is not None:
            exports[env_key] = str(hyperparams[key])
    return exports


def shell_exports(best_config: dict[str, Any], family: str) -> str:
    exports = env_exports(best_config, family)
    lines = []
    for key in sorted(exports):
        value = str(exports[key]).replace("'", "'\"'\"'")
        lines.append(f"export {key}='{value}'")
    prefix = "MARL_HPO_CORE" if normalize_family(family) == "masac_core" else "MARL_HPO_IMAGINATION"
    path = str(best_config_path(family, best_config.get("_root", DEFAULT_MARL_HPO_ROOT))).replace("'", "'\"'\"'")
    lines.append(f"export {prefix}_REUSED='1'")
    lines.append(f"export {prefix}_SCORE='{best_config.get('score', '')}'")
    lines.append(f"export {prefix}_CONFIG_PATH='{path}'")
    if best_config.get("best_run_url"):
        url = str(best_config["best_run_url"]).replace("'", "'\"'\"'")
        lines.append(f"export {prefix}_BEST_RUN_URL='{url}'")
    return "\n".join(lines)


def build_trial_summary(
    *,
    family: str,
    run_dir: str | Path,
    config: dict[str, Any],
    metrics: dict[str, Any],
    wandb_run_url: str | None = None,
    trial_id: str | None = None,
    sweep_id: str | None = None,
) -> dict[str, Any]:
    family = normalize_family(family)
    score = score_from_metrics(metrics, family)
    return {
        "hpo_family": family,
        "score": score,
        "metrics": metrics,
        "hyperparameters": extract_hyperparameters(config, family),
        "run_dir": str(run_dir),
        "wandb_run_url": wandb_run_url,
        "trial_id": trial_id,
        "sweep_id": sweep_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def write_trial_summary(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2)


def trial_summary_path(run_dir: str | Path) -> Path:
    return Path(run_dir) / "marl_hpo_trial_summary.json"


def select_best_config(
    *,
    family: str,
    trials_root: str | Path,
    results_root: str | Path = DEFAULT_MARL_HPO_ROOT,
    budget: dict[str, Any] | None = None,
    stage: str = "screen",
    top_k: int = 3,
) -> dict[str, Any]:
    family = normalize_family(family)
    candidates = []
    for path in sorted(Path(trials_root).glob("**/marl_hpo_trial_summary.json")):
        try:
            with path.open() as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("hpo_family") != family:
            continue
        score = _as_float(payload.get("score"), default=-math.inf)
        if math.isfinite(score):
            candidates.append((score, path, payload))
    if not candidates:
        raise FileNotFoundError(f"no valid MARL HPO trial summaries found for {family} under {trials_root}")
    candidates.sort(key=lambda item: item[0], reverse=True)
    write_stage_results(
        family=family,
        candidates=[payload for _, _, payload in candidates],
        results_root=results_root,
        stage=stage,
        top_k=top_k,
    )
    score, source_path, best = candidates[0]
    out = {
        "hpo_family": family,
        "score": score,
        "hyperparameters": best.get("hyperparameters", {}),
        "metrics": best.get("metrics", {}),
        "best_run_dir": best.get("run_dir"),
        "best_run_url": best.get("wandb_run_url"),
        "selected_from": str(source_path),
        "trial_count": len(candidates),
        "budget": budget or {},
        "selected_at": datetime.now(timezone.utc).isoformat(),
    }
    path = best_config_path(family, results_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(out, handle, indent=2)
    return out


def collect_ranked_trials(*, family: str, trials_root: str | Path) -> list[dict[str, Any]]:
    family = normalize_family(family)
    ranked = []
    for path in sorted(Path(trials_root).glob("**/marl_hpo_trial_summary.json")):
        try:
            with path.open() as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("hpo_family") != family:
            continue
        score = _as_float(payload.get("score"), default=-math.inf)
        if math.isfinite(score):
            payload = dict(payload)
            payload["_source_path"] = str(path)
            ranked.append(payload)
    ranked.sort(key=lambda item: _as_float(item.get("score"), default=-math.inf), reverse=True)
    return ranked


def write_stage_results(
    *,
    family: str,
    candidates: list[dict[str, Any]],
    results_root: str | Path = DEFAULT_MARL_HPO_ROOT,
    stage: str = "screen",
    top_k: int = 3,
) -> dict[str, Any]:
    family = normalize_family(family)
    top_k = max(1, int(top_k))
    root = Path(results_root) / family
    payload = {
        "hpo_family": family,
        "stage": stage,
        "candidate_count": len(candidates),
        "top_k": top_k,
        "candidates": candidates,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    if stage == "screen":
        out_path = root / "screen_results.json"
    else:
        out_path = root / "promoted_configs.json"
        payload["configs"] = [candidate.get("hyperparameters", {}) for candidate in candidates[:top_k]]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as handle:
        json.dump(payload, handle, indent=2)
    return payload


def write_summary(results_root: str | Path = DEFAULT_MARL_HPO_ROOT) -> dict[str, Any]:
    results_root = Path(results_root)
    summary = {"families": {}, "created_at": datetime.now(timezone.utc).isoformat()}
    for family in MARL_HPO_FAMILIES:
        path = best_config_path(family, results_root)
        if path.exists():
            with path.open() as handle:
                payload = json.load(handle)
            summary["families"][family] = {
                "available": True,
                "score": payload.get("score"),
                "path": str(path),
                "best_run_url": payload.get("best_run_url"),
                "best_run_dir": payload.get("best_run_dir"),
            }
        else:
            summary["families"][family] = {"available": False, "path": str(path)}
    out_path = results_root / "summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def score_from_metrics(metrics: dict[str, Any], family: str) -> float:
    eval_reward = first_metric(metrics, "MARL Evaluation/eval_agents_reward_episode_reward_mean", "eval/agents/reward/episode_reward_mean")
    if eval_reward is None:
        eval_reward = first_metric(metrics, "MARL Evaluation/eval_reward_episode_reward_mean", "eval/reward/episode_reward_mean")
    score = _as_float(eval_reward, default=-1e9)
    if normalize_family(family) == "mambpo_imagination":
        gap = abs(_as_float(first_metric(metrics, "MARL Evaluation/real_imagined_reward_gap"), default=0.0))
        failed = _as_float(first_metric(metrics, "MARL Evaluation/eval_imagined_generation_failed"), default=0.0)
        score -= 0.1 * gap + 1000.0 * failed
    return float(score)


def first_metric(metrics: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in metrics:
            return metrics[key]
    suffixes = tuple(key.split("/", 1)[-1] for key in keys)
    for key, value in metrics.items():
        if str(key).endswith(suffixes):
            return value
    return None


def extract_hyperparameters(config: dict[str, Any], family: str) -> dict[str, Any]:
    keys = set(CORE_ENV_KEYS)
    if normalize_family(family) == "mambpo_imagination":
        keys.update(IMAGINATION_ENV_KEYS)
    keys.update({"num_envs", "max_steps", "max_iters", "eval_every_iters", "eval_episodes"})
    return {key: config[key] for key in sorted(keys) if key in config}


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    families = sub.add_parser("families-for-baseline")
    families.add_argument("--baseline-id", required=True)
    families.add_argument("--downstream-algo", default="mambpo")

    export = sub.add_parser("export-env")
    export.add_argument("--family", choices=MARL_HPO_FAMILIES)
    export.add_argument("--baseline-id")
    export.add_argument("--downstream-algo", default="mambpo")
    export.add_argument("--root", default=str(DEFAULT_MARL_HPO_ROOT))
    export.add_argument("--require", action="store_true")

    select = sub.add_parser("select-best")
    select.add_argument("--family", required=True, choices=MARL_HPO_FAMILIES)
    select.add_argument("--trials-root", required=True)
    select.add_argument("--results-root", default=str(DEFAULT_MARL_HPO_ROOT))
    select.add_argument("--budget-json", default="{}")
    select.add_argument("--stage", default="screen")
    select.add_argument("--top-k", type=int, default=3)

    stage_results = sub.add_parser("write-stage-results")
    stage_results.add_argument("--family", required=True, choices=MARL_HPO_FAMILIES)
    stage_results.add_argument("--trials-root", required=True)
    stage_results.add_argument("--results-root", default=str(DEFAULT_MARL_HPO_ROOT))
    stage_results.add_argument("--stage", default="screen")
    stage_results.add_argument("--top-k", type=int, default=3)

    summary = sub.add_parser("write-summary")
    summary.add_argument("--results-root", default=str(DEFAULT_MARL_HPO_ROOT))

    args = parser.parse_args()
    if args.command == "families-for-baseline":
        print(" ".join(families_for_baseline(args.baseline_id, args.downstream_algo)))
        return
    if args.command == "export-env":
        family_list = (args.family,) if args.family else families_for_baseline(args.baseline_id, args.downstream_algo)
        for family in family_list:
            best = load_best_config(family, args.root)
            if best is None:
                message = f"[marl-hpo] no best_config.json found for {family} at {best_config_path(family, args.root)}"
                if args.require:
                    print(message, file=sys.stderr)
                    raise SystemExit(2)
                print(f"echo {json.dumps(message)} >&2")
                continue
            best["_root"] = args.root
            print(shell_exports(best, family))
        return
    if args.command == "select-best":
        print(json.dumps(select_best_config(family=args.family, trials_root=args.trials_root, results_root=args.results_root, budget=json.loads(args.budget_json), stage=args.stage, top_k=args.top_k), indent=2))
        return
    if args.command == "write-stage-results":
        ranked = collect_ranked_trials(family=args.family, trials_root=args.trials_root)
        print(json.dumps(write_stage_results(family=args.family, candidates=ranked, results_root=args.results_root, stage=args.stage, top_k=args.top_k), indent=2))
        return
    if args.command == "write-summary":
        print(json.dumps(write_summary(args.results_root), indent=2))


if __name__ == "__main__":
    main()
