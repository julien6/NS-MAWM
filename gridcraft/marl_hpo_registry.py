from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from .experiment_versions import validate_version_provenance, version_provenance
except ImportError:
    from experiment_versions import validate_version_provenance, version_provenance


MARL_HPO_FAMILIES = ("masac_core", "mambpo_imagination")
DEFAULT_MARL_HPO_ROOT = Path("hpo_results/marl")
HPO_STAGE_RANK = {"screen": 0, "promote": 1, "final": 2}

CORE_ENV_KEYS = {
    "model_type": "MARL_MODEL",
    "frames_per_batch": "MARL_FRAMES_PER_BATCH",
    "train_batch_size": "MARL_TRAIN_BATCH_SIZE",
    "optimizer_steps": "MARL_OPTIMIZER_STEPS",
    "hidden_size": "MARL_HIDDEN_SIZE",
    "lstm_layers": "MARL_LSTM_LAYERS",
    "lstm_dropout": "MARL_LSTM_DROPOUT",
    "lr": "MARL_LR",
    "gamma": "MARL_GAMMA",
    "polyak_tau": "MARL_POLYAK_TAU",
    "alpha_init": "MARL_ALPHA_INIT",
    "discrete_target_entropy_weight": "MARL_DISCRETE_TARGET_ENTROPY_WEIGHT",
    "entropy_profile": "MARL_ENTROPY_PROFILE",
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


def stage_satisfies(actual: str | None, required: str) -> bool:
    return HPO_STAGE_RANK.get(str(actual), -1) >= HPO_STAGE_RANK[required]


def checkpoint_checksum(checkpoint_dir: str | None) -> str | None:
    if not checkpoint_dir:
        return None
    root = Path(checkpoint_dir)
    vae_mdn_paths = [root / "vae.pt", root / "rnn.pt"]
    structured_paths = [root / "structured_wm.pt"]
    if all(path.is_file() for path in vae_mdn_paths):
        paths = vae_mdn_paths
    elif all(path.is_file() for path in structured_paths):
        paths = structured_paths
    else:
        return None
    payload = ":".join(f"{path.resolve()}:{path.stat().st_size}:{path.stat().st_mtime_ns}" for path in paths)
    return hashlib.sha256(payload.encode()).hexdigest()


def validate_best_config(
    best_config: dict[str, Any] | None,
    *,
    required_stage: str = "final",
    num_agents: int | None = None,
    required_model_type: str | None = None,
    external_checkpoint_dir: str | None = None,
    minimum_budget: dict[str, Any] | None = None,
) -> tuple[bool, str]:
    if best_config is None:
        return False, "best_config.json is missing"
    if not stage_satisfies(best_config.get("stage"), required_stage):
        return False, f"stage {best_config.get('stage')!r} does not satisfy {required_stage!r}"
    if best_config.get("selection_method") != "mean_across_seeds_v1":
        return False, "selection method is not mean_across_seeds_v1"
    provenance = best_config.get("provenance", {})
    versions_valid, versions_reason = validate_version_provenance(provenance)
    if not versions_valid:
        return False, versions_reason
    if num_agents is not None and int(provenance.get("num_agents", -1)) != int(num_agents):
        return False, f"num_agents={provenance.get('num_agents')!r} does not match {num_agents}"
    if required_model_type:
        hyperparams = best_config.get("hyperparameters", best_config)
        actual_model_type = str(hyperparams.get("model_type", "mlp"))
        if actual_model_type != str(required_model_type):
            return False, f"model_type={actual_model_type!r} does not match {required_model_type!r}"
    if external_checkpoint_dir is not None:
        expected = checkpoint_checksum(external_checkpoint_dir)
        actual = provenance.get("external_checkpoint_checksum")
        if expected is None:
            return False, f"external World Model checkpoints are missing in {external_checkpoint_dir}"
        if actual != expected:
            return False, "external World Model checkpoint provenance does not match"
    if minimum_budget:
        actual_budget = best_config.get("budget", {})
        for key, required_value in minimum_budget.items():
            if key == "seeds":
                required = set(str(required_value).split())
                actual = set(str(actual_budget.get(key, "")).split())
                if not required.issubset(actual):
                    return False, f"budget seeds={sorted(actual)} do not include {sorted(required)}"
                continue
            try:
                if float(actual_budget.get(key, -1)) < float(required_value):
                    return False, f"budget {key}={actual_budget.get(key)!r} is below {required_value!r}"
            except (TypeError, ValueError):
                return False, f"budget {key}={actual_budget.get(key)!r} is invalid"
    return True, "valid"


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
    lines.append(f"export {prefix}_STAGE='{best_config.get('stage', '')}'")
    provenance = best_config.get("provenance", {})
    lines.append(
        f"export {prefix}_CHECKPOINT_CHECKSUM="
        f"'{provenance.get('external_checkpoint_checksum', '')}'"
    )
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
    stage: str = "screen",
    num_agents: int | None = None,
    external_checkpoint_dir: str | None = None,
) -> dict[str, Any]:
    family = normalize_family(family)
    score = score_from_metrics(metrics, family)
    baseline_id = config.get("baseline_id")
    return {
        "hpo_family": family,
        "baseline_id": baseline_id,
        "score": score,
        "metrics": metrics,
        "hyperparameters": extract_hyperparameters(config, family),
        "run_dir": str(run_dir),
        "wandb_run_url": wandb_run_url,
        "trial_id": trial_id,
        "sweep_id": sweep_id,
        "stage": stage,
        "provenance": {
            "baseline_id": baseline_id,
            "num_agents": num_agents,
            "seed": config.get("seed"),
            "external_checkpoint_dir": external_checkpoint_dir,
            "external_checkpoint_checksum": checkpoint_checksum(external_checkpoint_dir),
            **version_provenance(),
        },
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
    required_model_type: str | None = None,
) -> dict[str, Any]:
    family = normalize_family(family)
    candidates = []
    for path in sorted(Path(trials_root).glob("**/marl_hpo_trial_summary.json")):
        try:
            with path.open() as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("hpo_family") != family or payload.get("stage") != stage:
            continue
        if required_model_type:
            actual_model_type = str(payload.get("hyperparameters", {}).get("model_type", "mlp"))
            if actual_model_type != str(required_model_type):
                continue
        if not validate_version_provenance(payload.get("provenance", {}))[0]:
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
    grouped: dict[str, list[tuple[float, Path, dict[str, Any]]]] = {}
    for candidate in candidates:
        key = _config_key(candidate[2].get("hyperparameters", {}))
        grouped.setdefault(key, []).append(candidate)
    ranked_groups = sorted(
        grouped.values(),
        key=lambda rows: sum(row[0] for row in rows) / len(rows),
        reverse=True,
    )
    best_group = ranked_groups[0]
    score = sum(row[0] for row in best_group) / len(best_group)
    score_std = float(
        math.sqrt(sum((row[0] - score) ** 2 for row in best_group) / len(best_group))
    )
    _, source_path, best = max(best_group, key=lambda row: row[0])
    out = {
        "hpo_family": family,
        "score": score,
        "score_std": score_std,
        "seed_scores": [row[0] for row in best_group],
        "selection_method": "mean_across_seeds_v1",
        "hyperparameters": best.get("hyperparameters", {}),
        "metrics": best.get("metrics", {}),
        "best_run_dir": best.get("run_dir"),
        "best_run_url": best.get("wandb_run_url"),
        "selected_from": str(source_path),
        "trial_count": len(candidates),
        "config_trial_count": len(best_group),
        "budget": budget or {},
        "stage": stage,
        "provenance": best.get("provenance", {}),
        "selected_at": datetime.now(timezone.utc).isoformat(),
    }
    path = best_config_path(family, results_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(out, handle, indent=2)
    return out


def collect_ranked_trials(
    *,
    family: str,
    trials_root: str | Path,
    stage: str | None = None,
    required_model_type: str | None = None,
) -> list[dict[str, Any]]:
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
        if not validate_version_provenance(payload.get("provenance", {}))[0]:
            continue
        if stage is not None and payload.get("stage") != stage:
            continue
        if required_model_type:
            actual_model_type = str(payload.get("hyperparameters", {}).get("model_type", "mlp"))
            if actual_model_type != str(required_model_type):
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
    required_model_type: str | None = None,
) -> dict[str, Any]:
    family = normalize_family(family)
    if required_model_type:
        candidates = [
            candidate for candidate in candidates
            if str(candidate.get("hyperparameters", {}).get("model_type", "mlp")) == str(required_model_type)
        ]
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
        payload["configs"] = _unique_configs(candidates, top_k)
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
    temp_reward = first_metric(
        metrics,
        "Policy hierarchy evaluation/temp_0.5/episode_return_mean",
        "Policy hierarchy evaluation/temp_0.5_episode_return_mean",
    )
    if temp_reward is not None:
        score = _as_float(temp_reward, default=-1e9)
        task_level = _as_float(
            first_metric(
                metrics,
                "Policy hierarchy evaluation/temp_0.5/task_level_max",
                "Policy hierarchy evaluation/temp_0.5_task_level_max",
            ),
            default=0.0,
        )
        tool_equipped = _as_float(
            first_metric(
                metrics,
                "Policy hierarchy evaluation/temp_0.5/event_count_tool_equipped",
                "Policy hierarchy evaluation/temp_0.5_event_count_tool_equipped",
            ),
            default=0.0,
        )
        stone_tool = _as_float(
            first_metric(
                metrics,
                "Policy hierarchy evaluation/temp_0.5/event_count_craft_stone_tool",
                "Policy hierarchy evaluation/temp_0.5_event_count_craft_stone_tool",
            ),
            default=0.0,
        )
        dominant_rate = _as_float(
            first_metric(
                metrics,
                "Policy hierarchy evaluation/temp_0.5/dominant_action_rate",
                "Policy hierarchy evaluation/temp_0.5_dominant_action_rate",
            ),
            default=0.0,
        )
        entropy = _as_float(
            first_metric(
                metrics,
                "Policy hierarchy evaluation/temp_0.5/policy_entropy_mean",
                "Policy hierarchy evaluation/temp_0.5_policy_entropy_mean",
            ),
            default=0.0,
        )
        score += 10.0 * task_level
        score += 0.5 * tool_equipped
        score += 1.0 * stone_tool
        if dominant_rate > 0.5:
            score -= 20.0 * (dominant_rate - 0.5)
        if task_level < 4.0 and entropy < 0.2:
            score -= 25.0 * (0.2 - entropy)
        if normalize_family(family) == "mambpo_imagination":
            gap = abs(_as_float(first_metric(metrics, "MARL Evaluation/real_imagined_reward_gap"), default=0.0))
            failed = _as_float(first_metric(metrics, "MARL Evaluation/eval_imagined_generation_failed"), default=0.0)
            score -= 0.1 * gap + 1000.0 * failed
        return float(score)

    eval_reward = first_metric(metrics, "MARL Evaluation/eval_agents_reward_episode_reward_mean", "eval/agents/reward/episode_reward_mean")
    if eval_reward is None:
        eval_reward = first_metric(metrics, "MARL Evaluation/eval_reward_episode_reward_mean", "eval/reward/episode_reward_mean")
    final_reward = _as_float(eval_reward, default=-1e9)
    curve_mean = _as_float(
        first_metric(metrics, "MARL Evaluation/eval_real_reward_curve_mean"),
        default=final_reward,
    )
    stability = abs(
        _as_float(
            first_metric(metrics, "MARL Evaluation/eval_real_reward_stability_std"),
            default=0.0,
        )
    )
    score = 0.5 * final_reward + 0.5 * curve_mean - 0.1 * stability
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


def _config_key(config: dict[str, Any]) -> str:
    return json.dumps(config, sort_keys=True, separators=(",", ":"))


def _unique_configs(candidates: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    configs = []
    seen = set()
    for candidate in candidates:
        config = candidate.get("hyperparameters", {})
        key = _config_key(config)
        if key in seen:
            continue
        seen.add(key)
        configs.append(config)
        if len(configs) >= limit:
            break
    return configs


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
    export.add_argument("--required-stage", choices=tuple(HPO_STAGE_RANK), default=None)
    export.add_argument("--num-agents", type=int)
    export.add_argument("--required-model-type", choices=("mlp", "lstm"))
    export.add_argument("--allow-missing-imagination", action="store_true")

    validate = sub.add_parser("validate")
    validate.add_argument("--family", required=True, choices=MARL_HPO_FAMILIES)
    validate.add_argument("--root", default=str(DEFAULT_MARL_HPO_ROOT))
    validate.add_argument("--required-stage", choices=tuple(HPO_STAGE_RANK), default="final")
    validate.add_argument("--num-agents", type=int)
    validate.add_argument("--required-model-type", choices=("mlp", "lstm"))
    validate.add_argument("--external-checkpoint-dir")
    validate.add_argument("--minimum-budget-json", default="{}")

    select = sub.add_parser("select-best")
    select.add_argument("--family", required=True, choices=MARL_HPO_FAMILIES)
    select.add_argument("--trials-root", required=True)
    select.add_argument("--results-root", default=str(DEFAULT_MARL_HPO_ROOT))
    select.add_argument("--budget-json", default="{}")
    select.add_argument("--stage", default="screen")
    select.add_argument("--top-k", type=int, default=3)
    select.add_argument("--required-model-type", choices=("mlp", "lstm"))

    stage_results = sub.add_parser("write-stage-results")
    stage_results.add_argument("--family", required=True, choices=MARL_HPO_FAMILIES)
    stage_results.add_argument("--trials-root", required=True)
    stage_results.add_argument("--results-root", default=str(DEFAULT_MARL_HPO_ROOT))
    stage_results.add_argument("--stage", default="screen")
    stage_results.add_argument("--source-stage")
    stage_results.add_argument("--top-k", type=int, default=3)
    stage_results.add_argument("--required-model-type", choices=("mlp", "lstm"))

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
                if args.require and not (family == "mambpo_imagination" and args.allow_missing_imagination):
                    print(message, file=sys.stderr)
                    raise SystemExit(2)
                print(f"echo {json.dumps(message)} >&2")
                continue
            if args.required_stage:
                valid, reason = validate_best_config(
                    best,
                    required_stage=args.required_stage,
                    num_agents=args.num_agents,
                    required_model_type=args.required_model_type,
                )
                if not valid:
                    message = f"[marl-hpo] incompatible config for {family}: {reason}"
                    if args.require and not (family == "mambpo_imagination" and args.allow_missing_imagination):
                        print(message, file=sys.stderr)
                        raise SystemExit(2)
                    print(f"echo {json.dumps(message)} >&2")
                    continue
            best["_root"] = args.root
            print(shell_exports(best, family))
        return
    if args.command == "validate":
        best = load_best_config(args.family, args.root)
        valid, reason = validate_best_config(
            best,
            required_stage=args.required_stage,
            num_agents=args.num_agents,
            required_model_type=args.required_model_type,
            external_checkpoint_dir=args.external_checkpoint_dir,
            minimum_budget=json.loads(args.minimum_budget_json),
        )
        if not valid:
            print(f"[marl-hpo] {args.family}: {reason}", file=sys.stderr)
            raise SystemExit(2)
        print(f"[marl-hpo] {args.family}: valid")
        return
    if args.command == "select-best":
        print(json.dumps(select_best_config(family=args.family, trials_root=args.trials_root, results_root=args.results_root, budget=json.loads(args.budget_json), stage=args.stage, top_k=args.top_k, required_model_type=args.required_model_type), indent=2))
        return
    if args.command == "write-stage-results":
        ranked = collect_ranked_trials(
            family=args.family,
            trials_root=args.trials_root,
            stage=args.source_stage,
            required_model_type=args.required_model_type,
        )
        print(json.dumps(write_stage_results(family=args.family, candidates=ranked, results_root=args.results_root, stage=args.stage, top_k=args.top_k, required_model_type=args.required_model_type), indent=2))
        return
    if args.command == "write-summary":
        print(json.dumps(write_summary(args.results_root), indent=2))


if __name__ == "__main__":
    main()
