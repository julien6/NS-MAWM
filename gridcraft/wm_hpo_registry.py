from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from .experiment_versions import validate_version_provenance, version_provenance
except ImportError:
    from experiment_versions import validate_version_provenance, version_provenance


HPO_FAMILIES = (
    "neural_k0.0",
    "regularization_k0.3",
    "regularization_k0.6",
    "residual_k0.3",
    "residual_k0.6",
)

HPO_BASELINE_IDS = {
    "neural_k0.0": "B10_neural_k0.0",
    "regularization_k0.3": "B25_regularization_k0.3",
    "regularization_k0.6": "B26_regularization_k0.6",
    "residual_k0.3": "B25_residual_k0.3",
    "residual_k0.6": "B26_residual_k0.6",
}

HPO_PSTR_PROFILES = {
    "neural_k0.0": "neural_k0.0",
    "regularization_k0.3": "ns_k0.3",
    "regularization_k0.6": "ns_k0.6",
    "residual_k0.3": "ns_k0.3",
    "residual_k0.6": "ns_k0.6",
}

HPO_ENV_KEYS = {
    "vae_z_size": "VAE_Z_SIZE",
    "vae_hidden_size": "VAE_HIDDEN_SIZE",
    "vae_kl_tolerance": "VAE_KL_TOLERANCE",
    "rnn_size": "RNN_SIZE",
    "rnn_num_mixture": "RNN_NUM_MIXTURE",
    "seq_len": "WM_SEQ_LEN",
    "wm_batch_size": "WM_BATCH_SIZE",
    "learning_rate": "WM_LEARNING_RATE",
    "mean_mse_weight": "WM_MEAN_MSE_WEIGHT",
    "reward_loss_weight": "WM_REWARD_LOSS_WEIGHT",
    "done_loss_weight": "WM_DONE_LOSS_WEIGHT",
    "lambda_sym": "LAMBDA_SYM",
    "lambda_residual": "LAMBDA_RESIDUAL",
}

DEFAULT_HPO_ROOT = Path("hpo_results/world_model")
HPO_STAGE_RANK = {"screen": 0, "promote": 1, "final": 2}


def hpo_family_for_baseline(baseline_id: str) -> str | None:
    text = str(baseline_id)
    if text.startswith("B00") or "model-free" in text:
        return None
    if "projection" in text:
        return "neural_k0.0"
    if "regularization" in text:
        return "regularization_k0.6" if "_k0.6" in text or "B26" in text else "regularization_k0.3"
    if "residual" in text:
        return "residual_k0.6" if "_k0.6" in text or "B26" in text else "residual_k0.3"
    return "neural_k0.0"


def baseline_for_hpo_family(hpo_family: str) -> str:
    family = normalize_hpo_family(hpo_family)
    return HPO_BASELINE_IDS[family]


def pstr_profile_for_hpo_family(hpo_family: str) -> str:
    family = normalize_hpo_family(hpo_family)
    return HPO_PSTR_PROFILES[family]


def normalize_hpo_family(hpo_family: str) -> str:
    family = str(hpo_family)
    if family not in HPO_FAMILIES:
        raise ValueError(f"unsupported HPO family {family!r}; expected one of {', '.join(HPO_FAMILIES)}")
    return family


def best_config_path(hpo_family: str, root: str | Path = DEFAULT_HPO_ROOT) -> Path:
    return Path(root) / normalize_hpo_family(hpo_family) / "best_config.json"


def load_best_config(hpo_family: str, root: str | Path = DEFAULT_HPO_ROOT) -> dict[str, Any] | None:
    path = best_config_path(hpo_family, root)
    if not path.exists():
        return None
    with path.open() as handle:
        payload = json.load(handle)
    return payload


def stage_satisfies(actual: str | None, required: str) -> bool:
    return HPO_STAGE_RANK.get(str(actual), -1) >= HPO_STAGE_RANK[required]


def dataset_checksum(dataset_path: str | None) -> str | None:
    if not dataset_path:
        return None
    path = Path(dataset_path).expanduser()
    try:
        stat = path.stat()
    except OSError:
        return None
    provenance = f"{path.resolve()}:{stat.st_size}:{path.name}"
    return hashlib.sha256(provenance.encode()).hexdigest()


def validate_best_config(
    best_config: dict[str, Any] | None,
    *,
    required_stage: str = "final",
    num_agents: int | None = None,
    require_checkpoints: bool = False,
    minimum_budget: dict[str, Any] | None = None,
) -> tuple[bool, str]:
    if best_config is None:
        return False, "best_config.json is missing"
    if required_stage not in HPO_STAGE_RANK:
        return False, f"unsupported required stage {required_stage!r}"
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
    if require_checkpoints:
        checkpoint_dir = Path(str(best_config.get("checkpoint_dir", "")))
        missing = [name for name in ("vae.pt", "rnn.pt") if not (checkpoint_dir / name).is_file()]
        if missing:
            return False, f"missing checkpoint files in {checkpoint_dir}: {', '.join(missing)}"
    stored_dataset_checksum = best_config.get("dataset_checksum")
    if stored_dataset_checksum:
        current_dataset_checksum = dataset_checksum(best_config.get("dataset_path"))
        if current_dataset_checksum != stored_dataset_checksum:
            return False, "dataset provenance checksum does not match"
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


def hpo_env_exports(best_config: dict[str, Any]) -> dict[str, str]:
    hyperparams = best_config.get("hyperparameters", best_config)
    exports: dict[str, str] = {}
    for key, env_key in HPO_ENV_KEYS.items():
        if key in hyperparams and hyperparams[key] is not None:
            exports[env_key] = str(hyperparams[key])
    return exports


def shell_exports(best_config: dict[str, Any]) -> str:
    exports = hpo_env_exports(best_config)
    lines = []
    for key in sorted(exports):
        value = exports[key].replace("'", "'\"'\"'")
        lines.append(f"export {key}='{value}'")
    return "\n".join(lines)


def trial_summary_path(run_dir: str | Path) -> Path:
    return Path(run_dir) / "hpo_trial_summary.json"


def build_trial_summary(
    *,
    hpo_family: str,
    run_dir: str | Path,
    config: dict[str, Any],
    metrics: dict[str, Any],
    wandb_run_url: str | None = None,
    dataset_path: str | None = None,
    sweep_id: str | None = None,
    trial_id: str | None = None,
    stage: str = "screen",
    num_agents: int | None = None,
) -> dict[str, Any]:
    family = normalize_hpo_family(hpo_family)
    score = _as_float(metrics.get("wm_hpo_score", math.inf), default=math.inf)
    return {
        "hpo_family": family,
        "baseline_id": baseline_for_hpo_family(family),
        "score": score,
        "metrics": metrics,
        "hyperparameters": extract_hpo_hyperparameters(config),
        "run_dir": str(run_dir),
        "wandb_run_url": wandb_run_url,
        "dataset_path": dataset_path,
        "dataset_checksum": dataset_checksum(dataset_path),
        "checkpoint_dir": str(Path(run_dir) / "checkpoints"),
        "stage": stage,
        "provenance": {
            "num_agents": num_agents,
            "seed": config.get("seed"),
            "episodes": config.get("episodes"),
            "max_steps": config.get("max_steps"),
            "vae_steps": config.get("vae_steps"),
            "rnn_steps": config.get("rnn_steps"),
            **version_provenance(),
        },
        "sweep_id": sweep_id,
        "trial_id": trial_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def write_trial_summary(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2)


def select_best_config(
    *,
    hpo_family: str,
    trials_root: str | Path,
    results_root: str | Path = DEFAULT_HPO_ROOT,
    budget: dict[str, Any] | None = None,
    stage: str = "screen",
    top_k: int = 3,
) -> dict[str, Any]:
    family = normalize_hpo_family(hpo_family)
    trial_files = sorted(Path(trials_root).glob("**/hpo_trial_summary.json"))
    candidates = []
    for path in trial_files:
        try:
            with path.open() as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("hpo_family") != family or payload.get("stage") != stage:
            continue
        if not validate_version_provenance(payload.get("provenance", {}))[0]:
            continue
        score = _as_float(payload.get("score"), default=math.inf)
        if math.isfinite(score):
            candidates.append((score, path, payload))
    if not candidates:
        raise FileNotFoundError(f"no valid HPO trial summaries found for {family} under {trials_root}")
    candidates.sort(key=lambda item: item[0])
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
    )
    best_group = ranked_groups[0]
    score = sum(row[0] for row in best_group) / len(best_group)
    score_std = float(
        math.sqrt(sum((row[0] - score) ** 2 for row in best_group) / len(best_group))
    )
    _, source_path, best = min(best_group, key=lambda row: row[0])
    best_config = {
        "hpo_family": family,
        "baseline_id": best.get("baseline_id", baseline_for_hpo_family(family)),
        "score": score,
        "score_std": score_std,
        "seed_scores": [row[0] for row in best_group],
        "selection_method": "mean_across_seeds_v1",
        "hyperparameters": best.get("hyperparameters", {}),
        "metrics": best.get("metrics", {}),
        "best_run_dir": best.get("run_dir"),
        "best_run_url": best.get("wandb_run_url"),
        "dataset_path": best.get("dataset_path"),
        "dataset_checksum": best.get("dataset_checksum"),
        "checkpoint_dir": best.get("checkpoint_dir") or str(Path(best.get("run_dir", "")) / "checkpoints"),
        "stage": stage,
        "provenance": best.get("provenance", {}),
        "selected_from": str(source_path),
        "trial_count": len(candidates),
        "config_trial_count": len(best_group),
        "budget": budget or {},
        "selected_at": datetime.now(timezone.utc).isoformat(),
    }
    out_path = best_config_path(family, results_root)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as handle:
        json.dump(best_config, handle, indent=2)
    return best_config


def collect_ranked_trials(
    *, hpo_family: str, trials_root: str | Path, stage: str | None = None
) -> list[dict[str, Any]]:
    family = normalize_hpo_family(hpo_family)
    ranked = []
    for path in sorted(Path(trials_root).glob("**/hpo_trial_summary.json")):
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
        score = _as_float(payload.get("score"), default=math.inf)
        if math.isfinite(score):
            payload = dict(payload)
            payload["_source_path"] = str(path)
            ranked.append(payload)
    ranked.sort(key=lambda item: _as_float(item.get("score"), default=math.inf))
    return ranked


def write_stage_results(
    *,
    family: str,
    candidates: list[dict[str, Any]],
    results_root: str | Path = DEFAULT_HPO_ROOT,
    stage: str = "screen",
    top_k: int = 3,
) -> dict[str, Any]:
    family = normalize_hpo_family(family)
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


def write_summary(results_root: str | Path = DEFAULT_HPO_ROOT) -> dict[str, Any]:
    results_root = Path(results_root)
    summary = {"families": {}, "created_at": datetime.now(timezone.utc).isoformat()}
    for family in HPO_FAMILIES:
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


def extract_hpo_hyperparameters(config: dict[str, Any]) -> dict[str, Any]:
    keys = set(HPO_ENV_KEYS)
    keys.update({"vae_steps", "rnn_steps", "episodes", "max_steps", "num_envs", "eval_every", "horizons"})
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

    family_parser = sub.add_parser("family-for-baseline")
    family_parser.add_argument("--baseline-id", required=True)

    export_parser = sub.add_parser("export-env")
    export_parser.add_argument("--hpo-family")
    export_parser.add_argument("--baseline-id")
    export_parser.add_argument("--root", default=str(DEFAULT_HPO_ROOT))
    export_parser.add_argument("--require", action="store_true")
    export_parser.add_argument("--required-stage", choices=tuple(HPO_STAGE_RANK), default=None)
    export_parser.add_argument("--num-agents", type=int)

    validate_parser = sub.add_parser("validate")
    validate_parser.add_argument("--hpo-family", required=True)
    validate_parser.add_argument("--root", default=str(DEFAULT_HPO_ROOT))
    validate_parser.add_argument("--required-stage", choices=tuple(HPO_STAGE_RANK), default="final")
    validate_parser.add_argument("--num-agents", type=int)
    validate_parser.add_argument("--require-checkpoints", action="store_true")
    validate_parser.add_argument("--print-checkpoint-dir", action="store_true")
    validate_parser.add_argument("--minimum-budget-json", default="{}")

    select_parser = sub.add_parser("select-best")
    select_parser.add_argument("--hpo-family", required=True)
    select_parser.add_argument("--trials-root", required=True)
    select_parser.add_argument("--results-root", default=str(DEFAULT_HPO_ROOT))
    select_parser.add_argument("--budget-json", default="{}")
    select_parser.add_argument("--stage", default="screen")
    select_parser.add_argument("--top-k", type=int, default=3)

    ranked_parser = sub.add_parser("write-stage-results")
    ranked_parser.add_argument("--hpo-family", required=True)
    ranked_parser.add_argument("--trials-root", required=True)
    ranked_parser.add_argument("--results-root", default=str(DEFAULT_HPO_ROOT))
    ranked_parser.add_argument("--stage", default="screen")
    ranked_parser.add_argument("--source-stage")
    ranked_parser.add_argument("--top-k", type=int, default=3)

    summary_parser = sub.add_parser("write-summary")
    summary_parser.add_argument("--results-root", default=str(DEFAULT_HPO_ROOT))

    args = parser.parse_args()
    if args.command == "family-for-baseline":
        family = hpo_family_for_baseline(args.baseline_id)
        if family:
            print(family)
        return
    if args.command == "export-env":
        family = args.hpo_family or hpo_family_for_baseline(args.baseline_id)
        if not family:
            return
        best = load_best_config(family, args.root)
        if best is None:
            message = f"[wm-hpo] no best_config.json found for {family} at {best_config_path(family, args.root)}"
            if args.require:
                print(message, file=sys.stderr)
                raise SystemExit(2)
            print(f"echo {json.dumps(message)} >&2")
            return
        if args.required_stage:
            valid, reason = validate_best_config(
                best,
                required_stage=args.required_stage,
                num_agents=args.num_agents,
            )
            if not valid:
                message = f"[wm-hpo] incompatible config for {family}: {reason}"
                if args.require:
                    print(message, file=sys.stderr)
                    raise SystemExit(2)
                print(f"echo {json.dumps(message)} >&2")
                return
        print(shell_exports(best))
        path = str(best_config_path(family, args.root)).replace("'", "'\"'\"'")
        print(f"export WM_HPO_FAMILY='{family}'")
        print("export WM_HPO_CONFIG_REUSED='1'")
        print(f"export WM_HPO_CONFIG_PATH='{path}'")
        print(f"export WM_HPO_SCORE='{best.get('score', '')}'")
        print(f"export WM_HPO_STAGE='{best.get('stage', '')}'")
        print(f"export WM_HPO_DATASET_CHECKSUM='{best.get('dataset_checksum', '')}'")
        print(f"export WM_HPO_CHECKPOINT_DIR='{best.get('checkpoint_dir', '')}'")
        if best.get("best_run_url"):
            url = str(best["best_run_url"]).replace("'", "'\"'\"'")
            print(f"export WM_HPO_BEST_RUN_URL='{url}'")
        return
    if args.command == "validate":
        best = load_best_config(args.hpo_family, args.root)
        valid, reason = validate_best_config(
            best,
            required_stage=args.required_stage,
            num_agents=args.num_agents,
            require_checkpoints=args.require_checkpoints,
            minimum_budget=json.loads(args.minimum_budget_json),
        )
        if not valid:
            print(f"[wm-hpo] {args.hpo_family}: {reason}", file=sys.stderr)
            raise SystemExit(2)
        if args.print_checkpoint_dir:
            print(best["checkpoint_dir"])
        else:
            print(f"[wm-hpo] {args.hpo_family}: valid")
        return
    if args.command == "select-best":
        budget = json.loads(args.budget_json)
        payload = select_best_config(
            hpo_family=args.hpo_family,
            trials_root=args.trials_root,
            results_root=args.results_root,
            budget=budget,
            stage=args.stage,
            top_k=args.top_k,
        )
        print(json.dumps(payload, indent=2))
        return
    if args.command == "write-stage-results":
        ranked = collect_ranked_trials(
            hpo_family=args.hpo_family,
            trials_root=args.trials_root,
            stage=args.source_stage,
        )
        payload = write_stage_results(
            family=args.hpo_family,
            candidates=ranked,
            results_root=args.results_root,
            stage=args.stage,
            top_k=args.top_k,
        )
        print(json.dumps(payload, indent=2))
        return
    if args.command == "write-summary":
        print(json.dumps(write_summary(args.results_root), indent=2))


if __name__ == "__main__":
    main()
