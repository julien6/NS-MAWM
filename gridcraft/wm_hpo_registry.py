from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


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
        if payload.get("hpo_family") != family:
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
    score, source_path, best = candidates[0]
    best_config = {
        "hpo_family": family,
        "baseline_id": best.get("baseline_id", baseline_for_hpo_family(family)),
        "score": score,
        "hyperparameters": best.get("hyperparameters", {}),
        "metrics": best.get("metrics", {}),
        "best_run_dir": best.get("run_dir"),
        "best_run_url": best.get("wandb_run_url"),
        "dataset_path": best.get("dataset_path"),
        "selected_from": str(source_path),
        "trial_count": len(candidates),
        "budget": budget or {},
        "selected_at": datetime.now(timezone.utc).isoformat(),
    }
    out_path = best_config_path(family, results_root)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as handle:
        json.dump(best_config, handle, indent=2)
    return best_config


def collect_ranked_trials(*, hpo_family: str, trials_root: str | Path) -> list[dict[str, Any]]:
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
        payload["configs"] = [candidate.get("hyperparameters", {}) for candidate in candidates[:top_k]]
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
        print(shell_exports(best))
        path = str(best_config_path(family, args.root)).replace("'", "'\"'\"'")
        print(f"export WM_HPO_FAMILY='{family}'")
        print("export WM_HPO_CONFIG_REUSED='1'")
        print(f"export WM_HPO_CONFIG_PATH='{path}'")
        print(f"export WM_HPO_SCORE='{best.get('score', '')}'")
        if best.get("best_run_url"):
            url = str(best["best_run_url"]).replace("'", "'\"'\"'")
            print(f"export WM_HPO_BEST_RUN_URL='{url}'")
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
        ranked = collect_ranked_trials(hpo_family=args.hpo_family, trials_root=args.trials_root)
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
