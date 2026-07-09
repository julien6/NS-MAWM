import json
import subprocess
import sys
from pathlib import Path

from marl_hpo_registry import (
    checkpoint_checksum,
    env_exports,
    families_for_baseline,
    select_best_config,
    validate_best_config,
)


def test_families_for_baseline():
    assert families_for_baseline("B00_model-free-control", "masac") == ("masac_core",)
    assert families_for_baseline("B10_neural_k0.0", "mambpo") == ("masac_core", "mambpo_imagination")
    assert families_for_baseline("B25_projection_k0.3", "mambpo") == ("masac_core", "mambpo_imagination")
    assert families_for_baseline("B25_projection_k0.3", "mpc_cem") == ("masac_core",)


def test_env_exports_core_and_imagination():
    core = {"hyperparameters": {"lr": 5e-5, "gamma": 0.99, "frames_per_batch": 8192}}
    imagination = {"hyperparameters": {"mb_imagined_horizon": 3, "mb_lambda_imagined": 0.7}}
    assert env_exports(core, "masac_core")["MARL_LR"] == "5e-05"
    assert env_exports(core, "masac_core")["MARL_GAMMA"] == "0.99"
    assert env_exports(core, "masac_core")["MARL_FRAMES_PER_BATCH"] == "8192"
    assert env_exports(imagination, "mambpo_imagination")["MB_IMAGINED_HORIZON"] == "3"
    assert env_exports(imagination, "mambpo_imagination")["MB_LAMBDA_IMAGINED"] == "0.7"


def test_export_env_cli(tmp_path):
    root = tmp_path / "marl"
    core_dir = root / "masac_core"
    imagination_dir = root / "mambpo_imagination"
    core_dir.mkdir(parents=True)
    imagination_dir.mkdir(parents=True)
    (core_dir / "best_config.json").write_text(
        json.dumps({"score": 12.0, "hyperparameters": {"lr": 0.0001, "hidden_size": 256}})
    )
    (imagination_dir / "best_config.json").write_text(
        json.dumps({"score": 10.0, "hyperparameters": {"mb_imagined_horizon": 5}})
    )
    result = subprocess.run(
        [
            sys.executable,
            "marl_hpo_registry.py",
            "export-env",
            "--baseline-id",
            "B10_neural_k0.0",
            "--downstream-algo",
            "mambpo",
            "--root",
            str(root),
            "--require",
        ],
        cwd=Path(__file__).resolve().parent,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "export MARL_LR='0.0001'" in result.stdout
    assert "export MARL_HPO_CORE_REUSED='1'" in result.stdout
    assert "export MB_IMAGINED_HORIZON='5'" in result.stdout
    assert "export MARL_HPO_IMAGINATION_REUSED='1'" in result.stdout


def test_mambpo_validation_tracks_external_world_model(tmp_path):
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "vae.pt").write_bytes(b"vae")
    (checkpoint_dir / "rnn.pt").write_bytes(b"rnn")
    best = {
        "stage": "final",
        "selection_method": "mean_across_seeds_v1",
        "provenance": {
            "num_agents": 3,
            "external_checkpoint_checksum": checkpoint_checksum(str(checkpoint_dir)),
        },
    }
    valid, reason = validate_best_config(
        best,
        required_stage="final",
        num_agents=3,
        external_checkpoint_dir=str(checkpoint_dir),
    )
    assert valid, reason

    (checkpoint_dir / "rnn.pt").write_bytes(b"different-rnn")
    valid, reason = validate_best_config(
        best,
        required_stage="final",
        num_agents=3,
        external_checkpoint_dir=str(checkpoint_dir),
    )
    assert not valid
    assert "provenance" in reason


def test_best_config_uses_mean_across_seeds(tmp_path):
    trials = tmp_path / "trials"
    for index, (config, score) in enumerate(
        [
            ({"lr": 1e-4}, 100.0),
            ({"lr": 1e-4}, -100.0),
            ({"lr": 5e-5}, 40.0),
            ({"lr": 5e-5}, 50.0),
        ]
    ):
        path = trials / str(index)
        path.mkdir(parents=True)
        (path / "marl_hpo_trial_summary.json").write_text(
            json.dumps(
                {
                    "hpo_family": "masac_core",
                    "stage": "final",
                    "score": score,
                    "hyperparameters": config,
                    "provenance": {"num_agents": 3},
                }
            )
        )
    best = select_best_config(
        family="masac_core",
        trials_root=trials,
        results_root=tmp_path / "results",
        stage="final",
    )
    assert best["hyperparameters"]["lr"] == 5e-5
    assert best["score"] == 45.0
    assert best["config_trial_count"] == 2
