import json
import subprocess
import sys
from pathlib import Path

from experiment_versions import version_provenance
from marl_hpo_registry import (
    checkpoint_checksum,
    env_exports,
    families_for_baseline,
    score_from_metrics,
    select_best_config,
    validate_best_config,
)


def test_families_for_baseline():
    assert families_for_baseline("B00_model-free-control", "masac") == ("masac_core",)
    assert families_for_baseline("B10_neural_k0.0", "mambpo") == ("masac_core", "mambpo_imagination")
    assert families_for_baseline("B25_projection_k0.3", "mambpo") == ("masac_core", "mambpo_imagination")
    assert families_for_baseline("B25_projection_k0.3", "mpc_cem") == ("masac_core",)


def test_env_exports_core_and_imagination():
    core = {
        "hyperparameters": {
            "model_type": "lstm",
            "lstm_layers": 2,
            "lstm_dropout": 0.0,
            "lr": 5e-5,
            "gamma": 0.99,
            "frames_per_batch": 8192,
        }
    }
    imagination = {"hyperparameters": {"mb_imagined_horizon": 3, "mb_lambda_imagined": 0.7}}
    assert env_exports(core, "masac_core")["MARL_LR"] == "5e-05"
    assert env_exports(core, "masac_core")["MARL_GAMMA"] == "0.99"
    assert env_exports(core, "masac_core")["MARL_FRAMES_PER_BATCH"] == "8192"
    assert env_exports(core, "masac_core")["MARL_MODEL"] == "lstm"
    assert env_exports(core, "masac_core")["MARL_LSTM_LAYERS"] == "2"
    assert env_exports(core, "masac_core")["MARL_LSTM_DROPOUT"] == "0.0"
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


def test_export_env_requires_masac_core_even_when_imagination_can_be_missing(tmp_path):
    root = tmp_path / "marl"
    result = subprocess.run(
        [
            sys.executable,
            "marl_hpo_registry.py",
            "export-env",
            "--baseline-id",
            "B11_structured_neural_k0.0",
            "--downstream-algo",
            "mambpo",
            "--root",
            str(root),
            "--require",
            "--allow-missing-imagination",
        ],
        cwd=Path(__file__).resolve().parent,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "masac_core" in result.stderr


def test_export_env_allows_missing_imagination_for_disabled_ablation(tmp_path):
    root = tmp_path / "marl"
    core_dir = root / "masac_core"
    core_dir.mkdir(parents=True)
    (core_dir / "best_config.json").write_text(
        json.dumps({"score": 12.0, "hyperparameters": {"model_type": "lstm", "lr": 0.0001}})
    )
    result = subprocess.run(
        [
            sys.executable,
            "marl_hpo_registry.py",
            "export-env",
            "--baseline-id",
            "B11_structured_neural_k0.0",
            "--downstream-algo",
            "mambpo",
            "--root",
            str(root),
            "--require",
            "--allow-missing-imagination",
        ],
        cwd=Path(__file__).resolve().parent,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "export MARL_HPO_CORE_REUSED='1'" in result.stdout
    assert "mambpo_imagination" in result.stdout
    assert "no best_config.json found" in result.stdout


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
            **version_provenance(),
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


def test_marl_hpo_rejects_pre_hierarchy_reward_provenance():
    valid, reason = validate_best_config(
        {
            "stage": "final",
            "selection_method": "mean_across_seeds_v1",
            "provenance": {"num_agents": 3},
        },
        required_stage="final",
        num_agents=3,
    )
    assert not valid
    assert "version" in reason


def test_marl_hpo_rejects_wrong_model_type():
    best = {
        "stage": "final",
        "selection_method": "mean_across_seeds_v1",
        "hyperparameters": {"model_type": "mlp"},
        "provenance": {"num_agents": 3, **version_provenance()},
    }
    valid, reason = validate_best_config(
        best,
        required_stage="final",
        num_agents=3,
        required_model_type="lstm",
    )
    assert not valid
    assert "model_type" in reason


def test_marl_hpo_score_prioritizes_temp_05_policy_quality():
    weak_temp_05 = {
        "Policy hierarchy evaluation/temp_0.5/episode_return_mean": 0.0,
        "Policy hierarchy evaluation/temp_0.5/task_level_max": 2.0,
        "Policy hierarchy evaluation/temp_0.5/event_count_tool_equipped": 0.0,
        "Policy hierarchy evaluation/temp_0.25/episode_return_mean": 1000.0,
        "Policy hierarchy evaluation/temp_0.25/task_level_max": 8.0,
    }
    strong_temp_05 = {
        "Policy hierarchy evaluation/temp_0.5/episode_return_mean": 10.0,
        "Policy hierarchy evaluation/temp_0.5/task_level_max": 5.0,
        "Policy hierarchy evaluation/temp_0.5/event_count_tool_equipped": 4.0,
        "Policy hierarchy evaluation/temp_0.25/episode_return_mean": -1000.0,
        "Policy hierarchy evaluation/temp_0.25/task_level_max": 0.0,
    }
    assert score_from_metrics(strong_temp_05, "masac_core") > score_from_metrics(weak_temp_05, "masac_core")


def test_marl_hpo_score_penalizes_temp_05_action_collapse():
    base = {
        "Policy hierarchy evaluation/temp_0.5/episode_return_mean": 50.0,
        "Policy hierarchy evaluation/temp_0.5/task_level_max": 5.0,
        "Policy hierarchy evaluation/temp_0.5/dominant_action_rate": 0.5,
    }
    collapsed = {
        **base,
        "Policy hierarchy evaluation/temp_0.5/dominant_action_rate": 0.9,
    }
    assert score_from_metrics(base, "masac_core") > score_from_metrics(collapsed, "masac_core")


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
                        "provenance": {"num_agents": 3, **version_provenance()},
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
