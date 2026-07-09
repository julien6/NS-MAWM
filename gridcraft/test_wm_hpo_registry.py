import json
import subprocess
import sys

from experiment_versions import version_provenance
from wm_hpo_registry import (
    hpo_env_exports,
    hpo_family_for_baseline,
    select_best_config,
    validate_best_config,
)


def test_hpo_family_for_baseline():
    assert hpo_family_for_baseline("B00_model-free-control") is None
    assert hpo_family_for_baseline("B10_neural_k0.0") == "neural_k0.0"
    assert hpo_family_for_baseline("B25_projection_k0.3") == "neural_k0.0"
    assert hpo_family_for_baseline("B26_projection_k0.6") == "neural_k0.0"
    assert hpo_family_for_baseline("B25_regularization_k0.3") == "regularization_k0.3"
    assert hpo_family_for_baseline("B26_regularization_k0.6") == "regularization_k0.6"
    assert hpo_family_for_baseline("B25_residual_k0.3") == "residual_k0.3"
    assert hpo_family_for_baseline("B26_residual_k0.6") == "residual_k0.6"


def test_hpo_env_exports():
    payload = {
        "hyperparameters": {
            "vae_z_size": 128,
            "rnn_size": 256,
            "rnn_num_mixture": 8,
            "reward_loss_weight": 2.0,
            "lambda_sym": 0.5,
        }
    }
    exports = hpo_env_exports(payload)
    assert exports["VAE_Z_SIZE"] == "128"
    assert exports["RNN_SIZE"] == "256"
    assert exports["RNN_NUM_MIXTURE"] == "8"
    assert exports["WM_REWARD_LOSS_WEIGHT"] == "2.0"
    assert exports["LAMBDA_SYM"] == "0.5"


def test_export_env_cli(tmp_path):
    config_dir = tmp_path / "world_model" / "neural_k0.0"
    config_dir.mkdir(parents=True)
    (config_dir / "best_config.json").write_text(
        json.dumps(
            {
                "score": 1.25,
                "best_run_url": "https://wandb.example/run",
                "hyperparameters": {"vae_z_size": 96, "rnn_size": 384},
            }
        )
    )
    result = subprocess.run(
        [
            sys.executable,
            "wm_hpo_registry.py",
            "export-env",
            "--baseline-id",
            "B25_projection_k0.3",
            "--root",
            str(tmp_path / "world_model"),
            "--require",
        ],
        cwd=__import__("pathlib").Path(__file__).resolve().parent,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "export WM_HPO_FAMILY='neural_k0.0'" in result.stdout
    assert "export VAE_Z_SIZE='96'" in result.stdout
    assert "export RNN_SIZE='384'" in result.stdout


def test_final_validation_rejects_screen_and_requires_checkpoints(tmp_path):
    screen = {
        "stage": "screen",
        "selection_method": "mean_across_seeds_v1",
        "provenance": {"num_agents": 3, **version_provenance()},
        "checkpoint_dir": str(tmp_path / "checkpoints"),
    }
    valid, reason = validate_best_config(
        screen, required_stage="final", num_agents=3, require_checkpoints=True
    )
    assert not valid
    assert "stage" in reason

    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "vae.pt").write_bytes(b"vae")
    (checkpoint_dir / "rnn.pt").write_bytes(b"rnn")
    final = {**screen, "stage": "final"}
    valid, reason = validate_best_config(
        final, required_stage="final", num_agents=3, require_checkpoints=True
    )
    assert valid, reason


def test_wm_hpo_rejects_pre_hierarchy_reward_provenance():
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


def test_best_wm_config_uses_mean_across_seeds(tmp_path):
    trials = tmp_path / "trials"
    for index, (config, score) in enumerate(
        [
            ({"rnn_size": 128}, 0.1),
            ({"rnn_size": 128}, 10.0),
            ({"rnn_size": 256}, 2.0),
            ({"rnn_size": 256}, 3.0),
        ]
    ):
        path = trials / str(index)
        path.mkdir(parents=True)
        (path / "hpo_trial_summary.json").write_text(
            json.dumps(
                {
                    "hpo_family": "neural_k0.0",
                    "stage": "final",
                    "score": score,
                    "hyperparameters": config,
                        "provenance": {"num_agents": 3, **version_provenance()},
                    "run_dir": str(path),
                }
            )
        )
    best = select_best_config(
        hpo_family="neural_k0.0",
        trials_root=trials,
        results_root=tmp_path / "results",
        stage="final",
    )
    assert best["hyperparameters"]["rnn_size"] == 256
    assert best["score"] == 2.5
    assert best["config_trial_count"] == 2
