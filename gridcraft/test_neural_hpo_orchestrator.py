from pathlib import Path
import os
import subprocess


def test_orchestrator_dry_run_has_dependency_order():
    root = Path(__file__).resolve().parent
    env = {
        **os.environ,
        "DRY_RUN": "1",
        "SEEDS": "1",
        "HPO_SEEDS": "1",
        "MARL_HPO_SEEDS": "1",
        "AUTO_RESOURCE_PROFILE": "0",
    }
    result = subprocess.run(
        ["bash", "run_neural_baselines_with_hpo.bash"],
        cwd=root,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    output = result.stdout
    phases = [
        "Phase 1/6: mandatory MASAC core HPO",
        "Phase 2/6: final model-free MASAC baseline",
        "Phase 3/6: mandatory neural World Model HPO",
        "Phase 4/6: validate selected neural World Model checkpoint",
        "Phase 5/6: mandatory MAMBPO imagination HPO",
        "Phase 6/6: final neural World Model + MAMBPO baseline",
    ]
    positions = [output.index(phase) for phase in phases]
    assert positions == sorted(positions)
    assert "MARL_HPO_FAMILIES=masac_core" in output
    assert "HPO_FAMILIES=neural_k0.0" in output
    assert "MARL_HPO_FAMILIES=mambpo_imagination" in output
    assert "BASELINES=B00_model-free-control" in output
    assert "BASELINES=B10_neural_k0.0" in output
    assert "REQUIRE_WM_HPO=1" in output
    assert "REQUIRE_MARL_HPO=1" in output
