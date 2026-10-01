from copy import deepcopy
import json
import pytest
import torch
import yaml
from ns_mawm.environments import make_env, collect
from ns_mawm.libraries import make_library, save_library, load_library
from ns_mawm.data import Dataset
from ns_mawm.training import train, evaluate, build_network
from ns_mawm.control import RecurrentMASAC, BenchMARLBridge, control
from ns_mawm.refinement import RuleWorkflow, leakage_audit
from ns_mawm.rules import Library, Context, PSWM, compile_rule


def tiny(name="gridcraft"):
    return {"run": {"env": name, "seed": 0}, "world_model": {"backbone": "jopm_lstm", "strategy": "none", "lambda": .3,
        "sizes": {"enc": [8], "lstm": 8, "g": [8], "dec": [8]},
        "optim": {"updates": 1, "batch_seqs": 1, "seq_len": 2, "burn_in": 1}},
        "environment": {"agents": 2, "max_steps": 3}, "evaluation": {"horizon": 2}}


@pytest.fixture(scope="module")
def grid():
    torch.set_num_threads(1)
    env = make_env("gridcraft", agents=2, view_size=3, max_steps=4)
    data = Dataset.split(collect(env, 8))
    schema = env.schema.fit(torch.cat([e.observations for e in data.get("train")]))
    yield env, data, schema, make_library(env, schema)
    env.close()


@pytest.mark.parametrize("strategy", ["none", "projection", "residual", "regularization", "feature_weighting"])
def test_short_training_all_strategies(grid, strategy):
    env, data, schema, library = grid
    cfg = tiny()
    cfg["world_model"]["strategy"] = strategy
    network, timing = train(data, schema, library, env.action_size, cfg)
    result = evaluate(data, schema, library, network, cfg)
    assert timing["updates"] == 1
    assert result["rollout_steps"] > 0
    assert 0 <= result["metrics"]["mse"] < 1e6


@pytest.mark.parametrize("backbone", ["mamba_wm", "pswm_only", "dreamer_central"])
def test_baselines(grid, backbone):
    env, data, schema, library = grid
    cfg = tiny()
    cfg["world_model"]["backbone"] = backbone
    network, _ = train(data, schema, library, env.action_size, cfg)
    result = evaluate(data, schema, library, network, cfg)
    assert result["metrics"]["mse"] >= 0


def test_checkpoint_and_dataset_roundtrip(grid, tmp_path):
    env, data, schema, library = grid
    data.save(tmp_path / "data")
    restored = Dataset.load(tmp_path / "data")
    assert restored.hash == data.hash
    save_library(library, tmp_path / "library.json")
    restored_lib = load_library(tmp_path / "library.json", schema)
    assert restored_lib.hash == library.hash
    cfg = tiny()
    network, _ = train(data, schema, library, env.action_size, cfg)
    torch.save(network.state_dict(), tmp_path / "net.pt")
    another = build_network(schema, env.action_size, cfg)
    another.load_state_dict(torch.load(tmp_path / "net.pt", weights_only=True))
    assert evaluate(data, schema, library, network, cfg)["metrics"]['mse'] == evaluate(data, schema, library, another, cfg)["metrics"]['mse']


@pytest.mark.parametrize("name", ["overcooked", "predator_prey"])
def test_real_environment_roundtrip_and_training(name):
    pytest.importorskip("overcooked_ai_py" if name == "overcooked" else "mpe2")
    env = make_env(name, agents=2, max_steps=3)
    try:
        episodes = collect(env, 4, seed=2)
        again = collect(env, 4, seed=2)
        assert torch.equal(episodes[0].observations, again[0].observations)
        data = Dataset.split(episodes)
        schema = env.schema.fit(torch.cat([e.observations for e in data.get("train")]))
        library = make_library(env, schema)
        cfg = tiny(name)
        network, _ = train(data, schema, library, env.action_size, cfg)
        result = evaluate(data, schema, library, network, cfg)
        assert result["metrics"]["mse"] >= 0
        assert any(r.scope == "joint" for r in library.rules)
    finally:
        env.close()


def test_recurrent_masac_update(grid):
    env, data, schema, library = grid
    cfg = {"hidden": 8, "burn_in": 1}
    learner = RecurrentMASAC(schema, env.action_size, cfg)
    episode = data.get("train")[0]
    row = (episode.observations[1], episode.actions[1].tolist(), 0., episode.observations[2], False, [episode.observations[0]])
    loss = learner.update([row])
    assert all(torch.isfinite(torch.tensor(v)) for v in loss.values())


def test_short_control(grid):
    env, data, schema, library = grid
    cfg = tiny()
    cfg["environment"].update(view_size=3, max_steps=4)
    cfg["control"] = {"hidden": 8, "real_steps": 8, "checkpoint_interval": 4, "fit_interval": 4,
        "world_model_updates": 1, "rollout_cap": 2, "warmup": 2, "batch_size": 2, "eval_episodes": 1}
    result = control(env, data, schema, library, cfg)
    assert result["real_steps"] == 8 and result["imagined_replay"] > 0
    assert len(result["returns"]) == 2


def test_t13_leakage(tmp_path):
    (tmp_path / "sim.py").write_text("def step(a, b): return a + b + 123")
    assert leakage_audit([{"source": "public_docs", "text": "def step(a, b): return a + b + 123"}], tmp_path)["flagged"]


def test_llm_budget_archive_and_leakage(grid, tmp_path):
    env, data, schema, library = grid
    def mock(payload):
        return {"choices": [{"message": {"content": '{"patches": []}'}}], "usage": {"prompt_tokens": 10, "completion_tokens": 3}}
    workflow = RuleWorkflow(schema, data, tmp_path, {"max_candidates": 1}, mock)
    with pytest.raises(PermissionError):
        workflow.call("P1", [{"source": "simulator_source", "text": "anything"}])
    test_id = data.manifest["splits"]["test"][0]
    with pytest.raises(PermissionError):
        workflow.call("P3", [], evidence=[{"split": "rev", "evidence_id": test_id + ":0"}])
    assert workflow.call("P1", []) == {"patches": []}
    assert (tmp_path / "calls.jsonl").exists()
    with pytest.raises(RuntimeError, match="budget"):
        workflow.call("P1", [])
