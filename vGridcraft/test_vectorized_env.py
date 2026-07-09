from __future__ import annotations

import sys
from pathlib import Path

import torch

from vgridcraft.config import VGridcraftConfig
from vgridcraft.dataset import collect_dataset, dataset_key
from vgridcraft.env import BLOCK_EMPTY, BLOCK_TREE, ITEM_APPLE, ITEM_WOOD, VectorizedGridcraftEnv


def test_collect_dataset_marks_padding_invalid_after_terminal():
    data = collect_dataset(
        episodes=4,
        max_steps=3,
        num_envs=2,
        device="cpu",
        seed=7,
        config=VGridcraftConfig(num_agents=1, max_steps=3, seed=7),
    )
    assert data["transition_valid"].shape == (4, 3)
    assert data["transition_valid"].all()
    assert data["episode_length"].tolist() == [3, 3, 3, 3]
    assert data["metadata"]["invalid_padded_transition_count"] == 0


def test_partial_reset_only_resets_selected_envs():
    config = VGridcraftConfig(num_agents=1, max_steps=10, seed=11)
    env = VectorizedGridcraftEnv(num_envs=3, num_agents=1, device="cpu", seed=11, config=config)
    before = env.agent_x.clone()
    env.step(torch.zeros((3, 1), dtype=torch.long))
    env.reset(env_ids=torch.tensor([1]))
    assert env.step_count.tolist() == [1, 0, 1]
    assert env.agent_x[0].item() == before[0].item()
    assert env.agent_x[2].item() == before[2].item()


def test_vectorized_local_grid_matches_loop_reference():
    config = VGridcraftConfig(num_agents=3, max_steps=10, seed=19)
    env = VectorizedGridcraftEnv(num_envs=5, num_agents=3, device="cpu", seed=19, config=config)
    try:
        actions = torch.randint(0, config.action_size, (5, 3), generator=torch.Generator().manual_seed(123))
        env.step(actions)
        assert torch.equal(env.local_grid(), env.local_grid_loop())
    finally:
        env.close()


def test_collect_dataset_reports_vectorized_collection_metrics():
    data = collect_dataset(
        episodes=4,
        max_steps=3,
        num_envs=2,
        device="cpu",
        seed=17,
        config=VGridcraftConfig(num_agents=2, max_steps=3, seed=17),
    )
    assert data["metadata"]["collection_env_steps_per_second"] > 0
    assert data["metadata"]["collection_agent_steps_per_second"] > 0
    assert "collection_cpu_copy_time" in data["metadata"]


def test_dataset_key_changes_with_dynamics_and_reward_versions():
    base = VGridcraftConfig(num_agents=1)
    changed_reward = VGridcraftConfig(
        num_agents=1, reward_schema_version="different_reward"
    )
    changed_dynamics = VGridcraftConfig(
        num_agents=1, environment_dynamics_version="different_dynamics"
    )
    key = dataset_key(base, episodes=4, max_steps=5, seed=1)
    assert key != dataset_key(changed_reward, episodes=4, max_steps=5, seed=1)
    assert key != dataset_key(changed_dynamics, episodes=4, max_steps=5, seed=1)


def test_vectorized_harvest_tree_updates_block_and_inventory():
    config = VGridcraftConfig(num_agents=1, max_steps=10, seed=23, tree_apple_drop_chance=0.0)
    env = VectorizedGridcraftEnv(num_envs=3, num_agents=1, device="cpu", seed=23, config=config)
    try:
        env.agent_x[:, 0] = 5
        env.agent_y[:, 0] = 5
        x = env.agent_x[:, 0]
        y = env.agent_y[:, 0]
        env.blocks[:] = BLOCK_EMPTY
        env.blocks[torch.arange(3), y, x - 1] = BLOCK_TREE
        env.step(torch.full((3, 1), 5, dtype=torch.long))
        assert env.inventory[:, 0, ITEM_WOOD].tolist() == [1, 1, 1]
        assert torch.equal(env.blocks[torch.arange(3), y, x - 1], torch.zeros(3, dtype=torch.long))
    finally:
        env.close()


def test_vectorized_pickup_collects_adjacent_items():
    config = VGridcraftConfig(num_agents=1, max_steps=10, seed=29)
    env = VectorizedGridcraftEnv(num_envs=2, num_agents=1, device="cpu", seed=29, config=config)
    try:
        env.agent_x[:, 0] = 5
        env.agent_y[:, 0] = 5
        env.item_alive[:] = False
        env.item_alive[:, 0] = True
        env.item_type[:, 0] = ITEM_APPLE
        env.item_count[:, 0] = torch.tensor([2, 3])
        env.item_x[:, 0] = env.agent_x[:, 0] + 1
        env.item_y[:, 0] = env.agent_y[:, 0]
        env.step(torch.full((2, 1), 6, dtype=torch.long))
        assert env.inventory[:, 0, ITEM_APPLE].tolist() == [2, 3]
        assert not bool(env.item_alive[:, 0].any())
    finally:
        env.close()


def test_vectorized_attack_hits_adjacent_mob():
    config = VGridcraftConfig(num_agents=1, max_steps=10, seed=31, item_drop_chance=0.0)
    env = VectorizedGridcraftEnv(num_envs=2, num_agents=1, device="cpu", seed=31, config=config)
    try:
        env.agent_x[:, 0] = 5
        env.agent_y[:, 0] = 5
        env.mob_alive[:] = False
        env.mob_alive[:, 0] = True
        env.mob_hp[:, 0] = config.wood_sword_damage
        env.mob_x[:, 0] = env.agent_x[:, 0] + 1
        env.mob_y[:, 0] = env.agent_y[:, 0]
        env.equipped[:, 0] = 4
        env.step(torch.full((2, 1), 7, dtype=torch.long))
        assert not bool(env.mob_alive[:, 0].any())
    finally:
        env.close()


def test_dream_torchrl_env_smoke(tmp_path):
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "gridcraft"))
    from vgridcraft.dream_torchrl_env import GridcraftDreamTorchRLEnv
    from torch_world_model import TorchGridcraftRNN, TorchGridcraftVAE

    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    torch.save(TorchGridcraftVAE().state_dict(), checkpoint_dir / "vae.pt")
    torch.save(TorchGridcraftRNN().state_dict(), checkpoint_dir / "rnn.pt")
    config = VGridcraftConfig(num_agents=2, max_steps=5, seed=3)
    env = GridcraftDreamTorchRLEnv(
        num_envs=2,
        device="cpu",
        seed=3,
        config=config,
        checkpoint_dir=checkpoint_dir,
    )
    td = env.reset()
    assert td.get(("agents", "observation")).shape == torch.Size([2, 2, config.obs_size])
    td.set(("agents", "action"), torch.zeros((2, 2), dtype=torch.long))
    step = env.step(td)
    assert step.get(("next", "agents", "observation")).shape == torch.Size([2, 2, config.obs_size])
    assert step.get(("next", "agents", "reward")).shape == torch.Size([2, 2, 1])
