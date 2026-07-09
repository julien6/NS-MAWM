from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "gridcraft"))
sys.path.insert(0, str(ROOT / "Gridcraft"))

from diagnose_gridcraft_task_hierarchy import POLICY_LEVELS, run_episode
from vgridcraft.config import VGridcraftConfig
from vgridcraft.env import (
    BLOCK_EMPTY,
    BLOCK_TREE,
    EVENT_INDEX,
    ITEM_WOOD,
    REWARD_COMPONENT_INDEX,
    VectorizedGridcraftEnv,
)
from vgridcraft.torchrl_env import GridcraftTorchRLEnv


def test_reward_components_sum_to_reward_and_failed_action_is_not_success():
    config = VGridcraftConfig(
        num_agents=1,
        max_steps=5,
        seed=3,
        tree_apple_drop_chance=0.0,
        mob_spawn_rate=0,
    )
    env = VectorizedGridcraftEnv(1, device="cpu", config=config)
    env.blocks[:] = BLOCK_EMPTY
    env.agent_x[0, 0] = 5
    env.agent_y[0, 0] = 5
    env.blocks[0, 5, 6] = BLOCK_TREE
    _, reward, _, _, info = env.step(torch.tensor([[5]]))
    assert torch.allclose(info["reward_component_sum"], reward)
    assert info["event_success"][0, 0, EVENT_INDEX["harvest_wood"]] == 1
    assert info["reward_components"][0, 0, REWARD_COMPONENT_INDEX["harvest_wood"]] == 1
    _, reward, _, _, info = env.step(torch.tensor([[5]]))
    assert torch.allclose(info["reward_component_sum"], reward)
    assert info["event_success"][0, 0, EVENT_INDEX["harvest_wood"]] == 0
    assert env.inventory[0, 0, ITEM_WOOD] == 1


def test_controlled_policies_reach_exact_expected_level():
    rewards = []
    complexities = []
    for policy, expected_level in POLICY_LEVELS.items():
        _, summary = run_episode(
            policy=policy,
            protocol="controlled",
            seed=1,
            num_agents=3,
            max_steps=2,
            device="cpu",
        )
        assert summary["observed_level"] == expected_level
        rewards.append(summary["cumulative_reward"])
        complexities.append(summary["complexity_unique"])
    assert all(left < right for left, right in zip(rewards, rewards[1:]))
    assert all(left < right for left, right in zip(complexities, complexities[1:]))


def test_torchrl_propagates_hierarchy_stats():
    env = GridcraftTorchRLEnv(
        num_envs=2,
        device="cpu",
        config=VGridcraftConfig(num_agents=2, max_steps=3, seed=5),
    )
    td = env.reset()
    td.set(("agents", "action"), torch.zeros((2, 2), dtype=torch.long))
    step = env.step(td)
    assert step.get(("next", "agents", "event_success")).shape[-1] > 0
    assert step.get(("next", "agents", "reward_components")).shape[-1] > 0
    assert step.get(("next", "agents", "task_level_max")).shape == (2, 2, 1)


def test_classic_and_vectorized_harvest_telemetry_match():
    from gridcraft.config import GridcraftConfig
    from gridcraft.constants import Block, Terrain
    from gridcraft.world import GridcraftWorld

    classic_config = GridcraftConfig(
        width=16,
        height=16,
        num_agents=1,
        max_steps=5,
        seed=7,
        tree_apple_drop_chance=0.0,
        mob_spawn_rate=1000,
    )
    classic = GridcraftWorld(classic_config, np.random.default_rng(7))
    classic.reset(["agent_0"])
    classic.terrain[:] = Terrain.GRASS
    classic.blocks[:] = Block.EMPTY
    classic.mobs.clear()
    agent = classic.agents["agent_0"]
    agent.x = agent.y = 5
    classic.blocks[5, 6] = Block.TREE
    classic_result = classic.step({"agent_0": 5})

    vector = VectorizedGridcraftEnv(
        1,
        device="cpu",
        config=VGridcraftConfig(
            num_agents=1,
            max_steps=5,
            seed=7,
            tree_apple_drop_chance=0.0,
            mob_spawn_rate=0,
        ),
    )
    vector.terrain[:] = 0
    vector.blocks[:] = BLOCK_EMPTY
    vector.mob_alive[:] = False
    vector.agent_x[0, 0] = vector.agent_y[0, 0] = 5
    vector.blocks[0, 5, 6] = BLOCK_TREE
    _, vector_reward, _, _, vector_info = vector.step(torch.tensor([[5]]))

    assert classic_result.infos["agent_0"]["event_success"]["harvest_wood"] == 1
    assert vector_info["event_success"][0, 0, EVENT_INDEX["harvest_wood"]] == 1
    assert classic_result.rewards["agent_0"] == pytest.approx(float(vector_reward[0, 0]))
    assert abs(classic_result.infos["agent_0"]["reward_decomposition_error"]) < 1e-6
