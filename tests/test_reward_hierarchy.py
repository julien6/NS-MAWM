from __future__ import annotations


import pytest
import torch


from vGridcraft.vgridcraft.config import VGridcraftConfig
from vGridcraft.vgridcraft.env import (
    BLOCK_EMPTY,
    BLOCK_TREE,
    EVENT_INDEX,
    ITEM_WOOD,
    ITEM_WOOD_SWORD,
    REWARD_COMPONENT_INDEX,
    VectorizedGridcraftEnv,
)


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
    assert info["reward_components"][
        0, 0, REWARD_COMPONENT_INDEX["harvest_wood"]
    ] == pytest.approx(config.harvest_wood_reward)
    assert info["reward_components"][
        0, 0, REWARD_COMPONENT_INDEX["milestone"]
    ] == pytest.approx(config.task_milestone_rewards[2])
    _, reward, _, _, info = env.step(torch.tensor([[5]]))
    assert torch.allclose(info["reward_component_sum"], reward)
    assert info["event_success"][0, 0, EVENT_INDEX["harvest_wood"]] == 0
    assert env.inventory[0, 0, ITEM_WOOD] == 1


def test_torchrl_propagates_hierarchy_stats():
    pytest.importorskip("torchrl")
    from vGridcraft.vgridcraft.torchrl_env import GridcraftTorchRLEnv
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


def test_dense_reward_is_capped_but_milestone_is_not_repeated():
    config = VGridcraftConfig(
        width=32,
        height=8,
        num_agents=1,
        max_mobs=0,
        dense_reward_event_cap=10,
        max_steps=20,
    )
    env = VectorizedGridcraftEnv(1, device="cpu", config=config)
    env.terrain[:] = 0
    env.blocks[:] = BLOCK_EMPTY
    env.agent_x[0, 0] = 2
    env.agent_y[0, 0] = 3
    env.visited[:] = False
    env.visited[0, 0, 3, 2] = True
    dense_total = 0.0
    milestone_total = 0.0
    for _ in range(12):
        _, _, _, _, info = env.step(torch.tensor([[4]]))
        dense_total += float(
            info["reward_components"][
                0, 0, REWARD_COMPONENT_INDEX["exploration"]
            ]
        )
        milestone_total += float(
            info["reward_components"][
                0, 0, REWARD_COMPONENT_INDEX["milestone"]
            ]
        )
    assert dense_total == pytest.approx(10 * config.new_cell_reward)
    assert milestone_total == pytest.approx(config.task_milestone_rewards[1])


def test_armed_attack_can_kill_and_unlock_level_eight():
    config = VGridcraftConfig(
        num_agents=1,
        max_mobs=1,
        mob_move_prob=0.0,
        mob_spawn_rate=0,
    )
    env = VectorizedGridcraftEnv(1, device="cpu", config=config)
    env.terrain[:] = 0
    env.blocks[:] = BLOCK_EMPTY
    env.agent_x[0, 0] = env.agent_y[0, 0] = 3
    env.equipped[0, 0] = ITEM_WOOD_SWORD
    env.mob_alive[0, 0] = True
    env.mob_x[0, 0] = 4
    env.mob_y[0, 0] = 3
    env.mob_hp[0, 0] = config.wood_sword_damage

    _, reward, _, _, info = env.step(torch.tensor([[7]]))

    assert not bool(env.mob_alive[0, 0])
    assert int(info["task_level_max"][0, 0]) == 8
    assert info["event_success"][0, 0, EVENT_INDEX["mob_kill_armed"]] == 1
    assert info["event_success"][0, 0, EVENT_INDEX["mob_kill_unarmed"]] == 0
    assert torch.allclose(info["reward_component_sum"], reward)


def test_mob_spawn_respects_initial_grace_period():
    config = VGridcraftConfig(
        num_agents=1,
        max_mobs=1,
        mob_spawn_grace_steps=3,
        mob_spawn_rate=1,
        max_steps=10,
    )
    env = VectorizedGridcraftEnv(1, device="cpu", config=config)
    assert not bool(env.mob_alive.any())
    env.step(torch.tensor([[0]]))
    env.step(torch.tensor([[0]]))
    assert not bool(env.mob_alive.any())
    env.step(torch.tensor([[0]]))
    assert bool(env.mob_alive.any())
