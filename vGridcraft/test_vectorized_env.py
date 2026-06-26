from __future__ import annotations

import sys
from pathlib import Path

import torch

from vgridcraft.config import VGridcraftConfig
from vgridcraft.dataset import collect_dataset
from vgridcraft.env import VectorizedGridcraftEnv


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
