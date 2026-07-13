import torch

from torch_world_model import StructuredGridcraftWorldModel
from vgridcraft.dataset import vector_from_tabular


def _obs(batch=2, agents=3):
    grid = torch.zeros(batch, agents, 3, 7, 7, dtype=torch.long)
    grid[..., 1, :, :] = 1
    self_vec = torch.zeros(batch, agents, 11, dtype=torch.long)
    self_vec[..., :2] = 20
    return vector_from_tabular(grid, self_vec)


def test_structured_world_model_forward_and_decode_shapes():
    model = StructuredGridcraftWorldModel(
        grid_embed_dim=8,
        cnn_channels=16,
        self_hidden_size=16,
        agent_hidden_size=32,
        attention_heads=2,
        num_attention_layers=1,
        transition_hidden_size=32,
    )
    obs = _obs()
    action = torch.zeros(2, 3, dtype=torch.long)
    out, hidden = model.step(obs, action)
    assert out["terrain_logits"].shape == (2, 3, 49, 3)
    assert out["block_logits"].shape == (2, 3, 49, 4)
    assert out["entity_logits"].shape == (2, 3, 49, 4)
    assert out["self_pred"].shape == (2, 3, 11)
    assert out["reward_pred"].shape == (2, 3, 1)
    assert out["done_logit"].shape == (2, 3, 1)
    assert hidden.shape == (2, 3, 32)
    decoded = model.decode_to_obs_vector(out)
    assert decoded.shape == (2, 3, 550)


def test_structured_world_model_loss_is_finite():
    model = StructuredGridcraftWorldModel(
        grid_embed_dim=8,
        cnn_channels=16,
        self_hidden_size=16,
        agent_hidden_size=32,
        attention_heads=2,
        num_attention_layers=1,
        transition_hidden_size=32,
    )
    obs = torch.stack([_obs(), _obs()], dim=1)
    actions = torch.zeros(2, 1, 3, dtype=torch.long)
    rewards = torch.zeros(2, 1, 3)
    dones = torch.zeros(2, 1, dtype=torch.bool)
    loss, metrics = model.loss(obs, actions, rewards, dones)
    assert torch.isfinite(loss)
    assert metrics["training_structured_total_loss"] >= 0
