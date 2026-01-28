"""Neural backbone for MAWM."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn


@dataclass
class BackboneConfig:
    n_agents: int
    n_features: int
    action_dim: int
    latent_dim: int
    hidden_dim: int
    encoder_hidden: int
    decoder_hidden: int
    lstm_layers: int = 1
    dropout: float = 0.0


class MAWMBackbone(nn.Module):
    """Encoder + LSTM latent dynamics + decoder."""

    def __init__(self, config: BackboneConfig):
        super().__init__()
        self.config = config
        in_dim = config.n_agents * config.n_features
        act_dim = config.n_agents * config.action_dim
        self.encoder = nn.Sequential(
            nn.Linear(in_dim + act_dim, config.encoder_hidden),
            nn.ReLU(),
            nn.Linear(config.encoder_hidden, config.latent_dim),
            nn.ReLU(),
        )
        self.core = nn.LSTM(
            input_size=config.latent_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.lstm_layers,
            batch_first=True,
            dropout=config.dropout if config.lstm_layers > 1 else 0.0,
        )
        self.decoder = nn.Sequential(
            nn.Linear(config.hidden_dim, config.decoder_hidden),
            nn.ReLU(),
            nn.Linear(config.decoder_hidden, in_dim),
        )

    def forward(
        self,
        obs_t: torch.Tensor,
        act_t: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Predict next observation and return new hidden state."""
        batch_size = obs_t.shape[0]
        obs_flat = obs_t.reshape(batch_size, -1)
        act_flat = act_t.reshape(batch_size, -1)
        enc_input = torch.cat([obs_flat, act_flat], dim=-1)
        latent = self.encoder(enc_input)
        latent = latent.unsqueeze(1)
        out, hidden = self.core(latent, hidden)
        decoded = self.decoder(out[:, 0])
        next_obs = decoded.view(batch_size, self.config.n_agents, self.config.n_features)
        return next_obs, hidden

    def predict_step(
        self,
        obs_t: torch.Tensor,
        act_t: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        return self.forward(obs_t, act_t, hidden)
