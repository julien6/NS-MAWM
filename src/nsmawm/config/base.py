"""Configuration dataclasses for NS-MAWM."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ModelConfig:
    n_agents: int = 2
    n_features: int = 4
    action_dim: int = 2
    latent_dim: int = 64
    hidden_dim: int = 64
    encoder_hidden: int = 128
    decoder_hidden: int = 128
    lstm_layers: int = 1
    dropout: float = 0.0


@dataclass
class TrainingConfig:
    batch_size: int = 64
    max_epochs: int = 5
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    lambda_symb: float = 1.0
    strategy: str = "reg"
    seed: int = 7


@dataclass
class DataConfig:
    dataset_size: int = 1024
    sequence_length: int = 1


@dataclass
class TrainerConfig:
    accelerator: str = "cpu"
    devices: int = 1
    log_every_n_steps: int = 10
    enable_checkpointing: bool = True


@dataclass
class NSMAWMConfig:
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()
    data: DataConfig = DataConfig()
    trainer: TrainerConfig = TrainerConfig()
