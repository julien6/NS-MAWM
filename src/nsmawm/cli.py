"""CLI for NS-MAWM."""

from __future__ import annotations

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from rich import print

from nsmawm.config.base import NSMAWMConfig
from nsmawm.data.datasets import TransitionsDataset
from nsmawm.models.mawm_backbone import BackboneConfig
from nsmawm.models.nsmawm import NSMAWM
from nsmawm.symbolic.engine import RuleEngine
from nsmawm.training.trainer import fit


@hydra.main(config_path=None, config_name=None)
def main(cfg: DictConfig) -> None:
    config = OmegaConf.merge(OmegaConf.structured(NSMAWMConfig()), cfg)
    print("[bold cyan]NS-MAWM Config[/bold cyan]")
    print(OmegaConf.to_yaml(config))

    model_cfg = config.model
    backbone_cfg = BackboneConfig(
        n_agents=model_cfg.n_agents,
        n_features=model_cfg.n_features,
        action_dim=model_cfg.action_dim,
        latent_dim=model_cfg.latent_dim,
        hidden_dim=model_cfg.hidden_dim,
        encoder_hidden=model_cfg.encoder_hidden,
        decoder_hidden=model_cfg.decoder_hidden,
        lstm_layers=model_cfg.lstm_layers,
        dropout=model_cfg.dropout,
    )

    rule_engine = RuleEngine([])
    model = NSMAWM.from_config(
        backbone_cfg,
        rule_engine=rule_engine,
        strategy=config.training.strategy,
        lambda_symb=config.training.lambda_symb,
    )

    obs = torch.randn(config.data.dataset_size, model_cfg.n_agents, model_cfg.n_features)
    act = torch.randn(config.data.dataset_size, model_cfg.n_agents, model_cfg.action_dim)
    next_obs = torch.randn_like(obs)

    dataset = TransitionsDataset(obs, act, next_obs)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=config.training.batch_size, shuffle=True)

    fit(
        model,
        train_loader,
        max_epochs=config.training.max_epochs,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        accelerator=config.trainer.accelerator,
        devices=config.trainer.devices,
        log_every_n_steps=config.trainer.log_every_n_steps,
        enable_checkpointing=config.trainer.enable_checkpointing,
    )


if __name__ == "__main__":
    main()
