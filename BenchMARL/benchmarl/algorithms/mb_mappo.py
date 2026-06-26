#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

from __future__ import annotations

import math
import sys
import warnings
from dataclasses import dataclass, field, MISSING
from pathlib import Path
from typing import Dict, List, Tuple, Type

import torch
from tensordict import TensorDict
from tensordict import TensorDictBase
from torch import nn
from torch.nn import functional as F
from torchrl.data import Categorical, OneHot

from benchmarl.algorithms.common import Algorithm
from benchmarl.algorithms.mappo import Mappo, MappoConfig


@dataclass
class WorldModelConfig:
    """Configuration for the supervised model used by MB-MAPPO."""

    enabled: bool = True
    hidden_sizes: List[int] = field(default_factory=lambda: [256, 256])
    lr: float = 3e-4
    train_epochs: int = 5
    batch_size: int = 256
    predict_delta_obs: bool = True
    predict_done: bool = False
    external_model_type: str = "mlp"
    external_checkpoint_dir: str | None = None
    external_ns_variant: str = "neural"
    external_ns_coverage: float = 0.0
    external_num_agents: int = 1


@dataclass
class ImaginedRolloutsConfig:
    """Configuration for short model-based value expansion rollouts."""

    enabled: bool = True
    horizon: int = 3
    num_branches: int = 4
    use_for_actor: bool = False
    use_for_critic: bool = True
    value_mixing: str = "lambda"
    lambda_imagined: float = 0.5
    model_uncertainty_penalty: float = 0.0


class WorldModel(nn.Module):
    """Small MLP dynamics model for one BenchMARL agent group."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_sizes: List[int],
        predict_done: bool,
    ):
        super().__init__()
        layers = []
        input_dim = obs_dim + action_dim
        for hidden_size in hidden_sizes:
            layers += [nn.Linear(input_dim, hidden_size), nn.ReLU()]
            input_dim = hidden_size
        self.backbone = nn.Sequential(*layers)
        self.obs_head = nn.Linear(input_dim, obs_dim)
        self.reward_head = nn.Linear(input_dim, 1)
        self.done_head = nn.Linear(input_dim, 1) if predict_done else None

    def forward(
        self, obs: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.backbone(torch.cat([obs, action], dim=-1))
        obs_prediction = self.obs_head(features)
        reward_prediction = self.reward_head(features)
        done_logit = (
            self.done_head(features)
            if self.done_head is not None
            else torch.zeros_like(reward_prediction)
        )
        return obs_prediction, reward_prediction, done_logit


class WorldModelTrainer:
    """Supervised trainer that fits a world model from real on-policy batches."""

    def __init__(
        self,
        model: WorldModel,
        config: WorldModelConfig,
        device: torch.device,
    ):
        self.model = model
        self.config = config
        self.device = device
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.lr)
        self.last_metrics: Dict[str, torch.Tensor] = {}

    def train(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        next_obs: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if obs.numel() == 0:
            return {}

        n_samples = obs.shape[0]
        batch_size = min(self.config.batch_size, n_samples)
        n_epochs = max(0, self.config.train_epochs)
        obs_target = next_obs - obs if self.config.predict_delta_obs else next_obs
        metrics = {}

        self.model.train()
        for _ in range(n_epochs):
            permutation = torch.randperm(n_samples, device=self.device)
            for start in range(0, n_samples, batch_size):
                indices = permutation[start : start + batch_size]
                pred_obs, pred_reward, pred_done = self.model(
                    obs[indices], action[indices]
                )
                obs_loss = F.mse_loss(pred_obs, obs_target[indices])
                reward_loss = F.mse_loss(pred_reward, reward[indices])
                if self.config.predict_done:
                    done_loss = F.binary_cross_entropy_with_logits(
                        pred_done, done[indices].to(pred_done.dtype)
                    )
                else:
                    done_loss = pred_done.sum() * 0.0
                loss = obs_loss + reward_loss + done_loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                metrics = {
                    "mb_mappo/world_model_loss": loss.detach(),
                    "mb_mappo/world_model_obs_loss": obs_loss.detach(),
                    "mb_mappo/world_model_reward_loss": reward_loss.detach(),
                    "mb_mappo/world_model_done_loss": done_loss.detach(),
                }

        self.last_metrics = metrics
        return metrics


def _expand_env_branch_agent(
    tensor: torch.Tensor, n_agents: int, num_branches: int
) -> Tuple[int, torch.Tensor]:
    """Expand a flat [env * agent, ...] tensor into [env, branch, agent, ...].

    Keeping agents contiguous inside each branch matters for joint world-model
    adapters such as Gridcraft NS-MAWM, which interpret contiguous blocks as one
    joint environment state.
    """
    if tensor.shape[0] % n_agents != 0:
        raise ValueError(
            f"Cannot expand {tensor.shape[0]} rows into complete groups of "
            f"{n_agents} agents."
        )
    env_batch = tensor.shape[0] // n_agents
    expanded = (
        tensor.reshape(env_batch, n_agents, *tensor.shape[1:])
        .unsqueeze(1)
        .expand(env_batch, num_branches, n_agents, *tensor.shape[1:])
        .reshape(env_batch * num_branches * n_agents, *tensor.shape[1:])
    )
    return env_batch, expanded


class MBMappo(Mappo):
    """Model-Based MAPPO.

    MB-MAPPO keeps MAPPO's actor update strictly on-policy by default. A
    supervised world model is trained from real on-policy batches and is used
    only to alter critic targets / advantages through short value-expansion
    rollouts.
    """

    def __init__(
        self,
        world_model: WorldModelConfig,
        imagined_rollouts: ImaginedRolloutsConfig,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if isinstance(world_model, dict):
            world_model = WorldModelConfig(**world_model)
        if isinstance(imagined_rollouts, dict):
            imagined_rollouts = ImaginedRolloutsConfig(**imagined_rollouts)
        self.world_model_config = world_model
        self.imagined_rollouts_config = imagined_rollouts
        if self.imagined_rollouts_config.use_for_actor:
            warnings.warn(
                "MB-MAPPO configured with imagined_rollouts.use_for_actor=True. "
                "This breaks the strict on-policy interpretation of MAPPO and is "
                "experimental.",
                UserWarning,
            )
        if self.imagined_rollouts_config.value_mixing != "lambda":
            raise ValueError("MB-MAPPO v1 only supports value_mixing='lambda'.")

        self._world_models: Dict[str, WorldModel] = {}
        self._world_model_trainers: Dict[str, WorldModelTrainer] = {}
        self._external_world_models: Dict[str, Dict] = {}
        self._pending_world_model_state: Dict = {}
        self.latest_metrics: Dict[str, torch.Tensor] = {}

    def process_batch(self, group: str, batch: TensorDictBase) -> TensorDictBase:
        processed_batch = super().process_batch(group, batch)
        self.latest_metrics = {}

        if (
            not self.world_model_config.enabled
            or not self.imagined_rollouts_config.enabled
            or not self.imagined_rollouts_config.use_for_critic
            or self.imagined_rollouts_config.lambda_imagined <= 0.0
            or self.imagined_rollouts_config.horizon <= 0
        ):
            return processed_batch

        try:
            data = self._extract_world_model_batch(group, processed_batch)
            if data is None:
                return processed_batch
            obs, action, next_obs, reward, done = data
            if self._uses_external_gridcraft_world_model():
                metrics = {
                    "mb_mappo/external_world_model_used": torch.tensor(1.0, device=self.device),
                    "mb_mappo/world_model_loss": torch.tensor(0.0, device=self.device),
                }
                imagined_target = self._estimate_external_gridcraft_target(group, processed_batch, obs, action)
            else:
                trainer = self._get_world_model_trainer(group, obs, action)
                metrics = trainer.train(obs, action, next_obs, reward, done)
                imagined_target = self._estimate_imagined_target(
                    group=group,
                    batch=processed_batch,
                    obs=obs,
                    action=action,
                    trainer=trainer,
                )
            if imagined_target is None:
                return processed_batch
            self._mix_value_targets(group, processed_batch, imagined_target)
            metrics.update(
                {
                    "mb_mappo/imagined_target_mean": imagined_target.mean().detach(),
                    "mb_mappo/lambda_imagined": torch.tensor(
                        self.imagined_rollouts_config.lambda_imagined,
                        device=self.device,
                    ),
                }
            )
            self.latest_metrics = metrics
        except (KeyError, RuntimeError, ValueError) as err:
            warnings.warn(
                f"MB-MAPPO skipped imagined value expansion for group '{group}': {err}",
                UserWarning,
            )
        return processed_batch

    def _uses_external_gridcraft_world_model(self) -> bool:
        return (
            self.world_model_config.enabled
            and self.world_model_config.external_model_type == "gridcraft_vae_mdn_rnn"
            and bool(self.world_model_config.external_checkpoint_dir)
        )

    def _get_observation_keys(self, group: str):
        return list(self.observation_spec[group].keys(True, True))

    def _flatten_observation(
        self, td: TensorDictBase, group: str, prefix: Tuple = ()
    ) -> Tuple[torch.Tensor, List[Tuple[Tuple, torch.Size]]]:
        tensors = []
        shapes = []
        for key in self._get_observation_keys(group):
            key_tuple = key if isinstance(key, tuple) else (key,)
            value = td.get((*prefix, group, *key_tuple)).to(self.device)
            obs_shape = self._local_observation_shape(group, key_tuple)
            obs_ndim = len(obs_shape)
            leading_shape = value.shape[:-obs_ndim] if obs_ndim else value.shape
            shapes.append((key_tuple, obs_shape))
            tensors.append(value.reshape(*leading_shape, -1))
        return torch.cat(tensors, dim=-1), shapes

    def _local_observation_shape(self, group: str, key_tuple: Tuple) -> torch.Size:
        spec_shape = self.observation_spec[(group, *key_tuple)].shape
        n_agents = len(self.group_map[group])
        if len(spec_shape) >= 2 and spec_shape[-2] == n_agents:
            return torch.Size(spec_shape[-1:])
        return torch.Size(spec_shape)

    def _extract_world_model_batch(self, group: str, batch: TensorDictBase):
        obs, _ = self._flatten_observation(batch, group)
        next_obs, _ = self._flatten_observation(batch, group, prefix=("next",))
        action = self._encode_action(
            group, batch.get((group, "action")).to(self.device)
        )
        reward = batch.get(("next", group, "reward")).to(self.device)
        done = batch.get(("next", group, "done")).to(self.device)

        obs = obs.reshape(-1, obs.shape[-1]).detach()
        next_obs = next_obs.reshape(-1, next_obs.shape[-1]).detach()
        action = action.reshape(-1, action.shape[-1]).detach()
        reward = reward.reshape(-1, reward.shape[-1]).detach()
        done = done.reshape(-1, done.shape[-1]).detach().to(torch.float)
        return obs, action, next_obs, reward, done

    def _encode_action(self, group: str, action: torch.Tensor) -> torch.Tensor:
        action_space = self.action_spec[group, "action"]
        if isinstance(action_space, Categorical):
            n = action_space.space.n
            if action.shape[-1:] == (1,):
                action = action.squeeze(-1)
            encoded = F.one_hot(action.to(torch.long), num_classes=n)
            return encoded.to(torch.float)
        if isinstance(action_space, OneHot):
            return action.to(torch.float)
        return action.to(torch.float)

    def _get_world_model_trainer(
        self, group: str, obs: torch.Tensor, action: torch.Tensor
    ) -> WorldModelTrainer:
        if group not in self._world_model_trainers:
            model = WorldModel(
                obs_dim=obs.shape[-1],
                action_dim=action.shape[-1],
                hidden_sizes=self.world_model_config.hidden_sizes,
                predict_done=self.world_model_config.predict_done,
            ).to(self.device)
            self._world_models[group] = model
            self._world_model_trainers[group] = WorldModelTrainer(
                model=model,
                config=self.world_model_config,
                device=torch.device(self.device),
            )
            if group in self._pending_world_model_state:
                state = self._pending_world_model_state.pop(group)
                self._world_models[group].load_state_dict(state["model"])
                self._world_model_trainers[group].optimizer.load_state_dict(
                    state["optimizer"]
                )
        return self._world_model_trainers[group]

    def _get_external_gridcraft_world_model(self, group: str) -> Dict:
        if group in self._external_world_models:
            return self._external_world_models[group]
        checkpoint_dir = Path(self.world_model_config.external_checkpoint_dir).expanduser()
        if not checkpoint_dir.is_absolute():
            checkpoint_dir = Path.cwd() / checkpoint_dir
        vae_path = checkpoint_dir / "vae.pt"
        rnn_path = checkpoint_dir / "rnn.pt"
        if not vae_path.exists() or not rnn_path.exists():
            raise FileNotFoundError(f"Missing external Gridcraft world model checkpoints: {vae_path} / {rnn_path}")
        root = checkpoint_dir
        for parent in checkpoint_dir.parents:
            if (parent / "gridcraft").exists() and (parent / "vGridcraft").exists():
                root = parent
                break
        gridcraft_dir = root / "gridcraft"
        if str(gridcraft_dir) not in sys.path:
            sys.path.insert(0, str(gridcraft_dir))
        from torch_world_model import TorchGridcraftRNN, TorchGridcraftVAE
        from run_benchmarl_dyna_gridcraft import apply_ns_mawm_to_latent_step

        vae = TorchGridcraftVAE().to(self.device)
        rnn = TorchGridcraftRNN().to(self.device)
        vae.load_state_dict(torch.load(vae_path, map_location=self.device))
        rnn.load_state_dict(torch.load(rnn_path, map_location=self.device))
        vae.eval()
        rnn.eval()
        for param in vae.parameters():
            param.requires_grad_(False)
        for param in rnn.parameters():
            param.requires_grad_(False)
        state = {
            "vae": vae,
            "rnn": rnn,
            "apply_ns": apply_ns_mawm_to_latent_step,
        }
        self._external_world_models[group] = state
        return state

    @torch.no_grad()
    def _estimate_imagined_target(
        self,
        group: str,
        batch: TensorDictBase,
        obs: torch.Tensor,
        action: torch.Tensor,
        trainer: WorldModelTrainer,
    ) -> torch.Tensor:
        horizon = self.imagined_rollouts_config.horizon
        num_branches = max(1, self.imagined_rollouts_config.num_branches)
        gamma = self.experiment_config.gamma
        n_agents = len(self.group_map[group])
        env_batch, obs_rollout = _expand_env_branch_agent(
            obs, n_agents=n_agents, num_branches=num_branches
        )
        returns = torch.zeros(obs_rollout.shape[0], 1, device=self.device)
        discount = torch.ones_like(returns)

        trainer.model.eval()
        for _ in range(horizon):
            action_rollout = self._sample_encoded_action_from_flat_obs(
                group, batch, obs_rollout
            )
            obs_prediction, reward_prediction, done_logit = trainer.model(
                obs_rollout, action_rollout
            )
            next_obs = (
                obs_rollout + obs_prediction
                if self.world_model_config.predict_delta_obs
                else obs_prediction
            )
            done_prob = (
                torch.sigmoid(done_logit)
                if self.world_model_config.predict_done
                else torch.zeros_like(reward_prediction)
            )
            reward_prediction = reward_prediction - (
                self.imagined_rollouts_config.model_uncertainty_penalty * done_prob
            )
            returns = returns + discount * reward_prediction
            discount = discount * gamma * (1.0 - done_prob)
            obs_rollout = next_obs

        bootstrap_value = self._critic_value_from_flat_obs(group, batch, obs_rollout)
        returns = returns + discount * bootstrap_value
        returns = returns.reshape(env_batch, num_branches, n_agents, 1).mean(dim=1).reshape(env_batch * n_agents, 1)
        return returns

    @torch.no_grad()
    def _estimate_external_gridcraft_target(
        self,
        group: str,
        batch: TensorDictBase,
        obs: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        state = self._get_external_gridcraft_world_model(group)
        vae = state["vae"]
        rnn = state["rnn"]
        apply_ns = state["apply_ns"]
        horizon = self.imagined_rollouts_config.horizon
        num_branches = max(1, self.imagined_rollouts_config.num_branches)
        gamma = self.experiment_config.gamma
        num_agents = max(1, int(self.world_model_config.external_num_agents))
        env_batch, obs_rollout = _expand_env_branch_agent(
            obs, n_agents=num_agents, num_branches=num_branches
        )
        z = vae.encode(obs_rollout, sample=False)
        returns = torch.zeros(z.shape[0], 1, device=self.device)
        discount = torch.ones_like(returns)
        rnn_state = None
        ns_memory = None
        for _ in range(horizon):
            current_obs = vae.decode(z)
            action_rollout = self._sample_encoded_action_from_flat_obs(
                group, batch, current_obs
            )
            if action_rollout.shape[-1] > 1:
                action_index = action_rollout.argmax(dim=-1)
            else:
                action_index = action_rollout.reshape(-1).long()
            next_z, reward, done_logit, rnn_state = rnn.step(z, action_index, rnn_state, deterministic=True)
            if self.world_model_config.external_ns_variant in ("projection", "residual"):
                next_z, ns_memory, _ = apply_ns(
                    vae=vae,
                    current_z=z,
                    predicted_z=next_z,
                    action=action_index,
                    ns_memory=ns_memory,
                    ns_variant=self.world_model_config.external_ns_variant,
                    ns_coverage=float(self.world_model_config.external_ns_coverage),
                    num_agents=num_agents,
                    device=torch.device(self.device),
                )
            done_prob = torch.sigmoid(done_logit).reshape(-1, 1)
            reward = reward.reshape(-1, 1)
            returns = returns + discount * reward
            discount = discount * gamma * (1.0 - done_prob)
            z = next_z
        imagined_obs = vae.decode(z)
        bootstrap_value = self._critic_value_from_flat_obs(group, batch, imagined_obs)
        returns = returns + discount * bootstrap_value
        returns = returns.reshape(env_batch, num_branches, num_agents, 1).mean(dim=1).reshape(env_batch * num_agents, 1)
        return returns

    def _flat_obs_to_tensordict(
        self, group: str, reference_batch: TensorDictBase, flat_obs: torch.Tensor
    ) -> TensorDict:
        keys = self._get_observation_keys(group)
        n_agents = len(self.group_map[group])
        if flat_obs.shape[0] % n_agents != 0:
            raise ValueError("Flat imagined observations are not divisible by n_agents.")
        batch_size = flat_obs.shape[0] // n_agents
        td = TensorDict({}, batch_size=(batch_size,), device=self.device)

        cursor = 0
        for key in keys:
            key_tuple = key if isinstance(key, tuple) else (key,)
            obs_shape = self._local_observation_shape(group, key_tuple)
            obs_numel = math.prod(obs_shape)
            value = flat_obs[:, cursor : cursor + obs_numel].reshape(
                batch_size, n_agents, *obs_shape
            )
            td.set((group, *key_tuple), value)
            cursor += obs_numel

        if self.state_spec is not None and len(self.state_spec.keys(True, True)):
            state_key = self.state_spec.keys(True, True)[0]
            state_key = state_key if isinstance(state_key, tuple) else (state_key,)
            state = reference_batch.get(state_key)
            state_shape = self.state_spec[state_key].shape
            state = state.reshape(-1, *state.shape[-len(state_shape) :])
            if state.shape[0] == batch_size:
                td.set(state_key, state.to(self.device))
            else:
                td.set(
                    state_key,
                    state[:1].expand(batch_size, *state.shape[1:]).to(self.device),
                )
        return td

    def _critic_value_from_flat_obs(
        self, group: str, reference_batch: TensorDictBase, flat_obs: torch.Tensor
    ) -> torch.Tensor:
        td = self._flat_obs_to_tensordict(group, reference_batch, flat_obs)
        critic_td = self.get_critic(group)(td)
        return critic_td.get((group, "state_value")).reshape(-1, 1)

    def _sample_encoded_action_from_flat_obs(
        self, group: str, reference_batch: TensorDictBase, flat_obs: torch.Tensor
    ) -> torch.Tensor:
        td = self._flat_obs_to_tensordict(group, reference_batch, flat_obs)
        action_td = self.get_policy_for_loss(group)(td)
        encoded = self._encode_action(group, action_td.get((group, "action")))
        return encoded.reshape(-1, encoded.shape[-1])

    def _mix_value_targets(
        self, group: str, batch: TensorDictBase, imagined_target: torch.Tensor
    ) -> None:
        value_target = batch.get((group, "value_target"))
        state_value = batch.get((group, "state_value"))
        lambda_imagined = self.imagined_rollouts_config.lambda_imagined
        imagined_target = imagined_target.reshape_as(value_target)
        mixed_target = (
            (1.0 - lambda_imagined) * value_target
            + lambda_imagined * imagined_target.to(value_target.device)
        )
        batch.set((group, "value_target"), mixed_target)
        batch.set((group, "advantage"), mixed_target - state_value)

    def state_dict(self) -> Dict:
        return {
            group: {
                "model": model.state_dict(),
                "optimizer": self._world_model_trainers[group].optimizer.state_dict(),
            }
            for group, model in self._world_models.items()
        }

    def load_state_dict(self, state_dict: Dict) -> None:
        self._pending_world_model_state = dict(state_dict)
        for group, value in state_dict.items():
            if group not in self._world_model_trainers:
                continue
            self._world_models[group].load_state_dict(value["model"])
            self._world_model_trainers[group].optimizer.load_state_dict(
                value["optimizer"]
            )
            self._pending_world_model_state.pop(group, None)


@dataclass
class MBMappoConfig(MappoConfig):
    """Configuration dataclass for :class:`~benchmarl.algorithms.MBMappo`."""

    share_param_critic: bool = MISSING
    clip_epsilon: float = MISSING
    entropy_coef: float = MISSING
    critic_coef: float = MISSING
    loss_critic_type: str = MISSING
    lmbda: float = MISSING
    scale_mapping: str = MISSING
    use_tanh_normal: bool = MISSING
    minibatch_advantage: bool = MISSING
    world_model: WorldModelConfig = field(default_factory=WorldModelConfig)
    imagined_rollouts: ImaginedRolloutsConfig = field(
        default_factory=ImaginedRolloutsConfig
    )

    @staticmethod
    def associated_class() -> Type[Algorithm]:
        return MBMappo

    @classmethod
    def get_from_yaml(cls, path=None):
        config = super().get_from_yaml(path)
        if isinstance(config.world_model, dict):
            config.world_model = WorldModelConfig(**config.world_model)
        if isinstance(config.imagined_rollouts, dict):
            config.imagined_rollouts = ImaginedRolloutsConfig(
                **config.imagined_rollouts
            )
        return config

    @staticmethod
    def supports_continuous_actions() -> bool:
        return True

    @staticmethod
    def supports_discrete_actions() -> bool:
        return True

    @staticmethod
    def on_policy() -> bool:
        return True

    @staticmethod
    def has_centralized_critic() -> bool:
        return True
