#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

from __future__ import annotations

import random
from dataclasses import MISSING, dataclass, field
from typing import Dict, List, Optional, Tuple, Type

import torch
from tensordict import TensorDictBase
from torch import nn
from torch.nn import functional as F
from torchrl.data import Categorical, LazyTensorStorage, OneHot, TensorDictReplayBuffer
from torchrl.data.replay_buffers import RandomSampler

from benchmarl.algorithms.common import Algorithm
from benchmarl.algorithms.masac import Masac, MasacConfig


@dataclass
class MambpoWorldModelConfig:
    """Configuration for the supervised ensemble dynamics model."""

    enabled: bool = True
    n_models: int = 5
    n_elites: int = 5
    hidden_sizes: List[int] = field(default_factory=lambda: [200, 200, 200, 200])
    lr: float = 1e-3
    train_steps: int = 50
    train_interval: int = 250
    batch_size: int = 512
    stochastic: bool = True
    predict_delta_obs: bool = True
    predict_done: bool = False


@dataclass
class MambpoImaginedRolloutsConfig:
    """Configuration for model-generated replay data."""

    enabled: bool = True
    rollout_length: int = 1
    rollout_schedule: Optional[List[List[int]]] = None
    model_batch_size: int = 2048
    real_ratio: float = 0.1
    diverse_actions: bool = True
    model_buffer_size: int = 100000


class EnsembleWorldModel(nn.Module):
    """Probabilistic ensemble model for joint multi-agent dynamics."""

    def __init__(
        self,
        input_dim: int,
        obs_dim: int,
        reward_dim: int,
        hidden_sizes: List[int],
        n_models: int,
        n_elites: int,
        stochastic: bool,
        predict_done: bool,
    ):
        super().__init__()
        self.n_models = n_models
        self.n_elites = min(n_elites, n_models)
        self.stochastic = stochastic
        self.predict_done = predict_done
        self.models = nn.ModuleList(
            [
                _SingleWorldModel(
                    input_dim=input_dim,
                    obs_dim=obs_dim,
                    reward_dim=reward_dim,
                    hidden_sizes=hidden_sizes,
                    stochastic=stochastic,
                    predict_done=predict_done,
                )
                for _ in range(n_models)
            ]
        )
        self.elite_indices = list(range(self.n_elites))

    def forward(
        self, obs: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        inputs = torch.cat([obs, action], dim=-1)
        outs = [model(inputs) for model in self.models]
        obs_mu, obs_log_var, reward_mu, reward_log_var, done_logit = zip(*outs)
        return (
            torch.stack(obs_mu),
            torch.stack(obs_log_var),
            torch.stack(reward_mu),
            torch.stack(reward_log_var),
            torch.stack(done_logit),
        )

    @torch.no_grad()
    def sample(
        self, obs: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        obs_mu, obs_log_var, reward_mu, reward_log_var, done_logit = self(obs, action)
        elite = random.choice(self.elite_indices)
        if self.stochastic:
            next_obs = torch.normal(obs_mu[elite], torch.exp(0.5 * obs_log_var[elite]))
            reward = torch.normal(
                reward_mu[elite], torch.exp(0.5 * reward_log_var[elite])
            )
        else:
            next_obs = obs_mu[elite]
            reward = reward_mu[elite]
        done = torch.sigmoid(done_logit[elite])
        return next_obs, reward, done


class _SingleWorldModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        obs_dim: int,
        reward_dim: int,
        hidden_sizes: List[int],
        stochastic: bool,
        predict_done: bool,
    ):
        super().__init__()
        self.stochastic = stochastic
        self.predict_done = predict_done
        self.max_log_var = 0.5
        self.min_log_var = -10.0
        layers = []
        last_dim = input_dim
        for hidden_size in hidden_sizes:
            layers += [nn.Linear(last_dim, hidden_size), nn.SiLU()]
            last_dim = hidden_size
        self.backbone = nn.Sequential(*layers)
        self.obs_mu = nn.Linear(last_dim, obs_dim)
        self.obs_log_var = nn.Linear(last_dim, obs_dim)
        self.reward_mu = nn.Linear(last_dim, reward_dim)
        self.reward_log_var = nn.Linear(last_dim, reward_dim)
        self.done_logit = nn.Linear(last_dim, reward_dim) if predict_done else None

    def _bound_log_var(self, log_var: torch.Tensor) -> torch.Tensor:
        if not self.stochastic:
            return torch.zeros_like(log_var)
        log_var = self.max_log_var - F.softplus(self.max_log_var - log_var)
        return self.min_log_var + F.softplus(log_var - self.min_log_var)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.backbone(x)
        obs_log_var = self._bound_log_var(self.obs_log_var(features))
        reward_log_var = self._bound_log_var(self.reward_log_var(features))
        if self.done_logit is None:
            done_logit = torch.zeros(
                *features.shape[:-1],
                self.reward_mu.out_features,
                device=features.device,
                dtype=features.dtype,
            )
        else:
            done_logit = self.done_logit(features)
        return (
            self.obs_mu(features),
            obs_log_var,
            self.reward_mu(features),
            reward_log_var,
            done_logit,
        )


class WorldModelTrainer:
    def __init__(
        self,
        model: EnsembleWorldModel,
        config: MambpoWorldModelConfig,
        device: torch.device,
    ):
        self.model = model
        self.config = config
        self.device = device
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.lr)

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
        obs_target = next_obs - obs if self.config.predict_delta_obs else next_obs
        metrics = {}

        self.model.train()
        for _ in range(max(0, self.config.train_steps)):
            indices = torch.randint(n_samples, (batch_size,), device=self.device)
            obs_mu, obs_log_var, reward_mu, reward_log_var, done_logit = self.model(
                obs[indices], action[indices]
            )
            target_obs = obs_target[indices].unsqueeze(0)
            target_reward = reward[indices].unsqueeze(0)
            inv_obs_var = torch.exp(-obs_log_var)
            inv_reward_var = torch.exp(-reward_log_var)
            obs_loss = (
                (obs_mu - target_obs).pow(2) * inv_obs_var + obs_log_var
            ).mean()
            reward_loss = (
                (reward_mu - target_reward).pow(2) * inv_reward_var + reward_log_var
            ).mean()
            if self.config.predict_done:
                done_loss = F.binary_cross_entropy_with_logits(
                    done_logit, done[indices].to(done_logit.dtype).unsqueeze(0)
                )
            else:
                done_loss = done_logit.sum() * 0.0
            loss = obs_loss + reward_loss + done_loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            metrics = {
                "mambpo/world_model_loss": loss.detach(),
                "mambpo/world_model_obs_loss": obs_loss.detach(),
                "mambpo/world_model_reward_loss": reward_loss.detach(),
            }
            if self.config.predict_done:
                metrics["mambpo/world_model_done_loss"] = done_loss.detach()
        return metrics


class _MixedReplayBuffer:
    """Replay-buffer wrapper that samples real/model data without changing Experiment."""

    def __init__(self, real_buffer, algorithm: "Mambpo", group: str):
        self.real_buffer = real_buffer
        self.algorithm = algorithm
        self.group = group

    @property
    def storage(self):
        return self.real_buffer.storage

    def __len__(self):
        return len(self.real_buffer)

    def extend(self, *args, **kwargs):
        return self.real_buffer.extend(*args, **kwargs)

    def update_tensordict_priority(self, *args, **kwargs):
        return self.real_buffer.update_tensordict_priority(*args, **kwargs)

    def state_dict(self):
        return self.real_buffer.state_dict()

    def load_state_dict(self, state_dict):
        return self.real_buffer.load_state_dict(state_dict)

    def sample(self):
        model_buffer = self.algorithm._model_replay_buffers.get(self.group)
        if (
            model_buffer is None
            or len(model_buffer) == 0
            or not self.algorithm.imagined_rollouts.enabled
            or self.algorithm.imagined_rollouts.real_ratio >= 1.0
        ):
            self.algorithm._last_sampling_metrics = {
                "mambpo/real_batch_size": torch.tensor(
                    float(self.algorithm.experiment_config.train_batch_size(False)),
                    device=self.algorithm.device,
                ),
                "mambpo/imagined_batch_size": torch.tensor(
                    0.0, device=self.algorithm.device
                ),
            }
            return self.real_buffer.sample()

        target_batch = self.algorithm.experiment_config.train_batch_size(False)
        n_real = int(target_batch * self.algorithm.imagined_rollouts.real_ratio)
        n_real = min(max(n_real, 1), target_batch)
        n_model = target_batch - n_real
        if n_model <= 0:
            self.algorithm._last_sampling_metrics = {
                "mambpo/real_batch_size": torch.tensor(
                    float(target_batch), device=self.algorithm.device
                ),
                "mambpo/imagined_batch_size": torch.tensor(
                    0.0, device=self.algorithm.device
                ),
            }
            return self.real_buffer.sample(batch_size=target_batch)

        real = self.real_buffer.sample(batch_size=n_real)
        model = model_buffer.sample(batch_size=n_model)
        reward_key = ("next", self.group, "reward")
        imagined_reward = (
            model.get(reward_key).float().mean().to(self.algorithm.device)
            if reward_key in model.keys(True, True)
            else torch.tensor(0.0, device=self.algorithm.device)
        )
        self.algorithm._last_sampling_metrics = {
            "mambpo/real_batch_size": torch.tensor(
                float(n_real), device=self.algorithm.device
            ),
            "mambpo/imagined_batch_size": torch.tensor(
                float(n_model), device=self.algorithm.device
            ),
            "mambpo/training_sampled_imagined_reward": imagined_reward,
        }
        return torch.cat([real, model.to(real.device)], dim=0)


class Mambpo(Masac):
    """Model-Based Multi-Agent SAC with short ensemble-model rollouts."""

    def __init__(
        self,
        world_model: MambpoWorldModelConfig,
        imagined_rollouts: MambpoImaginedRolloutsConfig,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if isinstance(world_model, dict):
            world_model = MambpoWorldModelConfig(**world_model)
        if isinstance(imagined_rollouts, dict):
            imagined_rollouts = MambpoImaginedRolloutsConfig(**imagined_rollouts)
        self.world_model = world_model
        self.imagined_rollouts = imagined_rollouts
        self._world_models: Dict[str, EnsembleWorldModel] = {}
        self._world_model_trainers: Dict[str, WorldModelTrainer] = {}
        self._model_replay_buffers: Dict[str, TensorDictReplayBuffer] = {}
        self._steps: Dict[str, int] = {}
        self.latest_metrics: Dict[str, torch.Tensor] = {}
        self._last_sampling_metrics: Dict[str, torch.Tensor] = {}
        self._pending_world_model_state: Dict = {}

    def get_replay_buffer(self, group: str, transforms=None):
        real_buffer = super().get_replay_buffer(group, transforms)
        return _MixedReplayBuffer(real_buffer=real_buffer, algorithm=self, group=group)

    def process_batch(self, group: str, batch: TensorDictBase) -> TensorDictBase:
        batch = super().process_batch(group, batch)
        self.latest_metrics = dict(self._last_sampling_metrics)
        self._steps[group] = self._steps.get(group, 0) + 1

        if not self.world_model.enabled or not self.imagined_rollouts.enabled:
            return batch

        data = self._extract_model_data(group, batch)
        if data is None:
            return batch
        obs, action, next_obs, reward, done, flat_batch = data
        trainer = self._get_world_model_trainer(group, obs, action, reward)

        if self._steps[group] % max(1, self.world_model.train_interval) == 0:
            self.latest_metrics.update(
                trainer.train(obs, action, next_obs, reward, done)
            )

        if len(flat_batch) > 0:
            fake = self._generate_model_rollouts(group, flat_batch, trainer.model)
            if fake is not None and len(fake) > 0:
                model_buffer = self._get_model_replay_buffer(group)
                model_buffer.extend(fake.to(model_buffer.storage.device))
                imagined_metrics = self._summarize_imagined_transitions(group, fake)
                self.latest_metrics.update(
                    {
                        **imagined_metrics,
                        "mambpo/model_rollout_length": torch.tensor(
                            float(self._current_rollout_length()), device=self.device
                        ),
                        "mambpo/real_ratio": torch.tensor(
                            self.imagined_rollouts.real_ratio, device=self.device
                        ),
                        "mambpo/imagined_ratio": torch.tensor(
                            1.0 - self.imagined_rollouts.real_ratio,
                            device=self.device,
                        ),
                        "mambpo/model_buffer_size": torch.tensor(
                            float(len(model_buffer)), device=self.device
                        ),
                        "mambpo/model_batch_size": torch.tensor(
                            float(self.imagined_rollouts.model_batch_size),
                            device=self.device,
                        ),
                    }
                )

        return batch

    def _extract_model_data(self, group: str, batch: TensorDictBase):
        obs_key = (group, "observation")
        next_obs_key = ("next", group, "observation")
        action_key = (group, "action")
        reward_key = ("next", group, "reward")
        done_key = ("next", group, "done")
        if not all(key in batch.keys(True, True) for key in (obs_key, next_obs_key, action_key, reward_key, done_key)):
            return None

        flat_batch = batch.reshape(-1)
        obs = flat_batch.get(obs_key).to(self.device)
        next_obs = flat_batch.get(next_obs_key).to(self.device)
        action_raw = flat_batch.get(action_key).to(self.device)
        reward = flat_batch.get(reward_key).to(self.device)
        done = flat_batch.get(done_key).to(self.device)

        obs_flat = obs.reshape(obs.shape[0], -1).float()
        next_obs_flat = next_obs.reshape(next_obs.shape[0], -1).float()
        action = self._encode_action(group, action_raw)
        reward_flat = reward.reshape(reward.shape[0], -1).float()
        done_flat = done.reshape(done.shape[0], -1).float()
        return obs_flat, action, next_obs_flat, reward_flat, done_flat, flat_batch

    def _encode_action(self, group: str, action: torch.Tensor) -> torch.Tensor:
        action_space = self.action_spec[group, "action"]
        if isinstance(action_space, Categorical):
            n_actions = action_space.space.n
            action_long = action.long().squeeze(-1)
            return F.one_hot(action_long, num_classes=n_actions).reshape(
                action.shape[0], -1
            ).float()
        if isinstance(action_space, OneHot):
            return action.reshape(action.shape[0], -1).float()
        return action.reshape(action.shape[0], -1).float()

    def _get_world_model_trainer(
        self,
        group: str,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
    ) -> WorldModelTrainer:
        if group not in self._world_model_trainers:
            model = EnsembleWorldModel(
                input_dim=obs.shape[-1] + action.shape[-1],
                obs_dim=obs.shape[-1],
                reward_dim=reward.shape[-1],
                hidden_sizes=self.world_model.hidden_sizes,
                n_models=self.world_model.n_models,
                n_elites=self.world_model.n_elites,
                stochastic=self.world_model.stochastic,
                predict_done=self.world_model.predict_done,
            ).to(self.device)
            trainer = WorldModelTrainer(model, self.world_model, self.device)
            self._world_models[group] = model
            self._world_model_trainers[group] = trainer
            if group in self._pending_world_model_state:
                state = self._pending_world_model_state.pop(group)
                model.load_state_dict(state["model"])
                trainer.optimizer.load_state_dict(state["optimizer"])
        return self._world_model_trainers[group]

    def _get_model_replay_buffer(self, group: str) -> TensorDictReplayBuffer:
        if group not in self._model_replay_buffers:
            self._model_replay_buffers[group] = TensorDictReplayBuffer(
                storage=LazyTensorStorage(
                    self.imagined_rollouts.model_buffer_size,
                    device=self.buffer_device,
                ),
                sampler=RandomSampler(),
                batch_size=self.experiment_config.train_batch_size(False),
                priority_key=(group, "td_error"),
            )
        return self._model_replay_buffers[group]

    def _current_rollout_length(self) -> int:
        rollout_length = self.imagined_rollouts.rollout_length
        schedule = self.imagined_rollouts.rollout_schedule
        if schedule:
            total_step = max(self._steps.values(), default=0)
            for step, value in sorted(schedule):
                if total_step >= step:
                    rollout_length = value
        return max(1, int(rollout_length))

    @torch.no_grad()
    def _generate_model_rollouts(
        self,
        group: str,
        flat_batch: TensorDictBase,
        model: EnsembleWorldModel,
    ) -> Optional[TensorDictBase]:
        n_roots = min(self.imagined_rollouts.model_batch_size, len(flat_batch))
        if n_roots <= 0:
            return None
        indices = torch.randint(len(flat_batch), (n_roots,), device=flat_batch.device)
        roots = flat_batch[indices].clone()
        current_obs = roots.get((group, "observation")).to(self.device)
        transitions = []
        policy = self.get_policy_for_collection()
        for _ in range(self._current_rollout_length()):
            td = roots.clone()
            td.set((group, "observation"), current_obs.to(td.device))
            if self.imagined_rollouts.diverse_actions:
                policy(td.to(self.device))
            action_raw = td.get((group, "action")).to(self.device)
            action = self._encode_action(group, action_raw)
            obs_flat = current_obs.reshape(current_obs.shape[0], -1).float()
            pred_obs, pred_reward, pred_done = model.sample(obs_flat, action)
            if self.world_model.predict_delta_obs:
                pred_obs = obs_flat + pred_obs
            next_obs = pred_obs.reshape_as(current_obs)
            reward = pred_reward.reshape_as(td.get(("next", group, "reward")))
            done_prob = pred_done.reshape_as(td.get(("next", group, "done")))
            done = done_prob > 0.5 if self.world_model.predict_done else torch.zeros_like(done_prob, dtype=torch.bool)
            transition = td.clone()
            transition.set((group, "action"), action_raw.to(transition.device))
            transition.set(("next", group, "observation"), next_obs.to(transition.device))
            transition.set(("next", group, "reward"), reward.to(transition.device))
            transition.set(("next", group, "done"), done.to(transition.device))
            transition.set(("next", group, "terminated"), done.to(transition.device))
            if ("next", "done") in transition.keys(True, True):
                transition.set(("next", "done"), done.any(dim=-2).to(transition.device))
            if ("next", "terminated") in transition.keys(True, True):
                transition.set(("next", "terminated"), done.any(dim=-2).to(transition.device))
            transitions.append(transition)
            current_obs = next_obs
        return torch.cat(transitions, dim=0)

    def _summarize_imagined_transitions(
        self, group: str, transitions: TensorDictBase
    ) -> Dict[str, torch.Tensor]:
        reward_key = ("next", group, "reward")
        if reward_key not in transitions.keys(True, True):
            zero = torch.tensor(0.0, device=self.device)
            return {
                "mambpo/training_imagined_reward": zero,
                "mambpo/training_imagined_reward_mean_step": zero,
            }
        rewards = transitions.get(reward_key).to(self.device).float()
        rollout_length = self._current_rollout_length()
        n_roots = max(1, rewards.shape[0] // rollout_length)
        rewards = rewards[: n_roots * rollout_length]
        reward_view = rewards.reshape(rollout_length, n_roots, *rewards.shape[1:])
        return {
            "mambpo/training_imagined_reward": reward_view.sum(dim=0).mean().detach(),
            "mambpo/training_imagined_reward_mean_step": rewards.mean().detach(),
        }

    @torch.no_grad()
    def evaluate_imagined_rollouts(
        self, group: str, batch: TensorDictBase
    ) -> Dict[str, torch.Tensor]:
        if (
            group not in self._world_models
            or not self.world_model.enabled
            or not self.imagined_rollouts.enabled
        ):
            return {}
        flat_batch = batch.reshape(-1).to(self.device)
        fake = self._generate_model_rollouts(group, flat_batch, self._world_models[group])
        if fake is None or len(fake) == 0:
            return {}
        metrics = self._summarize_imagined_transitions(group, fake)
        imagined_return = metrics["mambpo/training_imagined_reward"]
        result = {
            "mambpo/eval_imagined_reward": imagined_return.detach(),
            "mambpo/eval_imagined_episode_length": torch.tensor(
                float(self._current_rollout_length()), device=self.device
            ),
        }
        reward_key = ("next", group, "reward")
        if reward_key in flat_batch.keys(True, True):
            real_return = flat_batch.get(reward_key).float().mean().to(
                self.device
            ) * float(self._current_rollout_length())
            result["mambpo/real_imagined_reward_gap"] = (
                real_return - imagined_return
            ).detach()
        return result

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
        for group, value in list(state_dict.items()):
            if group not in self._world_model_trainers:
                continue
            self._world_models[group].load_state_dict(value["model"])
            self._world_model_trainers[group].optimizer.load_state_dict(
                value["optimizer"]
            )
            self._pending_world_model_state.pop(group, None)


@dataclass
class MambpoConfig(MasacConfig):
    """Configuration dataclass for :class:`~benchmarl.algorithms.Mambpo`."""

    share_param_critic: bool = MISSING
    num_qvalue_nets: int = MISSING
    loss_function: str = MISSING
    delay_qvalue: bool = MISSING
    target_entropy: float | str = MISSING
    discrete_target_entropy_weight: float = MISSING
    alpha_init: float = MISSING
    min_alpha: Optional[float] = MISSING
    max_alpha: Optional[float] = MISSING
    fixed_alpha: bool = MISSING
    scale_mapping: str = MISSING
    use_tanh_normal: bool = MISSING
    coupled_discrete_values: bool = MISSING
    world_model: MambpoWorldModelConfig = field(default_factory=MambpoWorldModelConfig)
    imagined_rollouts: MambpoImaginedRolloutsConfig = field(
        default_factory=MambpoImaginedRolloutsConfig
    )

    @staticmethod
    def associated_class() -> Type[Algorithm]:
        return Mambpo

    @classmethod
    def get_from_yaml(cls, path=None):
        config = super().get_from_yaml(path)
        if isinstance(config.world_model, dict):
            config.world_model = MambpoWorldModelConfig(**config.world_model)
        if isinstance(config.imagined_rollouts, dict):
            config.imagined_rollouts = MambpoImaginedRolloutsConfig(
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
        return False

    @staticmethod
    def has_centralized_critic() -> bool:
        return True
