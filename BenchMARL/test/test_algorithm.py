#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

import pytest
import torch

from benchmarl.algorithms import algorithm_config_registry, MambpoConfig, MBMappoConfig
from benchmarl.algorithms.mambpo import EnsembleWorldModel
from benchmarl.algorithms.mb_mappo import _expand_env_branch_agent
from benchmarl.algorithms.common import AlgorithmConfig
from benchmarl.environments import VmasTask
from benchmarl.experiment import Experiment
from benchmarl.hydra_config import load_algorithm_config_from_hydra
from benchmarl.models import MlpConfig
from hydra import compose, initialize
from torch import nn
from utils import _has_vmas


@pytest.mark.parametrize("algo_name", algorithm_config_registry.keys())
def test_loading_algorithms(algo_name):
    with initialize(version_base=None, config_path="../benchmarl/conf"):
        cfg = compose(
            config_name="config",
            overrides=[
                f"algorithm={algo_name}",
                "task=vmas/balance",
            ],
        )
        algo_config: AlgorithmConfig = load_algorithm_config_from_hydra(cfg.algorithm)
        assert algo_config == algorithm_config_registry[algo_name].get_from_yaml()


def test_mb_mappo_defaults_actor_on_policy():
    algo_config = MBMappoConfig.get_from_yaml()
    assert algo_config.imagined_rollouts.use_for_actor is False


def test_mambpo_is_registered_off_policy():
    algo_config = MambpoConfig.get_from_yaml()
    assert algorithm_config_registry["mambpo"] is MambpoConfig
    assert algo_config.on_policy() is False
    assert algo_config.world_model.enabled is True
    assert algo_config.imagined_rollouts.real_ratio < 1.0


def test_mambpo_ensemble_world_model_forward():
    model = EnsembleWorldModel(
        input_dim=7,
        obs_dim=4,
        reward_dim=2,
        hidden_sizes=[8],
        n_models=3,
        n_elites=2,
        stochastic=True,
        predict_done=True,
    )
    obs = torch.zeros(5, 4)
    action = torch.zeros(5, 3)
    obs_mu, obs_log_var, reward_mu, reward_log_var, done_logit = model(obs, action)
    assert obs_mu.shape == (3, 5, 4)
    assert obs_log_var.shape == (3, 5, 4)
    assert reward_mu.shape == (3, 5, 2)
    assert reward_log_var.shape == (3, 5, 2)
    assert done_logit.shape == (3, 5, 2)


def test_mambpo_metric_tensors_are_float():
    metrics = {
        "mambpo/model_buffer_size": torch.tensor(4.0),
        "mambpo/model_rollout_length": torch.tensor(1.0),
        "mambpo/real_batch_size": torch.tensor(2.0),
        "mambpo/imagined_batch_size": torch.tensor(2.0),
    }
    for value in metrics.values():
        assert value.is_floating_point()


def test_mb_mappo_branch_expansion_preserves_env_branch_agent_order():
    tensor = torch.tensor([[0], [1], [10], [11]])
    env_batch, expanded = _expand_env_branch_agent(
        tensor, n_agents=2, num_branches=3
    )
    assert env_batch == 2
    assert expanded.squeeze(-1).tolist() == [
        0,
        1,
        0,
        1,
        0,
        1,
        10,
        11,
        10,
        11,
        10,
        11,
    ]


@pytest.mark.skipif(not _has_vmas, reason="VMAS not found")
def test_mb_mappo_tiny_training(experiment_config):
    experiment_config.max_n_iters = 1
    experiment_config.on_policy_collected_frames_per_batch = 20
    experiment_config.on_policy_minibatch_size = 10
    experiment_config.on_policy_n_envs_per_worker = 2
    experiment_config.evaluation = False
    experiment_config.create_json = False
    experiment_config.loggers = []
    algo_config = MBMappoConfig.get_from_yaml()
    algo_config.world_model.train_epochs = 1
    algo_config.world_model.batch_size = 8
    algo_config.world_model.hidden_sizes = [16]
    algo_config.imagined_rollouts.horizon = 1
    algo_config.imagined_rollouts.num_branches = 1
    model_config = MlpConfig(
        num_cells=[8], activation_class=nn.Tanh, layer_class=nn.Linear
    )
    experiment = Experiment(
        algorithm_config=algo_config,
        model_config=model_config,
        seed=0,
        config=experiment_config,
        task=VmasTask.BALANCE.get_from_yaml(),
    )
    experiment.run()


@pytest.mark.skipif(not _has_vmas, reason="VMAS not found")
def test_mambpo_tiny_training(experiment_config):
    experiment_config.max_n_iters = 1
    experiment_config.off_policy_collected_frames_per_batch = 20
    experiment_config.off_policy_train_batch_size = 8
    experiment_config.off_policy_n_optimizer_steps = 1
    experiment_config.off_policy_n_envs_per_worker = 2
    experiment_config.off_policy_memory_size = 100
    experiment_config.evaluation = False
    experiment_config.create_json = False
    experiment_config.loggers = []
    algo_config = MambpoConfig.get_from_yaml()
    algo_config.world_model.train_interval = 1
    algo_config.world_model.train_steps = 1
    algo_config.world_model.batch_size = 8
    algo_config.world_model.hidden_sizes = [16]
    algo_config.world_model.n_models = 2
    algo_config.world_model.n_elites = 2
    algo_config.imagined_rollouts.rollout_length = 1
    algo_config.imagined_rollouts.model_batch_size = 8
    algo_config.imagined_rollouts.model_buffer_size = 100
    model_config = MlpConfig(
        num_cells=[8], activation_class=nn.Tanh, layer_class=nn.Linear
    )
    experiment = Experiment(
        algorithm_config=algo_config,
        model_config=model_config,
        seed=0,
        config=experiment_config,
        task=VmasTask.BALANCE.get_from_yaml(),
    )
    experiment.run()
