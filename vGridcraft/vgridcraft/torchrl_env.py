from __future__ import annotations

import torch

from .config import VGridcraftConfig
from .env import ACTION_NAMES, EVENT_NAMES, REWARD_COMPONENT_NAMES, VectorizedGridcraftEnv

try:
    from tensordict import TensorDict
    from torchrl.data import Composite, Categorical, Unbounded
    from torchrl.envs import EnvBase
except Exception:  # pragma: no cover
    TensorDict = None
    Composite = None
    Categorical = None
    Unbounded = None
    EnvBase = object


class GridcraftTorchRLEnv(EnvBase):
    def __init__(
        self,
        num_envs: int,
        device: str | torch.device = "cpu",
        seed: int | None = None,
        config: VGridcraftConfig | None = None,
    ):
        if TensorDict is None:
            raise ImportError("torchrl and tensordict are required for GridcraftTorchRLEnv")
        self.config = config or VGridcraftConfig(seed=seed)
        self.group_map = {"agents": [f"agent_{i}" for i in range(self.config.num_agents)]}
        super().__init__(device=torch.device(device), batch_size=torch.Size([num_envs]))
        self.v_env = VectorizedGridcraftEnv(num_envs=num_envs, num_agents=self.config.num_agents, device=self.device, seed=seed, config=self.config)
        self._make_specs()

    def _make_specs(self) -> None:
        obs_shape = torch.Size([*self.batch_size, self.config.num_agents, self.config.obs_size])
        reward_shape = torch.Size([*self.batch_size, self.config.num_agents, 1])
        action_shape = torch.Size([*self.batch_size, self.config.num_agents])
        agent_shape = torch.Size([*self.batch_size, self.config.num_agents])
        self.observation_spec = Composite(
            agents=Composite(
                observation=Unbounded(shape=obs_shape, dtype=torch.float32, device=self.device),
                action_attempts=Unbounded(
                    shape=torch.Size([*agent_shape, len(ACTION_NAMES)]),
                    dtype=torch.float32,
                    device=self.device,
                ),
                event_success=Unbounded(
                    shape=torch.Size([*agent_shape, len(EVENT_NAMES)]),
                    dtype=torch.float32,
                    device=self.device,
                ),
                reward_components=Unbounded(
                    shape=torch.Size([*agent_shape, len(REWARD_COMPONENT_NAMES)]),
                    dtype=torch.float32,
                    device=self.device,
                ),
                task_level_max=Unbounded(
                    shape=torch.Size([*agent_shape, 1]), dtype=torch.float32, device=self.device
                ),
                complexity_cumulative=Unbounded(
                    shape=torch.Size([*agent_shape, 1]), dtype=torch.float32, device=self.device
                ),
                complexity_exponential_cumulative=Unbounded(
                    shape=torch.Size([*agent_shape, 1]), dtype=torch.float32, device=self.device
                ),
                complexity_unique=Unbounded(
                    shape=torch.Size([*agent_shape, 1]), dtype=torch.float32, device=self.device
                ),
                shape=agent_shape,
                device=self.device,
            ),
            shape=self.batch_size,
            device=self.device,
        )
        self.action_spec = Composite(
            agents=Composite(
                action=Categorical(n=self.config.action_size, shape=action_shape, dtype=torch.long, device=self.device),
                shape=agent_shape,
                device=self.device,
            ),
            shape=self.batch_size,
            device=self.device,
        )
        self.reward_spec = Composite(
            agents=Composite(
                reward=Unbounded(shape=reward_shape, dtype=torch.float32, device=self.device),
                shape=agent_shape,
                device=self.device,
            ),
            shape=self.batch_size,
            device=self.device,
        )
        self.done_spec = Composite(
            done=Unbounded(shape=torch.Size([*self.batch_size, 1]), dtype=torch.bool, device=self.device),
            terminated=Unbounded(shape=torch.Size([*self.batch_size, 1]), dtype=torch.bool, device=self.device),
            truncated=Unbounded(shape=torch.Size([*self.batch_size, 1]), dtype=torch.bool, device=self.device),
            shape=self.batch_size,
            device=self.device,
        )

    def _reset(self, tensordict=None):
        env_ids = None
        if tensordict is not None:
            reset_mask = None
            for key in ("_reset", "done", "terminated", "truncated"):
                try:
                    if key in tensordict.keys():
                        value = tensordict.get(key)
                        reset_mask = value if reset_mask is None else (reset_mask | value)
                except Exception:
                    continue
            if reset_mask is not None:
                reset_mask = reset_mask.reshape(*self.batch_size, -1).any(dim=-1)
                if not bool(reset_mask.any()):
                    obs = self.v_env.observation()
                    return self._make_reset_tensordict(obs)
                if bool(reset_mask.any()) and not bool(reset_mask.all()):
                    env_ids = torch.nonzero(reset_mask, as_tuple=False).flatten().to(self.device)
        obs = self.v_env.reset(env_ids=env_ids)
        return self._make_reset_tensordict(obs)

    def _make_reset_tensordict(self, obs):
        return TensorDict(
            {
                "agents": TensorDict(
                    {
                        "observation": obs["vector"].float(),
                        **self._empty_stats(),
                    },
                    batch_size=torch.Size([*self.batch_size, self.config.num_agents]),
                    device=self.device,
                ),
                "done": torch.zeros((*self.batch_size, 1), dtype=torch.bool, device=self.device),
                "terminated": torch.zeros((*self.batch_size, 1), dtype=torch.bool, device=self.device),
                "truncated": torch.zeros((*self.batch_size, 1), dtype=torch.bool, device=self.device),
            },
            batch_size=self.batch_size,
            device=self.device,
        )

    def _step(self, tensordict):
        actions = tensordict.get(("agents", "action")).long()
        if actions.ndim == 3 and actions.shape[-1] == 1:
            actions = actions.squeeze(-1)
        obs, reward, done, truncated, info = self.v_env.step(actions)
        return TensorDict(
            {
                "agents": TensorDict(
                    {
                        "observation": obs["vector"].float(),
                        "reward": reward.unsqueeze(-1),
                        "action_attempts": info["action_attempts"],
                        "event_success": info["event_success"],
                        "reward_components": info["reward_components"],
                        "task_level_max": info["task_level_max"].float().unsqueeze(-1),
                        "complexity_cumulative": info["complexity_cumulative"].unsqueeze(-1),
                        "complexity_exponential_cumulative": info[
                            "complexity_exponential_cumulative"
                        ].unsqueeze(-1),
                        "complexity_unique": info["complexity_unique"].unsqueeze(-1),
                    },
                    batch_size=torch.Size([*self.batch_size, self.config.num_agents]),
                    device=self.device,
                ),
                "done": (done | truncated).unsqueeze(-1),
                "terminated": done.unsqueeze(-1),
                "truncated": truncated.unsqueeze(-1),
            },
            batch_size=self.batch_size,
            device=self.device,
        )

    def _empty_stats(self):
        agent_shape = (*self.batch_size, self.config.num_agents)
        return {
            "action_attempts": torch.zeros(
                (*agent_shape, len(ACTION_NAMES)), dtype=torch.float32, device=self.device
            ),
            "event_success": torch.zeros(
                (*agent_shape, len(EVENT_NAMES)), dtype=torch.float32, device=self.device
            ),
            "reward_components": torch.zeros(
                (*agent_shape, len(REWARD_COMPONENT_NAMES)),
                dtype=torch.float32,
                device=self.device,
            ),
            "task_level_max": torch.zeros(
                (*agent_shape, 1), dtype=torch.float32, device=self.device
            ),
            "complexity_cumulative": torch.zeros(
                (*agent_shape, 1), dtype=torch.float32, device=self.device
            ),
            "complexity_exponential_cumulative": torch.zeros(
                (*agent_shape, 1), dtype=torch.float32, device=self.device
            ),
            "complexity_unique": torch.zeros(
                (*agent_shape, 1), dtype=torch.float32, device=self.device
            ),
        }

    def _set_seed(self, seed: int | None):
        if seed is not None:
            self.v_env.generator.manual_seed(int(seed))
        return seed
