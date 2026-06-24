from __future__ import annotations

import torch

from .config import VGridcraftConfig
from .env import VectorizedGridcraftEnv

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
        obs = self.v_env.reset()
        return TensorDict(
            {
                "agents": TensorDict(
                    {"observation": obs["vector"].float()},
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
        obs, reward, done, truncated, _ = self.v_env.step(actions)
        return TensorDict(
            {
                "agents": TensorDict(
                    {
                        "observation": obs["vector"].float(),
                        "reward": reward.unsqueeze(-1),
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

    def _set_seed(self, seed: int | None):
        if seed is not None:
            self.v_env.generator.manual_seed(int(seed))
        return seed
