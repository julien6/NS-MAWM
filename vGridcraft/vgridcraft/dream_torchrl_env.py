from __future__ import annotations

from pathlib import Path
import sys

import torch

from .config import VGridcraftConfig

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


class GridcraftDreamTorchRLEnv(EnvBase):
    """TorchRL environment backed by the trained Gridcraft VAE + MDN-RNN."""

    def __init__(
        self,
        num_envs: int,
        device: str | torch.device = "cpu",
        seed: int | None = None,
        config: VGridcraftConfig | None = None,
        checkpoint_dir: str | Path | None = None,
        ns_variant: str = "neural",
        ns_coverage: float = 0.0,
        start_noise: float = 1.0,
    ):
        if TensorDict is None:
            raise ImportError("torchrl and tensordict are required for GridcraftDreamTorchRLEnv")
        if checkpoint_dir is None:
            raise ValueError("GridcraftDreamTorchRLEnv requires checkpoint_dir with vae.pt and rnn.pt")
        self.config = config or VGridcraftConfig(seed=seed)
        self.group_map = {"agents": [f"agent_{i}" for i in range(self.config.num_agents)]}
        super().__init__(device=torch.device(device), batch_size=torch.Size([num_envs]))
        self.checkpoint_dir = Path(checkpoint_dir)
        self.ns_variant = str(ns_variant)
        self.ns_coverage = float(ns_coverage)
        self.start_noise = float(start_noise)
        self.generator = torch.Generator(device=self.device)
        if seed is not None:
            self.generator.manual_seed(int(seed))
        self._load_world_model()
        self._make_specs()
        self.z = torch.zeros((num_envs * self.config.num_agents, self.vae.z_size), device=self.device)
        self.rnn_state = None
        self.ns_memory = [None for _ in range(num_envs)]
        self.step_count = torch.zeros((num_envs,), dtype=torch.long, device=self.device)

    def _load_world_model(self) -> None:
        root = self.checkpoint_dir.resolve()
        for parent in [root, *root.parents]:
            if (parent / "gridcraft").exists():
                sys.path.insert(0, str(parent / "gridcraft"))
                break
        from torch_world_model import load_world_model_config, make_rnn_from_config, make_vae_from_config
        from run_benchmarl_dyna_gridcraft import apply_ns_mawm_to_latent_step

        vae_path = self.checkpoint_dir / "vae.pt"
        rnn_path = self.checkpoint_dir / "rnn.pt"
        if not vae_path.exists() or not rnn_path.exists():
            raise FileNotFoundError(f"Missing dream world model checkpoints: {vae_path} / {rnn_path}")
        wm_config = load_world_model_config(self.checkpoint_dir)
        self.vae = make_vae_from_config(wm_config).to(self.device)
        self.rnn = make_rnn_from_config(wm_config).to(self.device)
        self.vae.load_state_dict(torch.load(vae_path, map_location=self.device))
        self.rnn.load_state_dict(torch.load(rnn_path, map_location=self.device))
        self.vae.eval()
        self.rnn.eval()
        for param in self.vae.parameters():
            param.requires_grad_(False)
        for param in self.rnn.parameters():
            param.requires_grad_(False)
        self.apply_ns = apply_ns_mawm_to_latent_step

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
        reset_mask = torch.ones(self.batch_size, dtype=torch.bool, device=self.device)
        if tensordict is not None:
            for key in ("_reset", "done", "terminated", "truncated"):
                if key in tensordict.keys():
                    reset_mask = tensordict.get(key).reshape(*self.batch_size, -1).any(dim=-1)
                    break
        env_ids = torch.nonzero(reset_mask, as_tuple=False).flatten()
        if env_ids.numel() > 0:
            flat = torch.cat([
                torch.arange(int(env_id) * self.config.num_agents, (int(env_id) + 1) * self.config.num_agents, device=self.device)
                for env_id in env_ids
            ])
            self.z[flat] = torch.randn((flat.numel(), self.vae.z_size), generator=self.generator, device=self.device) * self.start_noise
            self.step_count[env_ids] = 0
            for env_id in env_ids.detach().cpu().tolist():
                self.ns_memory[int(env_id)] = None
            self.rnn_state = None
        return self._make_tensordict(
            done=torch.zeros((*self.batch_size, 1), dtype=torch.bool, device=self.device),
            include_reward=False,
        )

    def _step(self, tensordict):
        actions = tensordict.get(("agents", "action")).long()
        if actions.ndim == 3 and actions.shape[-1] == 1:
            actions = actions.squeeze(-1)
        flat_action = actions.reshape(-1)
        with torch.no_grad():
            predicted_z, reward, done_logit, self.rnn_state = self.rnn.step(self.z, flat_action, self.rnn_state, deterministic=True)
            self.z, self.ns_memory, _ = self.apply_ns(
                vae=self.vae,
                current_z=self.z,
                predicted_z=predicted_z,
                action=flat_action,
                ns_memory=self.ns_memory,
                ns_variant=self.ns_variant,
                ns_coverage=self.ns_coverage,
                num_agents=self.config.num_agents,
                device=self.device,
            )
        self.step_count += 1
        agent_done = torch.sigmoid(done_logit).reshape(*self.batch_size, self.config.num_agents) > 0.5
        done = agent_done.all(dim=1) | (self.step_count >= self.config.max_steps)
        return self._make_tensordict(
            reward=reward.reshape(*self.batch_size, self.config.num_agents, 1),
            done=done.unsqueeze(-1),
        )

    def _make_tensordict(self, reward=None, done=None, include_reward=True):
        obs = self.vae.decode(self.z).reshape(*self.batch_size, self.config.num_agents, self.config.obs_size)
        if done is None:
            done = torch.zeros((*self.batch_size, 1), dtype=torch.bool, device=self.device)
        agent_payload = {"observation": obs.float()}
        if include_reward:
            if reward is None:
                reward = torch.zeros((*self.batch_size, self.config.num_agents, 1), dtype=torch.float32, device=self.device)
            agent_payload["reward"] = reward.float()
        return TensorDict(
            {
                "agents": TensorDict(
                    agent_payload,
                    batch_size=torch.Size([*self.batch_size, self.config.num_agents]),
                    device=self.device,
                ),
                "done": done.bool(),
                "terminated": done.bool(),
                "truncated": (self.step_count >= self.config.max_steps).unsqueeze(-1),
            },
            batch_size=self.batch_size,
            device=self.device,
        )

    def _set_seed(self, seed: int | None):
        if seed is not None:
            self.generator.manual_seed(int(seed))
        return seed
