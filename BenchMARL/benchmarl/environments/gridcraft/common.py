from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional

from torchrl.data import Composite
from torchrl.envs import EnvBase

from benchmarl.environments.common import Task, TaskClass
from benchmarl.utils import DEVICE_TYPING

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "vGridcraft"))

from vgridcraft import GridcraftDreamTorchRLEnv, VGridcraftConfig
from vgridcraft.torchrl_env import GridcraftTorchRLEnv


class GridcraftClass(TaskClass):
    def get_env_fun(
        self,
        num_envs: int,
        continuous_actions: bool,
        seed: Optional[int],
        device: DEVICE_TYPING,
    ) -> Callable[[], EnvBase]:
        config_dict = copy.deepcopy(self.config)
        env_kind = str(config_dict.pop("env_kind", "real"))
        checkpoint_dir = config_dict.pop("checkpoint_dir", None)
        ns_variant = config_dict.pop("ns_variant", "neural")
        ns_coverage = float(config_dict.pop("ns_coverage", 0.0))
        start_noise = float(config_dict.pop("start_noise", 1.0))
        config = VGridcraftConfig(**config_dict)
        if env_kind == "dream":
            return lambda: GridcraftDreamTorchRLEnv(
                num_envs=num_envs,
                device=device,
                seed=seed,
                config=config,
                checkpoint_dir=checkpoint_dir,
                ns_variant=ns_variant,
                ns_coverage=ns_coverage,
                start_noise=start_noise,
            )
        return lambda: GridcraftTorchRLEnv(
            num_envs=num_envs,
            device=device,
            seed=seed,
            config=config,
        )

    def supports_continuous_actions(self) -> bool:
        return False

    def supports_discrete_actions(self) -> bool:
        return True

    def max_steps(self, env: EnvBase) -> int:
        return self.config["max_steps"]

    def has_render(self, env: EnvBase) -> bool:
        return False

    def group_map(self, env: EnvBase) -> Dict[str, List[str]]:
        return env.group_map

    def observation_spec(self, env: EnvBase) -> Composite:
        return env.full_observation_spec_unbatched

    def info_spec(self, env: EnvBase):
        return None

    def state_spec(self, env: EnvBase):
        return None

    def action_spec(self, env: EnvBase) -> Composite:
        return env.full_action_spec_unbatched

    def action_mask_spec(self, env: EnvBase):
        return None

    @staticmethod
    def env_name() -> str:
        return "gridcraft"


class GridcraftTask(Task):
    SURVIVAL = None

    @staticmethod
    def associated_class():
        return GridcraftClass
