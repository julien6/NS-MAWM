from env import GridcraftSingleAgentEnv
from exp_config import GRIDCRAFT_MAX_STEPS


class GridcraftRenderEnv(GridcraftSingleAgentEnv):
  def __init__(self, seed=None, max_steps=GRIDCRAFT_MAX_STEPS):
    super().__init__(seed=seed, render_mode="human", max_steps=max_steps)


def make_env(seed=-1, render_mode=True, max_steps=GRIDCRAFT_MAX_STEPS):
  return GridcraftRenderEnv(seed=seed if seed >= 0 else None, max_steps=max_steps)
