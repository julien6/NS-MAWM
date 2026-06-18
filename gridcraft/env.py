import argparse
import os
import sys

import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT_DIR, "Gridcraft"))
sys.path.insert(1, ROOT_DIR)

from gridcraft import GridcraftConfig, GridcraftEnv
from gridcraft.constants import Block, Terrain
from gridcraft.render import PygameRenderer

from exp_config import (
  ACTION_SIZE,
  BLOCK_CLASSES,
  ENTITY_CLASSES,
  GRIDCRAFT_HEIGHT,
  GRIDCRAFT_MAX_STEPS,
  GRIDCRAFT_NUM_AGENTS,
  GRIDCRAFT_VIEW_SIZE,
  GRIDCRAFT_WIDTH,
  OBS_SIZE,
  SELF_FEATURES,
  TERRAIN_CLASSES,
)
from worldmodels_compat import SimpleImageViewer


def _one_hot(values, depth):
  values = np.asarray(values, dtype=np.int64).reshape(-1)
  values = np.clip(values, 0, depth - 1)
  return np.eye(depth, dtype=np.float32)[values].reshape(-1)


def obs_to_vector(obs):
  grid = obs["grid"]
  terrain = _one_hot(grid[0], TERRAIN_CLASSES)
  blocks = _one_hot(grid[1], BLOCK_CLASSES)
  entities = _one_hot(grid[2], ENTITY_CLASSES)
  self_vec = np.asarray(obs["self"], dtype=np.float32)
  # hp/hunger are bounded by 20 by default; inventory counts are clipped for scale.
  normalized = np.zeros((SELF_FEATURES,), dtype=np.float32)
  normalized[0:2] = self_vec[0:2] / 20.0
  normalized[2:] = np.clip(self_vec[2:], 0, 10) / 10.0
  vector = np.concatenate([terrain, blocks, entities, normalized]).astype(np.float32)
  assert vector.shape == (OBS_SIZE,)
  return vector


class GridcraftSingleAgentEnv:
  def __init__(self, seed=None, render_mode=None, max_steps=GRIDCRAFT_MAX_STEPS):
    self.agent_id = "agent_0"
    self.render_mode = render_mode
    self.config = GridcraftConfig(
      width=GRIDCRAFT_WIDTH,
      height=GRIDCRAFT_HEIGHT,
      num_agents=GRIDCRAFT_NUM_AGENTS,
      view_size=GRIDCRAFT_VIEW_SIZE,
      max_steps=max_steps,
      seed=seed,
      tile_size=24,
    )
    self.env = GridcraftEnv(config=self.config, render_mode=None)
    self.viewer = SimpleImageViewer() if render_mode == "human" else None
    self.tabular_renderer = PygameRenderer(self.config)
    self.action_size = ACTION_SIZE
    self.obs_size = OBS_SIZE

  def seed(self, seed=None):
    self.config.seed = seed
    return [seed]

  def reset(self, seed=None):
    obs, infos = self.env.reset(seed=seed)
    self.last_obs = obs[self.agent_id]
    return obs_to_vector(self.last_obs)

  def step(self, action):
    action = int(np.asarray(action).reshape(-1)[0])
    action = int(np.clip(action, 0, ACTION_SIZE - 1))
    obs, rewards, terminations, truncations, infos = self.env.step({self.agent_id: action})
    self.last_obs = obs[self.agent_id]
    done = bool(terminations[self.agent_id] or truncations[self.agent_id])
    info = infos.get(self.agent_id, {})
    return obs_to_vector(self.last_obs), float(rewards[self.agent_id]), done, info

  def render(self):
    frame = self.tabular_renderer.render(None, "rgb_array", tabular_observations={self.agent_id: self.last_obs})
    if self.viewer is not None:
      self.viewer.imshow(frame)
    return frame

  def close(self):
    if self.viewer is not None:
      self.viewer.close()
    self.env.close()

  def _render_rgb_array(self):
    tile = self.config.tile_size
    world = self.env.world
    h = self.config.height * tile
    w = self.config.width * tile
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    terrain_colors = {
      int(Terrain.GRASS): (74, 138, 65),
      int(Terrain.WATER): (49, 104, 165),
      int(Terrain.DIRT): (116, 96, 64),
    }
    block_colors = {
      int(Block.TREE): (105, 72, 39),
      int(Block.STONE): (78, 78, 78),
      int(Block.CRAFTING_TABLE): (160, 110, 55),
    }
    for y in range(self.config.height):
      for x in range(self.config.width):
        color = terrain_colors.get(int(world.terrain[y, x]), (40, 40, 40))
        frame[y * tile:(y + 1) * tile, x * tile:(x + 1) * tile] = color
        block = int(world.blocks[y, x])
        if block:
          margin = max(2, tile // 6)
          frame[y * tile + margin:(y + 1) * tile - margin, x * tile + margin:(x + 1) * tile - margin] = block_colors.get(block, (120, 90, 50))
    for item in getattr(world, "items", []):
      self._draw_square(frame, item.x, item.y, (235, 210, 80), tile)
    for mob in getattr(world, "mobs", []):
      self._draw_square(frame, mob.x, mob.y, (180, 50, 50), tile)
    for agent in world.agents.values():
      if agent.alive:
        self._draw_square(frame, agent.x, agent.y, (240, 240, 240), tile)
    return frame

  @staticmethod
  def _draw_square(frame, x, y, color, tile):
    margin = max(2, tile // 4)
    frame[y * tile + margin:(y + 1) * tile - margin, x * tile + margin:(x + 1) * tile - margin] = color


def make_env(seed=-1, render_mode=False, max_steps=GRIDCRAFT_MAX_STEPS):
  mode = "human" if render_mode else None
  env = GridcraftSingleAgentEnv(seed=seed if seed >= 0 else None, render_mode=mode, max_steps=max_steps)
  return env


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--steps", type=int, default=10)
  parser.add_argument("--seed", type=int, default=1)
  parser.add_argument("--render", action="store_true")
  args = parser.parse_args()

  env = make_env(seed=args.seed, render_mode=args.render, max_steps=args.steps)
  obs = env.reset(seed=args.seed)
  print("reset", obs.shape, obs.dtype)
  total = 0.0
  rng = np.random.default_rng(args.seed)
  for step in range(args.steps):
    action = int(rng.integers(0, ACTION_SIZE))
    obs, reward, done, info = env.step(action)
    total += reward
    if args.render:
      env.render()
    print("step", step, "action", action, "reward", reward, "done", done)
    if done:
      break
  print("total_reward", total)
  env.close()


if __name__ == "__main__":
  main()
