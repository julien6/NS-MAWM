from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Callable

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))

from vGridcraft.vgridcraft import VGridcraftConfig, VectorizedGridcraftEnv
from vGridcraft.vgridcraft.env import (
  BLOCK_EMPTY,
  BLOCK_STONE,
  BLOCK_TREE,
  ENTITY_AGENT,
  ENTITY_ITEM,
  ENTITY_MOB,
  ENTITY_NONE,
  ITEM_APPLE,
  ITEM_PLANK,
  ITEM_STICK,
  ITEM_STONE,
  ITEM_STONE_PICKAXE,
  ITEM_WOOD,
  ITEM_WOOD_PICKAXE,
  ITEM_WOOD_SWORD,
  TERRAIN_DIRT,
  TERRAIN_GRASS,
  TERRAIN_WATER,
)

from ns_symbolic import (
  ACTION_ATTACK,
  ACTION_EAT,
  ACTION_HARVEST,
  ACTION_MOVE_E,
  ACTION_MOVE_N,
  ACTION_MOVE_S,
  ACTION_MOVE_W,
  ACTION_PICKUP,
  ACTION_STAY,
  symbolic_joint_transition,
)
from wandb_schema import PSTR_RULES


ACTION_NAMES = {
  ACTION_STAY: "stay",
  ACTION_MOVE_N: "move_n",
  ACTION_MOVE_S: "move_s",
  ACTION_MOVE_W: "move_w",
  ACTION_MOVE_E: "move_e",
  ACTION_HARVEST: "harvest",
  ACTION_PICKUP: "pickup",
  ACTION_ATTACK: "attack",
  ACTION_EAT: "eat",
  9: "craft_plank",
  10: "craft_stick",
  11: "craft_wood_sword",
  12: "craft_stone_sword",
  13: "craft_wood_pickaxe",
  14: "craft_stone_pickaxe",
}

ITEM_NAMES = {
  ITEM_WOOD: "wood",
  ITEM_PLANK: "plank",
  ITEM_STICK: "stick",
  ITEM_STONE: "stone",
  ITEM_WOOD_SWORD: "wood_sword",
  ITEM_WOOD_PICKAXE: "wood_pickaxe",
  ITEM_STONE_PICKAXE: "stone_pickaxe",
  ITEM_APPLE: "apple",
}


@dataclass(frozen=True)
class PSTRScenarioSpec:
  rule_id: str
  action: dict[str, int]
  setup: Callable[[VectorizedGridcraftEnv], dict | None]
  description: str
  action_sequence: Callable[[int], list[dict[str, int]]] | None = None
  expected_rules: tuple[str, ...] | None = None


@dataclass
class PSTRScenarioResult:
  rule_id: str
  action_names: dict[str, str]
  description: str
  input_observations: dict[str, dict]
  real_next_observations: dict[str, dict]
  symbolic_next_observations: dict[str, dict]
  symbolic_masks: dict[str, dict]
  report: dict
  memory: dict
  frame: np.ndarray
  frames: list[np.ndarray]
  plain_frames: list[np.ndarray]
  actions_by_step: list[dict[str, str]]
  rewards_by_step: list[float]
  done_by_step: list[bool]
  reports_by_step: list[dict]
  masks_by_step: list[dict[str, dict]]
  strict_masks_by_step: list[dict[str, dict]]
  input_ascii: str
  output_ascii: str
  output_ascii_by_step: list[str]


def scenario_ids() -> list[str]:
  return ["PSTR_INDIV_LOCAL_GRID_TRANSLATION"] + [rule["id"] for rule in PSTR_RULES]


def scenario_specs() -> dict[str, PSTRScenarioSpec]:
  specs = {
    "PSTR_INDIV_LOCAL_GRID_TRANSLATION": PSTRScenarioSpec(
      "PSTR_INDIV_LOCAL_GRID_TRANSLATION",
      {"agent_0": ACTION_MOVE_E},
      _setup_shift,
      "Pedagogical alias showing the next local sub-grid prediction after movement: known static terrain/block cells translate opposite to the motion, while newly exposed cells are unknown.",
      action_sequence=lambda steps: [{"agent_0": action} for action in _movement_demo_actions(steps)],
      expected_rules=("PSTR_INDIV_STATIC_TERRAIN_SHIFT", "PSTR_INDIV_STATIC_BLOCK_SHIFT"),
    ),
    "PSTR_INDIV_STATIC_TERRAIN_SHIFT": PSTRScenarioSpec(
      "PSTR_INDIV_STATIC_TERRAIN_SHIFT",
      {"agent_0": ACTION_MOVE_E},
      _setup_shift,
      "Moving east shifts known static terrain in the egocentric observation; new border cells remain unknown.",
      action_sequence=lambda steps: [{"agent_0": action} for action in _movement_demo_actions(steps)],
    ),
    "PSTR_INDIV_STATIC_BLOCK_SHIFT": PSTRScenarioSpec(
      "PSTR_INDIV_STATIC_BLOCK_SHIFT",
      {"agent_0": ACTION_MOVE_E},
      _setup_shift,
      "Moving east shifts known static blocks in the egocentric observation; new border cells remain unknown.",
      action_sequence=lambda steps: [{"agent_0": action} for action in _movement_demo_actions(steps)],
    ),
    "PSTR_INDIV_CENTER_AGENT": PSTRScenarioSpec(
      "PSTR_INDIV_CENTER_AGENT",
      {"agent_0": ACTION_STAY},
      _setup_blank,
      "The controlled agent is always deterministically located at the center of its own observation.",
    ),
    "PSTR_INDIV_BLOCKED_WATER": PSTRScenarioSpec(
      "PSTR_INDIV_BLOCKED_WATER",
      {"agent_0": ACTION_MOVE_E},
      _setup_blocked_water,
      "A movement action into an observable water cell is blocked.",
    ),
    "PSTR_INDIV_BLOCKED_TREE_STONE": PSTRScenarioSpec(
      "PSTR_INDIV_BLOCKED_TREE_STONE",
      {"agent_0": ACTION_MOVE_E},
      _setup_blocked_tree,
      "A movement action into an observable tree or stone block is blocked.",
    ),
    "PSTR_INDIV_BLOCKED_ENTITY": PSTRScenarioSpec(
      "PSTR_INDIV_BLOCKED_ENTITY",
      {"agent_0": ACTION_MOVE_E},
      _setup_blocked_entity,
      "A movement action into an observable mob or agent entity is blocked.",
    ),
    "PSTR_INDIV_HARVEST_TREE_WOOD": PSTRScenarioSpec(
      "PSTR_INDIV_HARVEST_TREE_WOOD",
      {"agent_0": ACTION_HARVEST},
      _setup_harvest_tree,
      "Harvesting an adjacent tree removes the tree block and increases wood by one; apple drops are unknown.",
    ),
    "PSTR_INDIV_HARVEST_STONE_PICKAXE": PSTRScenarioSpec(
      "PSTR_INDIV_HARVEST_STONE_PICKAXE",
      {"agent_0": ACTION_HARVEST},
      _setup_harvest_stone,
      "Harvesting adjacent stone with a pickaxe removes the stone block and increases stone by one.",
    ),
    "PSTR_INDIV_PICKUP_ITEM": PSTRScenarioSpec(
      "PSTR_INDIV_PICKUP_ITEM",
      {"agent_0": ACTION_PICKUP},
      _setup_pickup_item,
      "Picking up a known adjacent item removes the item entity and increases the corresponding inventory count.",
    ),
    "PSTR_INDIV_EAT_APPLE": PSTRScenarioSpec(
      "PSTR_INDIV_EAT_APPLE",
      {"agent_0": ACTION_EAT},
      _setup_eat_apple,
      "Eating with apple inventory and non-max hunger consumes one apple and increases hunger.",
    ),
    "PSTR_INDIV_CRAFT_PLANK": PSTRScenarioSpec(
      "PSTR_INDIV_CRAFT_PLANK",
      {"agent_0": 9},
      _setup_craft_plank,
      "Crafting planks consumes one wood and produces two planks.",
    ),
    "PSTR_INDIV_CRAFT_STICK": PSTRScenarioSpec(
      "PSTR_INDIV_CRAFT_STICK",
      {"agent_0": 10},
      _setup_craft_stick,
      "Crafting sticks consumes two planks and produces four sticks.",
    ),
    "PSTR_INDIV_CRAFT_TOOLS": PSTRScenarioSpec(
      "PSTR_INDIV_CRAFT_TOOLS",
      {"agent_0": 11},
      _setup_craft_tool,
      "Crafting a tool consumes the required resources and adds the crafted tool to the inventory.",
    ),
    "PSTR_INDIV_ATTACK_MOB_LOCAL": PSTRScenarioSpec(
      "PSTR_INDIV_ATTACK_MOB_LOCAL",
      {"agent_0": ACTION_ATTACK},
      _setup_attack_mob,
      "Attacking an adjacent known mob updates symbolic mob hp memory; drops and mob movement remain unknown.",
    ),
    "PSTR_INDIV_HUNGER_COST_KNOWN_COUNTER": PSTRScenarioSpec(
      "PSTR_INDIV_HUNGER_COST_KNOWN_COUNTER",
      {"agent_0": ACTION_MOVE_E},
      _setup_hunger_counter,
      "When symbolic hunger counters are known, the rule predicts hunger decrease at the configured interval.",
    ),
    "PSTR_JOINT_RELATIVE_AGENT_ALIGNMENT": PSTRScenarioSpec(
      "PSTR_JOINT_RELATIVE_AGENT_ALIGNMENT",
      {"agent_0": ACTION_STAY, "agent_1": ACTION_STAY},
      _setup_two_agents_adjacent,
      "Mutual observations of agents align their relative positions in symbolic memory.",
    ),
    "PSTR_JOINT_MAP_FUSION": PSTRScenarioSpec(
      "PSTR_JOINT_MAP_FUSION",
      {"agent_0": ACTION_STAY, "agent_1": ACTION_STAY},
      _setup_two_agents_map_fusion,
      "Known relative positions allow local observations to be fused into a shared reconstructed map.",
    ),
    "PSTR_JOINT_GLOBAL_STATIC_QUERY": PSTRScenarioSpec(
      "PSTR_JOINT_GLOBAL_STATIC_QUERY",
      {"agent_0": ACTION_MOVE_E},
      _setup_global_static_query,
      "A reconstructed map can fill terrain and block cells in a future local observation.",
    ),
    "PSTR_JOINT_MULTI_AGENT_COLLISION": PSTRScenarioSpec(
      "PSTR_JOINT_MULTI_AGENT_COLLISION",
      {"agent_0": ACTION_MOVE_E, "agent_1": ACTION_MOVE_W},
      _setup_collision,
      "Two known agents attempting to enter the same cell are conservatively blocked.",
    ),
    "PSTR_JOINT_AGENT_ENTITY_PREDICTION": PSTRScenarioSpec(
      "PSTR_JOINT_AGENT_ENTITY_PREDICTION",
      {"agent_0": ACTION_STAY, "agent_1": ACTION_STAY},
      _setup_two_agents_adjacent,
      "Known relative positions let the symbolic layer insert other agents into future local entity planes.",
    ),
    "PSTR_JOINT_SHARED_WORLD_UPDATE": PSTRScenarioSpec(
      "PSTR_JOINT_SHARED_WORLD_UPDATE",
      {"agent_0": ACTION_HARVEST, "agent_1": ACTION_STAY},
      _setup_joint_harvest_tree,
      "A deterministic harvest updates the shared reconstructed block map for all agents.",
    ),
    "PSTR_JOINT_SHARED_ITEM_UPDATE": PSTRScenarioSpec(
      "PSTR_JOINT_SHARED_ITEM_UPDATE",
      {"agent_0": ACTION_PICKUP, "agent_1": ACTION_STAY},
      _setup_joint_pickup_item,
      "A deterministic pickup removes the known adjacent item from shared symbolic memory.",
    ),
  }
  ordered = {}
  for rule_id in scenario_ids():
    if rule_id not in specs:
      raise KeyError(f"missing PSTR scenario for {rule_id}")
    ordered[rule_id] = specs[rule_id]
  return ordered


def run_scenario(rule_id: str, render_mode: str = "rgb_array", steps: int = 6) -> PSTRScenarioResult:
  specs = scenario_specs()
  if rule_id not in specs:
    raise KeyError(f"unknown PSTR scenario {rule_id}")
  spec = specs[rule_id]
  steps = max(1, int(steps))
  action_sequence = _scenario_actions(spec, steps)
  agent_ids = set(spec.action)
  for action in action_sequence:
    agent_ids.update(action)
  num_agents = max(_agent_index(agent_id) for agent_id in agent_ids) + 1
  env = _new_env(num_agents=num_agents)
  memory = spec.setup(env) or {}
  input_obs = _observations(env)
  symbolic = _copy_joint_obs(input_obs)
  masks = _full_masks(symbolic)
  frames = [_render_prediction_frame(
    env,
    symbolic,
    masks,
    render_mode,
    overlay_info={"step": 0, "action": "initial", "reward": 0.0, "done": False},
  )]
  plain_frames = [_render_prediction_frame(env, symbolic, masks, render_mode)]
  rewards_by_step = [0.0]
  done_by_step = [False]
  reports_by_step = [{"rules": {}}]
  masks_by_step = [masks]
  strict_masks_by_step = [masks]
  output_ascii_by_step = [_joint_ascii(symbolic, masks)]
  real_next = input_obs
  for step_index, action in enumerate(action_sequence, start=1):
    symbolic, strict_masks, memory, report = symbolic_joint_transition(symbolic, action, memory=memory, coverage=1.0)
    masks = _display_masks(symbolic, strict_masks, action)
    reward, done = _step_env(env, action, num_agents)
    real_next = _observations(env)
    frames.append(_render_prediction_frame(
      env,
      symbolic,
      masks,
      render_mode,
      overlay_info={"step": step_index, "action": _action_names(action), "reward": reward, "done": done},
    ))
    plain_frames.append(_render_prediction_frame(env, symbolic, masks, render_mode))
    rewards_by_step.append(reward)
    done_by_step.append(done)
    reports_by_step.append(report)
    masks_by_step.append(masks)
    strict_masks_by_step.append(strict_masks)
    output_ascii_by_step.append(_joint_ascii(symbolic, masks))
  env.close()
  report = reports_by_step[-1]
  frame = frames[-1]
  return PSTRScenarioResult(
    rule_id=rule_id,
    action_names={agent_id: ACTION_NAMES.get(int(action), str(action)) for agent_id, action in spec.action.items()},
    description=spec.description,
    input_observations=input_obs,
    real_next_observations=real_next,
    symbolic_next_observations=symbolic,
    symbolic_masks=masks,
    report=report,
    memory=memory,
    frame=frame,
    frames=frames,
    plain_frames=plain_frames,
    actions_by_step=[_action_names(action) for action in action_sequence],
    rewards_by_step=rewards_by_step,
    done_by_step=done_by_step,
    reports_by_step=reports_by_step,
    masks_by_step=masks_by_step,
    strict_masks_by_step=strict_masks_by_step,
    input_ascii=_joint_ascii(input_obs),
    output_ascii=_joint_ascii(symbolic, masks),
    output_ascii_by_step=output_ascii_by_step,
  )


def expected_rules(rule_id: str) -> tuple[str, ...]:
  spec = scenario_specs()[rule_id]
  return spec.expected_rules or (rule_id,)


def _scenario_actions(spec: PSTRScenarioSpec, steps: int) -> list[dict[str, int]]:
  if steps <= 1:
    return []
  if spec.action_sequence is not None:
    actions = spec.action_sequence(steps - 1)
  else:
    actions = [spec.action] + [_stay_like(spec.action) for _ in range(max(0, steps - 2))]
  if len(actions) < steps - 1:
    actions = actions + [_stay_like(spec.action) for _ in range(steps - 1 - len(actions))]
  return actions[:steps - 1]


def _movement_demo_actions(count: int) -> list[int]:
  pattern = [ACTION_MOVE_E, ACTION_MOVE_E, ACTION_MOVE_S, ACTION_MOVE_W, ACTION_MOVE_N]
  return [pattern[idx % len(pattern)] for idx in range(max(0, count))]


def _stay_like(action: dict[str, int]) -> dict[str, int]:
  return {agent_id: ACTION_STAY for agent_id in action}


def _action_names(action: dict[str, int]) -> dict[str, str]:
  return {agent_id: ACTION_NAMES.get(int(action_id), str(action_id)) for agent_id, action_id in action.items()}


def _copy_joint_obs(joint_obs: dict[str, dict]) -> dict[str, dict]:
  return {
    agent_id: {
      "grid": np.asarray(obs["grid"], dtype=np.int8).copy(),
      "self": np.asarray(obs["self"], dtype=np.int16).copy(),
    }
    for agent_id, obs in joint_obs.items()
  }


def _full_masks(joint_obs: dict[str, dict]) -> dict[str, dict]:
  return {
    agent_id: {
      "grid": np.ones_like(obs["grid"], dtype=np.bool_),
      "self": np.ones_like(obs["self"], dtype=np.bool_),
    }
    for agent_id, obs in joint_obs.items()
  }


def _display_masks(symbolic: dict[str, dict], strict_masks: dict[str, dict], action: dict[str, int]) -> dict[str, dict]:
  display = _full_masks(symbolic)
  for agent_id, action_id in action.items():
    if int(action_id) in (ACTION_MOVE_N, ACTION_MOVE_S, ACTION_MOVE_W, ACTION_MOVE_E) and agent_id in strict_masks:
      display[agent_id]["grid"] = np.asarray(strict_masks[agent_id]["grid"], dtype=np.bool_).copy()
      display[agent_id]["self"] = np.ones_like(symbolic[agent_id]["self"], dtype=np.bool_)
  return display


def _render_prediction_frame(
  env: VectorizedGridcraftEnv,
  symbolic: dict[str, dict],
  masks: dict[str, dict],
  render_mode: str,
  overlay_info: object | None = None,
) -> np.ndarray:
  predicted = {
    agent_id: {
      "grid": symbolic[agent_id]["grid"],
      "self": symbolic[agent_id]["self"],
      "mask": masks[agent_id],
    }
    for agent_id in symbolic
  }
  frame = env.render(env_index=0, mode=render_mode, tabular_observations=predicted, overlay_info=overlay_info)
  if frame is None:
    frame = env.render(env_index=0, mode="rgb_array", tabular_observations=predicted, overlay_info=overlay_info)
  return frame


def _step_env(env: VectorizedGridcraftEnv, action: dict[str, int], num_agents: int) -> tuple[float, bool]:
  actions = torch.zeros((1, num_agents), dtype=torch.long, device=env.device)
  for agent_id, action_id in action.items():
    actions[0, _agent_index(agent_id)] = int(action_id)
  _, rewards, done, truncated, _ = env.step(actions)
  return float(rewards[0].sum().detach().cpu()), bool((done | truncated)[0].detach().cpu())


def _new_env(num_agents: int) -> VectorizedGridcraftEnv:
  cfg = VGridcraftConfig(
    width=11,
    height=11,
    num_agents=num_agents,
    view_size=7,
    max_steps=20,
    seed=7,
    tree_density=0.0,
    stone_density=0.0,
    water_density=0.0,
    mob_spawn_rate=0,
    max_mobs=4,
    mob_move_prob=0.0,
    mob_damage=0,
    item_drop_chance=0.0,
    tree_apple_drop_chance=0.0,
    health_regen_ticks=0,
    hunger_decay_ticks=0,
    tile_size=48,
    fps=8,
  )
  env = VectorizedGridcraftEnv(num_envs=1, num_agents=num_agents, device="cpu", seed=7, config=cfg)
  _clear(env)
  return env


def _clear(env: VectorizedGridcraftEnv) -> None:
  env.terrain.fill_(TERRAIN_GRASS)
  env.blocks.fill_(BLOCK_EMPTY)
  env.agent_x.fill_(5)
  env.agent_y.fill_(5)
  env.hp.fill_(20)
  env.hunger.fill_(20)
  env.inventory.zero_()
  env.equipped.fill_(-1)
  env.alive.fill_(True)
  env.visited.zero_()
  env.move_counter.zero_()
  env.harvest_counter.zero_()
  env.attack_counter.zero_()
  env.mob_alive.fill_(False)
  env.item_alive.fill_(False)
  env.step_count.zero_()
  for agent_idx in range(env.num_agents):
    env.agent_x[0, agent_idx] = 5 + agent_idx
    env.agent_y[0, agent_idx] = 5


def _observations(env: VectorizedGridcraftEnv) -> dict[str, dict]:
  obs = env.observation()
  grid = obs["grid"][0].detach().cpu().numpy().astype("int8")
  self_vec = obs["self"][0].detach().cpu().numpy().astype("int16")
  return {
    f"agent_{agent_idx}": {"grid": grid[agent_idx], "self": self_vec[agent_idx]}
    for agent_idx in range(env.num_agents)
  }


def _agent_index(agent_id: str) -> int:
  return int(str(agent_id).split("_")[-1])


def _setup_blank(env: VectorizedGridcraftEnv) -> dict:
  _clear(env)
  return {"agent_pos": {"agent_0": (0, 0)}}


def _setup_shift(env: VectorizedGridcraftEnv) -> dict:
  _clear(env)
  env.terrain[0, 3:7, 2] = TERRAIN_WATER
  env.terrain[0, 2, 7] = TERRAIN_DIRT
  env.blocks[0, 4, 3] = BLOCK_TREE
  env.blocks[0, 6, 7] = BLOCK_STONE
  return {"agent_pos": {"agent_0": (0, 0)}}


def _setup_blocked_water(env: VectorizedGridcraftEnv) -> dict:
  _clear(env)
  env.terrain[0, 5, 6] = TERRAIN_WATER
  return {"agent_pos": {"agent_0": (0, 0)}}


def _setup_blocked_tree(env: VectorizedGridcraftEnv) -> dict:
  _clear(env)
  env.blocks[0, 5, 6] = BLOCK_TREE
  return {"agent_pos": {"agent_0": (0, 0)}}


def _setup_blocked_entity(env: VectorizedGridcraftEnv) -> dict:
  _clear(env)
  env.mob_alive[0, 0] = True
  env.mob_x[0, 0] = 6
  env.mob_y[0, 0] = 5
  env.mob_hp[0, 0] = 10
  return {"agent_pos": {"agent_0": (0, 0)}, "mobs": {(1, 0): {"hp": 10}}}


def _setup_harvest_tree(env: VectorizedGridcraftEnv) -> dict:
  _clear(env)
  env.blocks[0, 5, 6] = BLOCK_TREE
  return {"agent_pos": {"agent_0": (0, 0)}, "blocks": {(1, 0): BLOCK_TREE}}


def _setup_harvest_stone(env: VectorizedGridcraftEnv) -> dict:
  _clear(env)
  env.blocks[0, 5, 6] = BLOCK_STONE
  env.inventory[0, 0, ITEM_WOOD_PICKAXE] = 1
  env.equipped[0, 0] = ITEM_WOOD_PICKAXE
  return {"agent_pos": {"agent_0": (0, 0)}, "blocks": {(1, 0): BLOCK_STONE}}


def _setup_pickup_item(env: VectorizedGridcraftEnv) -> dict:
  _clear(env)
  env.item_alive[0, 0] = True
  env.item_x[0, 0] = 6
  env.item_y[0, 0] = 5
  env.item_type[0, 0] = ITEM_WOOD
  env.item_count[0, 0] = 2
  return {"agent_pos": {"agent_0": (0, 0)}, "items": {(1, 0): (ITEM_WOOD, 2)}}


def _setup_eat_apple(env: VectorizedGridcraftEnv) -> dict:
  _clear(env)
  env.hunger[0, 0] = 12
  env.inventory[0, 0, ITEM_APPLE] = 2
  return {"agent_pos": {"agent_0": (0, 0)}}


def _setup_craft_plank(env: VectorizedGridcraftEnv) -> dict:
  _clear(env)
  env.inventory[0, 0, ITEM_WOOD] = 2
  return {"agent_pos": {"agent_0": (0, 0)}}


def _setup_craft_stick(env: VectorizedGridcraftEnv) -> dict:
  _clear(env)
  env.inventory[0, 0, ITEM_PLANK] = 3
  return {"agent_pos": {"agent_0": (0, 0)}}


def _setup_craft_tool(env: VectorizedGridcraftEnv) -> dict:
  _clear(env)
  env.inventory[0, 0, ITEM_PLANK] = 1
  env.inventory[0, 0, ITEM_STICK] = 1
  return {"agent_pos": {"agent_0": (0, 0)}}


def _setup_attack_mob(env: VectorizedGridcraftEnv) -> dict:
  _clear(env)
  env.mob_alive[0, 0] = True
  env.mob_x[0, 0] = 6
  env.mob_y[0, 0] = 5
  env.mob_hp[0, 0] = 10
  return {"agent_pos": {"agent_0": (0, 0)}, "mobs": {(1, 0): {"hp": 10}}}


def _setup_hunger_counter(env: VectorizedGridcraftEnv) -> dict:
  _clear(env)
  env.move_counter[0, 0] = 4
  return {"agent_pos": {"agent_0": (0, 0)}, "hunger_counters": {"agent_0": {"move": 4}}}


def _setup_two_agents_adjacent(env: VectorizedGridcraftEnv) -> dict:
  _clear(env)
  env.agent_x[0, 0] = 5
  env.agent_y[0, 0] = 5
  env.agent_x[0, 1] = 6
  env.agent_y[0, 1] = 5
  return {}


def _setup_two_agents_map_fusion(env: VectorizedGridcraftEnv) -> dict:
  _setup_two_agents_adjacent(env)
  env.blocks[0, 4, 4] = BLOCK_TREE
  env.terrain[0, 6, 7] = TERRAIN_WATER
  return {"agent_pos": {"agent_0": (0, 0), "agent_1": (1, 0)}}


def _setup_global_static_query(env: VectorizedGridcraftEnv) -> dict:
  _clear(env)
  env.terrain[0, 4, 8] = TERRAIN_WATER
  env.blocks[0, 6, 8] = BLOCK_TREE
  memory = {"agent_pos": {"agent_0": (0, 0)}, "terrain": {}, "blocks": {}}
  for gy in range(-3, 4):
    for gx in range(-3, 5):
      memory["terrain"][(gx, gy)] = TERRAIN_GRASS
      memory["blocks"][(gx, gy)] = BLOCK_EMPTY
  memory["terrain"][(3, -1)] = TERRAIN_WATER
  memory["blocks"][(3, 1)] = BLOCK_TREE
  return memory


def _setup_collision(env: VectorizedGridcraftEnv) -> dict:
  _clear(env)
  env.agent_x[0, 0] = 4
  env.agent_y[0, 0] = 5
  env.agent_x[0, 1] = 6
  env.agent_y[0, 1] = 5
  return {"agent_pos": {"agent_0": (-1, 0), "agent_1": (1, 0)}}


def _setup_joint_harvest_tree(env: VectorizedGridcraftEnv) -> dict:
  _setup_two_agents_adjacent(env)
  env.blocks[0, 5, 4] = BLOCK_TREE
  return {"agent_pos": {"agent_0": (0, 0), "agent_1": (1, 0)}, "blocks": {(-1, 0): BLOCK_TREE}}


def _setup_joint_pickup_item(env: VectorizedGridcraftEnv) -> dict:
  _setup_two_agents_adjacent(env)
  env.item_alive[0, 0] = True
  env.item_x[0, 0] = 6
  env.item_y[0, 0] = 5
  env.item_type[0, 0] = ITEM_WOOD
  env.item_count[0, 0] = 1
  return {
    "agent_pos": {"agent_0": (0, 0), "agent_1": (1, 0)},
    "items": {(1, 0): (ITEM_WOOD, 1)},
    "entities": {(1, 0): ENTITY_ITEM},
  }


def _joint_ascii(observations: dict[str, dict], masks: dict[str, dict] | None = None) -> str:
  chunks = []
  for agent_id in sorted(observations):
    obs = observations[agent_id]
    mask = masks.get(agent_id) if masks else None
    chunks.append(f"{agent_id}:")
    chunks.append(_grid_ascii(obs["grid"], mask.get("grid") if mask else None))
    chunks.append(_self_ascii(obs["self"], mask.get("self") if mask else None))
  return "\n".join(chunks)


def _grid_ascii(grid: np.ndarray, mask: np.ndarray | None = None) -> str:
  rows = []
  for y in range(grid.shape[1]):
    cells = []
    for x in range(grid.shape[2]):
      if mask is not None and not bool(np.any(mask[:, y, x])):
        cells.append("?")
      else:
        cells.append(_cell_char(grid[:, y, x]))
    rows.append(" ".join(cells))
  return "\n".join(rows)


def _cell_char(cell: np.ndarray) -> str:
  terrain, block, entity = [int(value) for value in cell]
  if entity == ENTITY_AGENT:
    return "A"
  if entity == ENTITY_MOB:
    return "M"
  if entity == ENTITY_ITEM:
    return "I"
  if block == BLOCK_TREE:
    return "T"
  if block == BLOCK_STONE:
    return "S"
  if terrain == TERRAIN_WATER:
    return "W"
  if terrain == TERRAIN_DIRT:
    return "D"
  if terrain == TERRAIN_GRASS:
    return "."
  return "?"


def _self_ascii(self_vec: np.ndarray, mask: np.ndarray | None = None) -> str:
  parts = []
  labels = ["hp", "hunger"] + [ITEM_NAMES.get(idx, f"item_{idx}") for idx in range(len(self_vec) - 2)]
  for idx, label in enumerate(labels[: len(self_vec)]):
    value = "?" if mask is not None and idx < len(mask) and not bool(mask[idx]) else str(int(self_vec[idx]))
    if value != "0" or label in ("hp", "hunger") or value == "?":
      parts.append(f"{label}={value}")
  return "self: " + ", ".join(parts)
