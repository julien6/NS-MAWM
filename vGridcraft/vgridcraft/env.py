from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import sys
from typing import Any

import torch
import torch.nn.functional as F

from .config import VGridcraftConfig


TERRAIN_GRASS = 0
TERRAIN_WATER = 1
TERRAIN_DIRT = 2

BLOCK_EMPTY = 0
BLOCK_TREE = 1
BLOCK_STONE = 2

ENTITY_NONE = 0
ENTITY_AGENT = 1
ENTITY_MOB = 2
ENTITY_ITEM = 3

ITEM_WOOD = 0
ITEM_PLANK = 1
ITEM_STICK = 2
ITEM_STONE = 3
ITEM_WOOD_SWORD = 4
ITEM_STONE_SWORD = 5
ITEM_WOOD_PICKAXE = 6
ITEM_STONE_PICKAXE = 7
ITEM_APPLE = 8

CRAFT_ACTIONS = {
    9: ((ITEM_WOOD, 1), (ITEM_PLANK, 2), "plank"),
    10: ((ITEM_PLANK, 2), (ITEM_STICK, 4), "stick"),
    11: (((ITEM_STICK, 1), (ITEM_PLANK, 1)), (ITEM_WOOD_SWORD, 1), "wood_tool"),
    12: (((ITEM_STICK, 1), (ITEM_STONE, 1)), (ITEM_STONE_SWORD, 1), "stone_tool"),
    13: (((ITEM_STICK, 1), (ITEM_PLANK, 1)), (ITEM_WOOD_PICKAXE, 1), "wood_tool"),
    14: (((ITEM_STICK, 1), (ITEM_STONE, 1)), (ITEM_STONE_PICKAXE, 1), "stone_tool"),
}

ACTION_NAMES = (
    "stay",
    "move_n",
    "move_s",
    "move_w",
    "move_e",
    "harvest",
    "pickup",
    "attack",
    "eat",
    "craft_plank",
    "craft_stick",
    "craft_wood_sword",
    "craft_stone_sword",
    "craft_wood_pickaxe",
    "craft_stone_pickaxe",
)

EVENT_NAMES = (
    "move_new_cell",
    "move_known_cell",
    "move_blocked",
    "harvest_wood",
    "harvest_apple",
    "harvest_stone",
    "pickup_item",
    "eat_apple",
    "craft_plank",
    "craft_stick",
    "craft_wood_tool",
    "craft_stone_tool",
    "attack_hit",
    "mob_kill",
    "mob_kill_unarmed",
    "mob_kill_armed",
    "tool_equipped",
    "damage_taken",
    "starvation_damage",
    "health_regen",
    "death",
)

REWARD_COMPONENT_NAMES = (
    "survival",
    "exploration",
    "harvest_wood",
    "harvest_apple",
    "harvest_stone",
    "pickup_item",
    "eat_apple",
    "craft_plank",
    "craft_stick",
    "craft_wood_tool",
    "craft_stone_tool",
    "attack_hit",
    "mob_kill",
    "health_regen",
    "mob_damage",
    "starvation_damage",
    "episode_death",
    "milestone",
)

EVENT_LEVELS = {
    "move_new_cell": 1,
    "harvest_wood": 2,
    "harvest_apple": 2,
    "pickup_item": 2,
    "eat_apple": 2,
    "craft_plank": 3,
    "craft_stick": 4,
    "craft_wood_tool": 5,
    "harvest_stone": 6,
    "craft_stone_tool": 7,
    "mob_kill": 8,
}

EVENT_INDEX = {name: index for index, name in enumerate(EVENT_NAMES)}
REWARD_COMPONENT_INDEX = {name: index for index, name in enumerate(REWARD_COMPONENT_NAMES)}
ACTION_SUCCESS_EVENTS = {
    "move_n": ("move_new_cell", "move_known_cell"),
    "move_s": ("move_new_cell", "move_known_cell"),
    "move_w": ("move_new_cell", "move_known_cell"),
    "move_e": ("move_new_cell", "move_known_cell"),
    "harvest": ("harvest_wood", "harvest_stone"),
    "pickup": ("pickup_item",),
    "attack": ("attack_hit",),
    "eat": ("eat_apple",),
    "craft_plank": ("craft_plank",),
    "craft_stick": ("craft_stick",),
    "craft_wood_sword": ("craft_wood_tool",),
    "craft_stone_sword": ("craft_stone_tool",),
    "craft_wood_pickaxe": ("craft_wood_tool",),
    "craft_stone_pickaxe": ("craft_stone_tool",),
}


class VectorizedGridcraftEnv:
    """Batched Gridcraft environment.

    The class is intentionally independent from Gym/PettingZoo so it can be used
    both for fast rollout collection and inside a TorchRL wrapper.
    """

    def __init__(
        self,
        num_envs: int,
        num_agents: int | None = None,
        device: str | torch.device = "cpu",
        seed: int | None = None,
        config: VGridcraftConfig | None = None,
    ):
        self.config = config or VGridcraftConfig()
        if num_agents is not None:
            self.config.num_agents = int(num_agents)
        self.num_envs = int(num_envs)
        self.num_agents = int(self.config.num_agents)
        self.device = torch.device(device)
        self.generator = torch.Generator(device=self.device)
        self.seed = self.config.seed if seed is None else seed
        if self.seed is not None:
            self.generator.manual_seed(int(self.seed))
        self._allocate()
        self._renderer = None
        self.reset()

    def _allocate(self) -> None:
        cfg = self.config
        e, a, h, w = self.num_envs, self.num_agents, cfg.height, cfg.width
        d = self.device
        self.terrain = torch.full((e, h, w), TERRAIN_GRASS, dtype=torch.long, device=d)
        self.blocks = torch.full((e, h, w), BLOCK_EMPTY, dtype=torch.long, device=d)
        self.agent_x = torch.zeros((e, a), dtype=torch.long, device=d)
        self.agent_y = torch.zeros((e, a), dtype=torch.long, device=d)
        self.hp = torch.full((e, a), cfg.hp_max, dtype=torch.long, device=d)
        self.hunger = torch.full((e, a), cfg.hunger_max, dtype=torch.long, device=d)
        self.inventory = torch.zeros((e, a, cfg.item_classes), dtype=torch.long, device=d)
        self.equipped = torch.full((e, a), -1, dtype=torch.long, device=d)
        self.alive = torch.ones((e, a), dtype=torch.bool, device=d)
        self.visited = torch.zeros((e, a, h, w), dtype=torch.bool, device=d)
        self.move_counter = torch.zeros((e, a), dtype=torch.long, device=d)
        self.harvest_counter = torch.zeros((e, a), dtype=torch.long, device=d)
        self.attack_counter = torch.zeros((e, a), dtype=torch.long, device=d)
        self.last_attack_step = torch.full((e, a), -1, dtype=torch.long, device=d)
        self.task_level_max = torch.zeros((e, a), dtype=torch.long, device=d)
        self.complexity_cumulative = torch.zeros((e, a), dtype=torch.float32, device=d)
        self.complexity_exponential_cumulative = torch.zeros((e, a), dtype=torch.float32, device=d)
        self.complexity_unique = torch.zeros((e, a), dtype=torch.float32, device=d)
        self.unlocked_task_levels = torch.zeros((e, a, 9), dtype=torch.bool, device=d)
        self.team_unlocked_task_levels = torch.zeros((e, 9), dtype=torch.bool, device=d)
        self.pending_team_levels = torch.zeros((e, 9), dtype=torch.bool, device=d)
        self.rewarded_event_counts = torch.zeros(
            (e, a, len(REWARD_COMPONENT_NAMES)), dtype=torch.long, device=d
        )
        self.last_action_attempts = torch.zeros((e, a, len(ACTION_NAMES)), dtype=torch.float32, device=d)
        self.last_event_success = torch.zeros((e, a, len(EVENT_NAMES)), dtype=torch.float32, device=d)
        self.last_reward_components = torch.zeros(
            (e, a, len(REWARD_COMPONENT_NAMES)), dtype=torch.float32, device=d
        )
        self.mob_x = torch.zeros((e, cfg.max_mobs), dtype=torch.long, device=d)
        self.mob_y = torch.zeros((e, cfg.max_mobs), dtype=torch.long, device=d)
        self.mob_hp = torch.zeros((e, cfg.max_mobs), dtype=torch.long, device=d)
        self.mob_alive = torch.zeros((e, cfg.max_mobs), dtype=torch.bool, device=d)
        self.item_x = torch.zeros((e, cfg.max_items), dtype=torch.long, device=d)
        self.item_y = torch.zeros((e, cfg.max_items), dtype=torch.long, device=d)
        self.item_type = torch.zeros((e, cfg.max_items), dtype=torch.long, device=d)
        self.item_count = torch.zeros((e, cfg.max_items), dtype=torch.long, device=d)
        self.item_alive = torch.zeros((e, cfg.max_items), dtype=torch.bool, device=d)
        self.step_count = torch.zeros((e,), dtype=torch.long, device=d)

    def reset(self, env_ids: torch.Tensor | None = None, seed: int | None = None) -> dict[str, torch.Tensor]:
        if seed is not None:
            self.generator.manual_seed(int(seed))
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        env_ids = env_ids.to(self.device).long()
        cfg = self.config
        n = env_ids.numel()
        self.step_count[env_ids] = 0
        self.terrain[env_ids] = TERRAIN_GRASS
        water = torch.rand((n, cfg.height, cfg.width), generator=self.generator, device=self.device) < cfg.water_density
        dirt = torch.rand((n, cfg.height, cfg.width), generator=self.generator, device=self.device) < 0.05
        grass = torch.full((n, cfg.height, cfg.width), TERRAIN_GRASS, dtype=torch.long, device=self.device)
        water_values = torch.full_like(grass, TERRAIN_WATER)
        dirt_values = torch.full_like(grass, TERRAIN_DIRT)
        self.terrain[env_ids] = torch.where(water, water_values, torch.where(dirt, dirt_values, grass))
        self.blocks[env_ids] = BLOCK_EMPTY
        tree = torch.rand((n, cfg.height, cfg.width), generator=self.generator, device=self.device) < cfg.tree_density
        stone = torch.rand((n, cfg.height, cfg.width), generator=self.generator, device=self.device) < cfg.stone_density
        empty = torch.full((n, cfg.height, cfg.width), BLOCK_EMPTY, dtype=torch.long, device=self.device)
        tree_values = torch.full_like(empty, BLOCK_TREE)
        stone_values = torch.full_like(empty, BLOCK_STONE)
        self.blocks[env_ids] = torch.where(tree, tree_values, torch.where(stone, stone_values, empty))
        self.blocks[env_ids] = torch.where(self.terrain[env_ids] == TERRAIN_WATER, empty, self.blocks[env_ids])
        self.hp[env_ids] = cfg.hp_max
        self.hunger[env_ids] = cfg.hunger_max
        self.inventory[env_ids] = 0
        self.equipped[env_ids] = -1
        self.alive[env_ids] = True
        self.agent_x[env_ids] = -1
        self.agent_y[env_ids] = -1
        self.visited[env_ids] = False
        self.move_counter[env_ids] = 0
        self.harvest_counter[env_ids] = 0
        self.attack_counter[env_ids] = 0
        self.last_attack_step[env_ids] = -1
        self.task_level_max[env_ids] = 0
        self.complexity_cumulative[env_ids] = 0
        self.complexity_exponential_cumulative[env_ids] = 0
        self.complexity_unique[env_ids] = 0
        self.unlocked_task_levels[env_ids] = False
        self.team_unlocked_task_levels[env_ids] = False
        self.pending_team_levels[env_ids] = False
        self.rewarded_event_counts[env_ids] = 0
        self.last_action_attempts[env_ids] = 0
        self.last_event_success[env_ids] = 0
        self.last_reward_components[env_ids] = 0
        self.mob_alive[env_ids] = False
        self.item_alive[env_ids] = False
        for agent_idx in range(self.num_agents):
            x, y = self._sample_open_cells(env_ids)
            self.agent_x[env_ids, agent_idx] = x
            self.agent_y[env_ids, agent_idx] = y
            self.visited[env_ids, agent_idx, y, x] = True
        return self.observation()

    def step(self, actions: torch.Tensor) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        cfg = self.config
        actions = actions.to(self.device).long().reshape(self.num_envs, self.num_agents).clamp(0, cfg.action_size - 1)
        rewards = torch.zeros((self.num_envs, self.num_agents), dtype=torch.float32, device=self.device)
        alive_before = self.alive.clone()
        self.last_action_attempts.zero_()
        self.last_event_success.zero_()
        self.last_reward_components.zero_()
        self.pending_team_levels.zero_()
        self.last_action_attempts.scatter_(2, actions.unsqueeze(-1), self.alive.float().unsqueeze(-1))
        self.step_count += 1
        for agent_idx in range(self.num_agents):
            action = actions[:, agent_idx]
            self._apply_movement(agent_idx, action, rewards)
            self._apply_harvest(agent_idx, action, rewards)
            self._apply_pickup(agent_idx, action, rewards)
            self._apply_attack(agent_idx, action, rewards)
            self._apply_eat(agent_idx, action, rewards)
            self._apply_craft(agent_idx, action, rewards)
        self._move_mobs()
        self._resolve_mob_attacks(rewards)
        self._handle_hunger(rewards)
        self._apply_team_milestones(rewards)
        self.mob_alive &= self.mob_hp > 0
        done = ~self.alive.any(dim=1)
        truncated = self.step_count >= cfg.max_steps
        died = alive_before & ~self.alive
        died_envs, died_agents = torch.nonzero(died, as_tuple=True)
        if died_envs.numel() > 0:
            death_penalty = float(cfg.death_penalty)
            rewards[died_envs, died_agents] -= death_penalty
            self.last_reward_components[
                died_envs, died_agents, REWARD_COMPONENT_INDEX["episode_death"]
            ] -= death_penalty
        self._record_event_pairs(died_envs, died_agents, "death")
        if cfg.mob_spawn_rate > 0:
            spawn_envs = torch.nonzero(
                (self.step_count >= cfg.mob_spawn_grace_steps)
                & ((self.step_count - cfg.mob_spawn_grace_steps) % cfg.mob_spawn_rate == 0),
                as_tuple=False,
            ).flatten()
            for env_id in spawn_envs.tolist():
                free = torch.nonzero(~self.mob_alive[env_id], as_tuple=False).flatten()
                if free.numel() > 0:
                    self._spawn_mob(env_id, int(free[0]))
        component_sum = self.last_reward_components.sum(dim=-1)
        return self.observation(), rewards, done, truncated, {
            "config": asdict(cfg),
            "task_level_max": self.task_level_max.clone(),
            "action_attempts": self.last_action_attempts.clone(),
            "event_success": self.last_event_success.clone(),
            "reward_components": self.last_reward_components.clone(),
            "reward_component_sum": component_sum,
            "reward_decomposition_error": component_sum - rewards,
            "complexity_cumulative": self.complexity_cumulative.clone(),
            "complexity_exponential_cumulative": self.complexity_exponential_cumulative.clone(),
            "complexity_unique": self.complexity_unique.clone(),
            "team_task_level_max": (
                self.team_unlocked_task_levels.long()
                * torch.arange(9, device=self.device).unsqueeze(0)
            ).max(dim=1).values,
        }

    def observation(self) -> dict[str, torch.Tensor]:
        grid = self.local_grid()
        self_vec = torch.cat([self.hp[..., None], self.hunger[..., None], self.inventory], dim=-1)
        return {
            "grid": grid,
            "self": self_vec,
            "vector": self.obs_to_vector(grid, self_vec),
        }

    def local_grid(self) -> torch.Tensor:
        cfg = self.config
        radius = cfg.view_size // 2
        offsets = torch.arange(-radius, radius + 1, device=self.device)
        dy, dx = torch.meshgrid(offsets, offsets, indexing="ij")
        x = self.agent_x[:, :, None, None] + dx
        y = self.agent_y[:, :, None, None] + dy
        inside = (x >= 0) & (x < cfg.width) & (y >= 0) & (y < cfg.height)
        sx = x.clamp(0, cfg.width - 1)
        sy = y.clamp(0, cfg.height - 1)
        env_idx = torch.arange(self.num_envs, device=self.device)[:, None, None, None]
        entity = self.entity_map()
        terrain = torch.where(
            inside,
            self.terrain[env_idx, sy, sx],
            torch.full_like(sx, TERRAIN_WATER),
        )
        blocks = torch.where(
            inside,
            self.blocks[env_idx, sy, sx],
            torch.full_like(sx, BLOCK_EMPTY),
        )
        entities = torch.where(
            inside,
            entity[env_idx, sy, sx],
            torch.full_like(sx, ENTITY_NONE),
        )
        return torch.stack([terrain, blocks, entities], dim=2)

    def local_grid_loop(self) -> torch.Tensor:
        cfg = self.config
        radius = cfg.view_size // 2
        grid = torch.zeros((self.num_envs, self.num_agents, 3, cfg.view_size, cfg.view_size), dtype=torch.long, device=self.device)
        entity = self.entity_map()
        for gy, dy in enumerate(range(-radius, radius + 1)):
            for gx, dx in enumerate(range(-radius, radius + 1)):
                x = self.agent_x + dx
                y = self.agent_y + dy
                inside = (x >= 0) & (x < cfg.width) & (y >= 0) & (y < cfg.height)
                sx = x.clamp(0, cfg.width - 1)
                sy = y.clamp(0, cfg.height - 1)
                env_idx = torch.arange(self.num_envs, device=self.device)[:, None].expand(-1, self.num_agents)
                grid[:, :, 0, gy, gx] = torch.where(inside, self.terrain[env_idx, sy, sx], torch.tensor(TERRAIN_WATER, device=self.device))
                grid[:, :, 1, gy, gx] = torch.where(inside, self.blocks[env_idx, sy, sx], torch.tensor(BLOCK_EMPTY, device=self.device))
                grid[:, :, 2, gy, gx] = torch.where(inside, entity[env_idx, sy, sx], torch.tensor(ENTITY_NONE, device=self.device))
        return grid

    def obs_to_vector(self, grid: torch.Tensor, self_vec: torch.Tensor) -> torch.Tensor:
        cfg = self.config
        terrain = F.one_hot(grid[:, :, 0].clamp(0, cfg.terrain_classes - 1), cfg.terrain_classes).float().flatten(start_dim=2)
        blocks = F.one_hot(grid[:, :, 1].clamp(0, cfg.block_classes - 1), cfg.block_classes).float().flatten(start_dim=2)
        entities = F.one_hot(grid[:, :, 2].clamp(0, cfg.entity_classes - 1), cfg.entity_classes).float().flatten(start_dim=2)
        numeric = torch.zeros_like(self_vec, dtype=torch.float32)
        numeric[..., :2] = self_vec[..., :2].float() / 20.0
        numeric[..., 2:] = self_vec[..., 2:].float().clamp(0, 10) / 10.0
        return torch.cat([terrain, blocks, entities, numeric], dim=-1)

    def render(
        self,
        env_index: int = 0,
        mode: str = "rgb_array",
        tabular_observations: object | None = None,
        overlay_info: object | None = None,
    ):
        """Render one environment from the batch with Gridcraft's PygameRenderer.

        Args:
            env_index: Batch index to render.
            mode: `"rgb_array"` returns an RGB frame, `"human"` opens a pygame window.
            tabular_observations: Optional extra imagined observations. When provided,
                the Gridcraft renderer appends them to the right of the real world and
                real observations, matching the existing comparison view.
        """
        renderer, render_config = self._ensure_renderer()
        world = self.to_gridcraft_world(env_index=env_index, render_config=render_config)
        return renderer.render(world, mode, tabular_observations=tabular_observations, overlay_info=overlay_info)

    def close(self) -> None:
        if self._renderer is not None:
            renderer, _ = self._renderer
            renderer.close()
            self._renderer = None

    def to_gridcraft_world(self, env_index: int = 0, render_config=None):
        modules = _gridcraft_render_modules()
        Item = modules["Item"]
        AgentState = modules["AgentState"]
        MobState = modules["MobState"]
        ItemDrop = modules["ItemDrop"]
        env_index = int(env_index)
        if env_index < 0 or env_index >= self.num_envs:
            raise IndexError(f"env_index {env_index} out of range for num_envs={self.num_envs}")

        class _VectorizedWorldSnapshot:
            def observations(snapshot_self):
                return self._observations_numpy(env_index)

        snapshot = _VectorizedWorldSnapshot()
        snapshot.config = render_config
        snapshot.terrain = self.terrain[env_index].detach().cpu().numpy().astype("int8")
        snapshot.blocks = self.blocks[env_index].detach().cpu().numpy().astype("int8")
        snapshot.agents = {}
        for agent_idx in range(self.num_agents):
            inventory = {
                Item(item_idx): int(self.inventory[env_index, agent_idx, item_idx].detach().cpu())
                for item_idx in range(self.config.item_classes)
                if int(self.inventory[env_index, agent_idx, item_idx].detach().cpu()) > 0
            }
            equipped_idx = int(self.equipped[env_index, agent_idx].detach().cpu())
            snapshot.agents[f"agent_{agent_idx}"] = AgentState(
                agent_id=f"agent_{agent_idx}",
                x=int(self.agent_x[env_index, agent_idx].detach().cpu()),
                y=int(self.agent_y[env_index, agent_idx].detach().cpu()),
                hp=int(self.hp[env_index, agent_idx].detach().cpu()),
                hunger=int(self.hunger[env_index, agent_idx].detach().cpu()),
                inventory=inventory,
                inventory_order=list(Item),
                equipped=Item(equipped_idx) if equipped_idx >= 0 else None,
                alive=bool(self.alive[env_index, agent_idx].detach().cpu()),
            )
        snapshot.mobs = [
            MobState(
                mob_id=mob_idx + 1,
                x=int(self.mob_x[env_index, mob_idx].detach().cpu()),
                y=int(self.mob_y[env_index, mob_idx].detach().cpu()),
                hp=int(self.mob_hp[env_index, mob_idx].detach().cpu()),
                alive=True,
            )
            for mob_idx in range(self.config.max_mobs)
            if bool(self.mob_alive[env_index, mob_idx].detach().cpu())
        ]
        snapshot.items = [
            ItemDrop(
                item=Item(int(self.item_type[env_index, item_idx].detach().cpu())),
                count=int(self.item_count[env_index, item_idx].detach().cpu()),
                x=int(self.item_x[env_index, item_idx].detach().cpu()),
                y=int(self.item_y[env_index, item_idx].detach().cpu()),
            )
            for item_idx in range(self.config.max_items)
            if bool(self.item_alive[env_index, item_idx].detach().cpu())
        ]
        return snapshot

    def _observations_numpy(self, env_index: int) -> dict[str, dict]:
        obs = self.observation()
        grid = obs["grid"][env_index].detach().cpu().numpy().astype("int8")
        self_vec = obs["self"][env_index].detach().cpu().numpy().astype("int16")
        return {
            f"agent_{agent_idx}": {
                "grid": grid[agent_idx],
                "self": self_vec[agent_idx],
            }
            for agent_idx in range(self.num_agents)
        }

    def _ensure_renderer(self):
        if self._renderer is not None:
            return self._renderer
        modules = _gridcraft_render_modules()
        GridcraftConfig = modules["GridcraftConfig"]
        PygameRenderer = modules["PygameRenderer"]
        render_config = GridcraftConfig(
            width=self.config.width,
            height=self.config.height,
            num_agents=self.config.num_agents,
            view_size=self.config.view_size,
            max_steps=self.config.max_steps,
            seed=self.config.seed,
            tile_size=self.config.tile_size,
            fps=self.config.fps,
        )
        self._renderer = (PygameRenderer(render_config), render_config)
        return self._renderer

    def entity_map(self) -> torch.Tensor:
        cfg = self.config
        entity = torch.zeros((self.num_envs, cfg.height, cfg.width), dtype=torch.long, device=self.device)
        env_idx = torch.arange(self.num_envs, device=self.device)[:, None]
        for agent_idx in range(self.num_agents):
            alive = self.alive[:, agent_idx]
            entity[torch.arange(self.num_envs, device=self.device)[alive], self.agent_y[alive, agent_idx], self.agent_x[alive, agent_idx]] = ENTITY_AGENT
        for mob_idx in range(cfg.max_mobs):
            alive = self.mob_alive[:, mob_idx]
            entity[torch.arange(self.num_envs, device=self.device)[alive], self.mob_y[alive, mob_idx], self.mob_x[alive, mob_idx]] = ENTITY_MOB
        for item_idx in range(cfg.max_items):
            alive = self.item_alive[:, item_idx]
            entity[torch.arange(self.num_envs, device=self.device)[alive], self.item_y[alive, item_idx], self.item_x[alive, item_idx]] = ENTITY_ITEM
        return entity

    def _apply_movement(self, agent_idx: int, action: torch.Tensor, rewards: torch.Tensor) -> None:
        move = (action >= 1) & (action <= 4) & self.alive[:, agent_idx]
        dx = torch.zeros_like(action)
        dy = torch.zeros_like(action)
        dy = torch.where(action == 1, -1, dy)
        dy = torch.where(action == 2, 1, dy)
        dx = torch.where(action == 3, -1, dx)
        dx = torch.where(action == 4, 1, dx)
        nx = self.agent_x[:, agent_idx] + dx
        ny = self.agent_y[:, agent_idx] + dy
        can = move & self.is_walkable(nx, ny) & ~self._occupied_by_agent(nx, ny, ignore_agent=agent_idx) & ~self._occupied_by_mob(nx, ny)
        blocked_envs = torch.nonzero(move & ~can, as_tuple=False).flatten()
        self._record_event(blocked_envs, agent_idx, "move_blocked")
        env_ids = torch.nonzero(can, as_tuple=False).flatten()
        if env_ids.numel() == 0:
            return
        old_new = ~self.visited[env_ids, agent_idx, ny[env_ids], nx[env_ids]]
        self.agent_x[env_ids, agent_idx] = nx[env_ids]
        self.agent_y[env_ids, agent_idx] = ny[env_ids]
        self.visited[env_ids, agent_idx, ny[env_ids], nx[env_ids]] = True
        new_envs = env_ids[old_new]
        known_envs = env_ids[~old_new]
        self._record_event(new_envs, agent_idx, "move_new_cell")
        self._record_event(known_envs, agent_idx, "move_known_cell")
        self._award_dense(
            rewards, new_envs, agent_idx, "exploration", self.config.new_cell_reward
        )
        self._mark_task_level(new_envs, agent_idx, 1)
        self._charge_hunger(env_ids, agent_idx, "move")

    def _apply_harvest(self, agent_idx: int, action: torch.Tensor, rewards: torch.Tensor) -> None:
        envs = torch.nonzero((action == 5) & self.alive[:, agent_idx], as_tuple=False).flatten()
        if envs.numel() == 0:
            return
        dirs = torch.tensor([[-1, 0], [1, 0], [0, -1], [0, 1]], dtype=torch.long, device=self.device)
        x = self.agent_x[envs, agent_idx][:, None] + dirs[None, :, 0]
        y = self.agent_y[envs, agent_idx][:, None] + dirs[None, :, 1]
        inside = (x >= 0) & (x < self.config.width) & (y >= 0) & (y < self.config.height)
        sx = x.clamp(0, self.config.width - 1)
        sy = y.clamp(0, self.config.height - 1)
        block = self.blocks[envs[:, None], sy, sx]
        tree = inside & (block == BLOCK_TREE)
        has_pickaxe = torch.isin(self.equipped[envs, agent_idx], torch.tensor([ITEM_WOOD_PICKAXE, ITEM_STONE_PICKAXE], device=self.device))
        stone = inside & (block == BLOCK_STONE) & has_pickaxe[:, None]
        harvestable = tree | stone
        has_target = harvestable.any(dim=1)
        if not bool(has_target.any()):
            return
        first = harvestable.float().argmax(dim=1)
        target_envs = envs[has_target]
        target_dir = first[has_target]
        target_x = x[has_target, target_dir]
        target_y = y[has_target, target_dir]
        target_block = block[has_target, target_dir]
        self.blocks[target_envs, target_y, target_x] = BLOCK_EMPTY
        tree_envs = target_envs[target_block == BLOCK_TREE]
        stone_envs = target_envs[target_block == BLOCK_STONE]
        if tree_envs.numel() > 0:
            self.inventory[tree_envs, agent_idx, ITEM_WOOD] += 1
            self._record_event(tree_envs, agent_idx, "harvest_wood")
            self._award_dense(
                rewards, tree_envs, agent_idx, "harvest_wood", self.config.harvest_wood_reward
            )
            self._mark_task_level(tree_envs, agent_idx, 2)
            apple = torch.rand((tree_envs.numel(),), generator=self.generator, device=self.device) < self.config.tree_apple_drop_chance
            apple_envs = tree_envs[apple]
            self.inventory[apple_envs, agent_idx, ITEM_APPLE] += 1
            self._record_event(apple_envs, agent_idx, "harvest_apple")
            self._award_dense(
                rewards,
                apple_envs,
                agent_idx,
                "harvest_apple",
                self.config.harvest_tree_apple_reward,
            )
        if stone_envs.numel() > 0:
            self.inventory[stone_envs, agent_idx, ITEM_STONE] += 1
            self._record_event(stone_envs, agent_idx, "harvest_stone")
            self._award_dense(
                rewards, stone_envs, agent_idx, "harvest_stone", self.config.harvest_stone_reward
            )
            self._mark_task_level(stone_envs, agent_idx, 6)
        self._charge_hunger(target_envs, agent_idx, "harvest")

    def _apply_pickup(self, agent_idx: int, action: torch.Tensor, rewards: torch.Tensor) -> None:
        envs = torch.nonzero((action == 6) & self.alive[:, agent_idx], as_tuple=False).flatten()
        if envs.numel() == 0:
            return
        x = self.agent_x[envs, agent_idx][:, None]
        y = self.agent_y[envs, agent_idx][:, None]
        reachable = self.item_alive[envs] & ((self.item_x[envs] - x).abs() + (self.item_y[envs] - y).abs() == 1)
        if not bool(reachable.any()):
            return
        row, item_slot = torch.nonzero(reachable, as_tuple=True)
        target_envs = envs[row]
        item_type = self.item_type[target_envs, item_slot]
        count = self.item_count[target_envs, item_slot]
        agent_index = torch.full_like(target_envs, int(agent_idx))
        self.inventory.index_put_((target_envs, agent_index, item_type), count, accumulate=True)
        self._record_event(target_envs, agent_idx, "pickup_item")
        self._award_dense(
            rewards,
            target_envs,
            agent_idx,
            "pickup_item",
            count.float() * self.config.pickup_item_reward,
        )
        stone_envs = target_envs[item_type == ITEM_STONE]
        other_envs = target_envs[item_type != ITEM_STONE]
        self._mark_task_level(other_envs.unique(), agent_idx, 2)
        self._mark_task_level(stone_envs.unique(), agent_idx, 6)
        self.item_alive[target_envs, item_slot] = False

    def _apply_attack(self, agent_idx: int, action: torch.Tensor, rewards: torch.Tensor) -> None:
        envs = torch.nonzero((action == 7) & self.alive[:, agent_idx], as_tuple=False).flatten()
        if envs.numel() == 0:
            return
        equipped_all = self.equipped[envs, agent_idx]
        armed = torch.isin(
            equipped_all,
            torch.tensor([ITEM_WOOD_SWORD, ITEM_STONE_SWORD], device=self.device),
        )
        envs = envs[armed]
        if envs.numel() == 0:
            return
        x = self.agent_x[envs, agent_idx][:, None]
        y = self.agent_y[envs, agent_idx][:, None]
        adjacent = self.mob_alive[envs] & ((self.mob_x[envs] - x).abs() + (self.mob_y[envs] - y).abs() == 1)
        has_target = adjacent.any(dim=1)
        if not bool(has_target.any()):
            return
        first = adjacent.float().argmax(dim=1)
        target_envs = envs[has_target]
        target_mobs = first[has_target]
        self.last_attack_step[target_envs, agent_idx] = self.step_count[target_envs]
        self._charge_hunger(target_envs, agent_idx, "attack")
        self._record_event(target_envs, agent_idx, "attack_hit")
        self._award_dense(
            rewards, target_envs, agent_idx, "attack_hit", self.config.attack_hit_reward
        )
        equipped = self.equipped[target_envs, agent_idx]
        damage = torch.where(
            equipped == ITEM_STONE_SWORD,
            torch.full_like(equipped, self.config.stone_sword_damage),
            torch.full_like(equipped, self.config.wood_sword_damage),
        )
        self.mob_hp[target_envs, target_mobs] -= damage
        killed = self.mob_hp[target_envs, target_mobs] <= 0
        killed_envs = target_envs[killed]
        killed_mobs = target_mobs[killed]
        if killed_envs.numel() > 0:
            self.mob_alive[killed_envs, killed_mobs] = False
            self._record_event(killed_envs, agent_idx, "mob_kill")
            self._award_dense(
                rewards, killed_envs, agent_idx, "mob_kill", self.config.mob_kill_reward
            )
            killed_equipped = equipped[killed]
            armed = torch.ones_like(killed_equipped, dtype=torch.bool)
            self._record_event(killed_envs, agent_idx, "mob_kill_armed")
            self._mark_task_level(killed_envs, agent_idx, 8)
            drop = torch.rand((killed_envs.numel(),), generator=self.generator, device=self.device) < self.config.item_drop_chance
            for env_id, mob_idx in zip(killed_envs[drop].tolist(), killed_mobs[drop].tolist()):
                self._drop_item(env_id, ITEM_APPLE, 1, int(self.mob_x[env_id, mob_idx]), int(self.mob_y[env_id, mob_idx]))

    def _apply_eat(self, agent_idx: int, action: torch.Tensor, rewards: torch.Tensor) -> None:
        can = (action == 8) & self.alive[:, agent_idx] & (self.hunger[:, agent_idx] < self.config.hunger_max) & (self.inventory[:, agent_idx, ITEM_APPLE] > 0)
        envs = torch.nonzero(can, as_tuple=False).flatten()
        self.inventory[envs, agent_idx, ITEM_APPLE] -= 1
        self.hunger[envs, agent_idx] = (self.hunger[envs, agent_idx] + 6).clamp(max=self.config.hunger_max)
        self._record_event(envs, agent_idx, "eat_apple")
        self._award_dense(
            rewards, envs, agent_idx, "eat_apple", self.config.eat_apple_reward
        )

    def _apply_craft(self, agent_idx: int, action: torch.Tensor, rewards: torch.Tensor) -> None:
        for action_id, (inputs, output, reward_name) in CRAFT_ACTIONS.items():
            envs = torch.nonzero((action == action_id) & self.alive[:, agent_idx], as_tuple=False).flatten()
            if envs.numel() == 0:
                continue
            if isinstance(inputs[0], tuple):
                required = inputs
            else:
                required = (inputs,)
            out_item, out_count = output
            can = torch.ones((envs.numel(),), dtype=torch.bool, device=self.device)
            for item, count in required:
                can &= self.inventory[envs, agent_idx, item] >= count
            envs = envs[can]
            if envs.numel() == 0:
                continue
            for item, count in required:
                self.inventory[envs, agent_idx, item] -= count
            self.inventory[envs, agent_idx, out_item] += out_count
            if out_item in (ITEM_WOOD_SWORD, ITEM_STONE_SWORD, ITEM_WOOD_PICKAXE, ITEM_STONE_PICKAXE):
                self.equipped[envs, agent_idx] = out_item
            event_name = f"craft_{reward_name}"
            reward_value = self._craft_reward(reward_name)
            self._record_event(envs, agent_idx, event_name)
            self._award_dense(rewards, envs, agent_idx, event_name, reward_value)
            if reward_name in ("wood_tool", "stone_tool"):
                self._record_event(envs, agent_idx, "tool_equipped")
            self._mark_task_level(envs, agent_idx, self._craft_task_level(reward_name))

    def _craft_reward(self, reward_name: str) -> float:
        if reward_name == "plank":
            return self.config.craft_plank_reward
        if reward_name == "stick":
            return self.config.craft_stick_reward
        if reward_name == "wood_tool":
            return self.config.craft_wood_tool_reward
        if reward_name == "stone_tool":
            return self.config.craft_stone_tool_reward
        return 0.0

    @staticmethod
    def _craft_task_level(reward_name: str) -> int:
        if reward_name == "plank":
            return 3
        if reward_name == "stick":
            return 4
        if reward_name == "wood_tool":
            return 5
        if reward_name == "stone_tool":
            return 7
        return 0

    def _mark_task_level(self, envs: torch.Tensor, agent_idx: int, level: int) -> None:
        if envs.numel() == 0:
            return
        level_tensor = torch.full((envs.numel(),), int(level), dtype=torch.long, device=self.device)
        self.task_level_max[envs, agent_idx] = torch.maximum(self.task_level_max[envs, agent_idx], level_tensor)
        self.pending_team_levels[envs, int(level)] = True

    def _record_event(
        self,
        envs: torch.Tensor,
        agent_idx: int,
        name: str,
        *,
        counts: torch.Tensor | None = None,
    ) -> None:
        if envs.numel() == 0:
            return
        if counts is None:
            counts = torch.ones((envs.numel(),), dtype=torch.float32, device=self.device)
        else:
            counts = counts.to(self.device).float()
        agents = torch.full_like(envs, int(agent_idx))
        self.last_event_success.index_put_(
            (envs, agents, torch.full_like(envs, EVENT_INDEX[name])),
            counts,
            accumulate=True,
        )
        level = EVENT_LEVELS.get(name, 0)
        if level <= 0:
            return
        self.complexity_cumulative.index_put_(
            (envs, agents),
            counts * float(level),
            accumulate=True,
        )
        self.complexity_exponential_cumulative.index_put_(
            (envs, agents),
            counts * float(2 ** (level - 1)),
            accumulate=True,
        )
        unique_envs = envs.unique()
        unique_agents = torch.full_like(unique_envs, int(agent_idx))
        newly_unlocked = ~self.unlocked_task_levels[unique_envs, unique_agents, level]
        if bool(newly_unlocked.any()):
            unlock_envs = unique_envs[newly_unlocked]
            unlock_agents = unique_agents[newly_unlocked]
            self.unlocked_task_levels[unlock_envs, unlock_agents, level] = True
            self.complexity_unique[unlock_envs, unlock_agents] += float(2 ** (level - 1))

    def _record_event_pairs(self, envs: torch.Tensor, agents: torch.Tensor, name: str) -> None:
        if envs.numel() == 0:
            return
        self.last_event_success.index_put_(
            (envs, agents, torch.full_like(envs, EVENT_INDEX[name])),
            torch.ones((envs.numel(),), dtype=torch.float32, device=self.device),
            accumulate=True,
        )

    def _record_reward(
        self,
        envs: torch.Tensor,
        agent_idx: int,
        component: str,
        values: float | torch.Tensor,
    ) -> None:
        if envs.numel() == 0:
            return
        if not torch.is_tensor(values):
            values = torch.full((envs.numel(),), float(values), dtype=torch.float32, device=self.device)
        else:
            values = values.to(self.device).float()
        agents = torch.full_like(envs, int(agent_idx))
        components = torch.full_like(envs, REWARD_COMPONENT_INDEX[component])
        self.last_reward_components.index_put_(
            (envs, agents, components),
            values,
            accumulate=True,
        )

    def _award_dense(
        self,
        rewards: torch.Tensor,
        envs: torch.Tensor,
        agent_idx: int,
        component: str,
        values: float | torch.Tensor,
    ) -> None:
        if envs.numel() == 0:
            return
        component_idx = REWARD_COMPONENT_INDEX[component]
        agents = torch.full_like(envs, int(agent_idx))
        under_cap = (
            self.rewarded_event_counts[envs, agents, component_idx]
            < self.config.dense_reward_event_cap
        )
        envs = envs[under_cap]
        agents = agents[under_cap]
        if envs.numel() == 0:
            return
        if not torch.is_tensor(values):
            values = torch.full(
                (under_cap.numel(),), float(values), dtype=torch.float32, device=self.device
            )
        else:
            values = values.to(self.device).float()
        values = values[under_cap]
        rewards.index_put_((envs, agents), values, accumulate=True)
        self.rewarded_event_counts[envs, agents, component_idx] += 1
        self._record_reward(envs, agent_idx, component, values)

    def _apply_team_milestones(self, rewards: torch.Tensor) -> None:
        milestone_component = REWARD_COMPONENT_INDEX["milestone"]
        for level in range(1, 9):
            event_indices = [
                EVENT_INDEX[name] for name, event_level in EVENT_LEVELS.items()
                if event_level == level
            ]
            if not event_indices:
                continue
            reached = (
                self.last_event_success[:, :, event_indices].sum(dim=(1, 2)) > 0
            ) | self.pending_team_levels[:, level]
            newly_reached = reached & ~self.team_unlocked_task_levels[:, level]
            envs = torch.nonzero(newly_reached, as_tuple=False).flatten()
            if envs.numel() == 0:
                continue
            self.team_unlocked_task_levels[envs, level] = True
            per_agent = float(self.config.task_milestone_rewards[level]) / self.num_agents
            rewards[envs] += per_agent
            self.last_reward_components[envs, :, milestone_component] += per_agent

    def _move_mobs(self) -> None:
        # Batched random wandering. It preserves collision constraints and is the
        # fast path needed for rollout diversity; exact pathfinding can be added
        # without changing the public API.
        dirs = torch.tensor([[1, 0], [-1, 0], [0, 1], [0, -1]], dtype=torch.long, device=self.device)
        for mob_idx in range(self.config.max_mobs):
            active = self.mob_alive[:, mob_idx] & (torch.rand((self.num_envs,), generator=self.generator, device=self.device) <= self.config.mob_move_prob)
            choices = dirs[torch.randint(0, 4, (self.num_envs,), generator=self.generator, device=self.device)]
            nx = self.mob_x[:, mob_idx] + choices[:, 0]
            ny = self.mob_y[:, mob_idx] + choices[:, 1]
            can = active & self.is_walkable(nx, ny) & ~self._occupied_by_agent(nx, ny) & ~self._occupied_by_mob(nx, ny, ignore_mob=mob_idx)
            envs = torch.nonzero(can, as_tuple=False).flatten()
            self.mob_x[envs, mob_idx] = nx[envs]
            self.mob_y[envs, mob_idx] = ny[envs]

    def _resolve_mob_attacks(self, rewards: torch.Tensor) -> None:
        for agent_idx in range(self.num_agents):
            for mob_idx in range(self.config.max_mobs):
                adjacent = self.mob_alive[:, mob_idx] & self.alive[:, agent_idx]
                adjacent &= (self.mob_x[:, mob_idx] - self.agent_x[:, agent_idx]).abs() + (self.mob_y[:, mob_idx] - self.agent_y[:, agent_idx]).abs() == 1
                adjacent &= self.last_attack_step[:, agent_idx] != self.step_count
                envs = torch.nonzero(adjacent, as_tuple=False).flatten()
                self.hp[envs, agent_idx] -= self.config.mob_damage
                rewards[envs, agent_idx] -= float(self.config.mob_damage)
                self._record_event(envs, agent_idx, "damage_taken")
                self._record_reward(
                    envs, agent_idx, "mob_damage", -float(self.config.mob_damage)
                )
                self.alive[envs, agent_idx] &= self.hp[envs, agent_idx] > 0

    def _handle_hunger(self, rewards: torch.Tensor) -> None:
        cfg = self.config
        if cfg.health_regen_ticks > 0:
            envs = torch.nonzero((self.step_count % cfg.health_regen_ticks) == 0, as_tuple=False).flatten()
            if envs.numel() > 0:
                can = self.alive[envs] & (self.hunger[envs] == cfg.hunger_max) & (self.hp[envs] < cfg.hp_max)
                self.hp[envs] = torch.where(can, (self.hp[envs] + 1).clamp(max=cfg.hp_max), self.hp[envs])
                regen_env_rows, regen_agents = torch.nonzero(can, as_tuple=True)
                regen_envs = envs[regen_env_rows]
                self._record_event_pairs(regen_envs, regen_agents, "health_regen")
                if regen_envs.numel() > 0 and cfg.health_regen_reward:
                    for agent_idx in range(self.num_agents):
                        agent_envs = regen_envs[regen_agents == agent_idx]
                        self._award_dense(
                            rewards,
                            agent_envs,
                            agent_idx,
                            "health_regen",
                            cfg.health_regen_reward,
                        )
        if cfg.hunger_decay_ticks <= 0:
            return
        envs = torch.nonzero((self.step_count % cfg.hunger_decay_ticks) == 0, as_tuple=False).flatten()
        starving = self.alive[envs] & (self.hunger[envs] == 0)
        old_hp = self.hp[envs].clone()
        self.hp[envs] = torch.where(starving, (self.hp[envs] - cfg.starvation_damage).clamp(min=cfg.starvation_min_hp), self.hp[envs])
        damage = (old_hp - self.hp[envs]).float()
        rewards[envs] -= damage
        damage_rows, damage_agents = torch.nonzero(damage > 0, as_tuple=True)
        damage_envs = envs[damage_rows]
        self._record_event_pairs(damage_envs, damage_agents, "starvation_damage")
        if damage_envs.numel() > 0:
            self.last_reward_components[
                damage_envs,
                damage_agents,
                REWARD_COMPONENT_INDEX["starvation_damage"],
            ] -= damage[damage_rows, damage_agents]

    def _charge_hunger(self, envs: torch.Tensor, agent_idx: int, kind: str) -> None:
        if kind == "move":
            counter = self.move_counter
            interval = self.config.move_hunger_cost_interval
        elif kind == "harvest":
            counter = self.harvest_counter
            interval = self.config.harvest_hunger_cost_interval
        else:
            counter = self.attack_counter
            interval = self.config.attack_hunger_cost_interval
        if interval <= 0:
            self.hunger[envs, agent_idx] = (self.hunger[envs, agent_idx] - 1).clamp(min=0)
            return
        counter[envs, agent_idx] += 1
        pay = counter[envs, agent_idx] >= interval
        pay_envs = envs[pay]
        self.hunger[pay_envs, agent_idx] = (self.hunger[pay_envs, agent_idx] - 1).clamp(min=0)
        counter[pay_envs, agent_idx] = 0

    def is_walkable(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        inside = (x >= 0) & (x < self.config.width) & (y >= 0) & (y < self.config.height)
        sx = x.clamp(0, self.config.width - 1)
        sy = y.clamp(0, self.config.height - 1)
        envs = torch.arange(self.num_envs, device=self.device)
        terrain = self.terrain[envs, sy, sx]
        blocks = self.blocks[envs, sy, sx]
        return inside & (terrain != TERRAIN_WATER) & (blocks != BLOCK_TREE) & (blocks != BLOCK_STONE)

    def _occupied_by_agent(self, x: torch.Tensor, y: torch.Tensor, ignore_agent: int | None = None) -> torch.Tensor:
        occupied = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        for agent_idx in range(self.num_agents):
            if ignore_agent is not None and agent_idx == ignore_agent:
                continue
            occupied |= self.alive[:, agent_idx] & (self.agent_x[:, agent_idx] == x) & (self.agent_y[:, agent_idx] == y)
        return occupied

    def _occupied_by_mob(self, x: torch.Tensor, y: torch.Tensor, ignore_mob: int | None = None) -> torch.Tensor:
        occupied = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        for mob_idx in range(self.config.max_mobs):
            if ignore_mob is not None and mob_idx == ignore_mob:
                continue
            occupied |= self.mob_alive[:, mob_idx] & (self.mob_x[:, mob_idx] == x) & (self.mob_y[:, mob_idx] == y)
        return occupied

    def _sample_open_cell(self, env_id: int) -> tuple[int, int]:
        for _ in range(1000):
            x = int(torch.randint(0, self.config.width, (), generator=self.generator, device=self.device))
            y = int(torch.randint(0, self.config.height, (), generator=self.generator, device=self.device))
            if self.terrain[env_id, y, x] != TERRAIN_WATER and self.blocks[env_id, y, x] == BLOCK_EMPTY:
                if not bool(((self.agent_x[env_id] == x) & (self.agent_y[env_id] == y) & self.alive[env_id]).any()):
                    return x, y
        raise RuntimeError("failed to sample open cell")

    def _sample_open_cells(self, env_ids: torch.Tensor, attempts: int = 256) -> tuple[torch.Tensor, torch.Tensor]:
        env_ids = env_ids.to(self.device).long()
        n = env_ids.numel()
        if n == 0:
            empty = torch.empty((0,), dtype=torch.long, device=self.device)
            return empty, empty
        x_candidates = torch.randint(0, self.config.width, (n, attempts), generator=self.generator, device=self.device)
        y_candidates = torch.randint(0, self.config.height, (n, attempts), generator=self.generator, device=self.device)
        env_idx = env_ids[:, None].expand(-1, attempts)
        open_cell = (self.terrain[env_idx, y_candidates, x_candidates] != TERRAIN_WATER) & (
            self.blocks[env_idx, y_candidates, x_candidates] == BLOCK_EMPTY
        )
        for agent_idx in range(self.num_agents):
            occupied = self.alive[env_ids, agent_idx][:, None] & (
                self.agent_x[env_ids, agent_idx][:, None] == x_candidates
            ) & (
                self.agent_y[env_ids, agent_idx][:, None] == y_candidates
            )
            open_cell &= ~occupied
        valid = open_cell.any(dim=1)
        first = open_cell.float().argmax(dim=1)
        x = x_candidates[torch.arange(n, device=self.device), first]
        y = y_candidates[torch.arange(n, device=self.device), first]
        if bool((~valid).any()):
            for row in torch.nonzero(~valid, as_tuple=False).flatten().tolist():
                fallback_x, fallback_y = self._sample_open_cell(int(env_ids[row]))
                x[row] = fallback_x
                y[row] = fallback_y
        return x, y

    def _spawn_mob(self, env_id: int, mob_idx: int) -> None:
        try:
            x, y = self._sample_open_cell(env_id)
        except RuntimeError:
            return
        self.mob_x[env_id, mob_idx] = x
        self.mob_y[env_id, mob_idx] = y
        self.mob_hp[env_id, mob_idx] = self.config.mob_hp
        self.mob_alive[env_id, mob_idx] = True

    def _spawn_mobs(self, env_ids: torch.Tensor, mob_idx: int) -> None:
        env_ids = env_ids.to(self.device).long()
        if env_ids.numel() == 0:
            return
        x, y = self._sample_open_cells(env_ids)
        self.mob_x[env_ids, mob_idx] = x
        self.mob_y[env_ids, mob_idx] = y
        self.mob_hp[env_ids, mob_idx] = self.config.mob_hp
        self.mob_alive[env_ids, mob_idx] = True

    def _drop_item(self, env_id: int, item: int, count: int, x: int, y: int) -> None:
        free = torch.nonzero(~self.item_alive[env_id], as_tuple=False).flatten()
        if free.numel() == 0:
            return
        idx = int(free[0])
        self.item_x[env_id, idx] = x
        self.item_y[env_id, idx] = y
        self.item_type[env_id, idx] = item
        self.item_count[env_id, idx] = count
        self.item_alive[env_id, idx] = True


def _gridcraft_render_modules():
    root = Path(__file__).resolve().parents[2]
    gridcraft_path = root / "Gridcraft"
    if gridcraft_path.exists() and str(gridcraft_path) not in sys.path:
        sys.path.insert(0, str(gridcraft_path))
    from gridcraft.config import GridcraftConfig
    from gridcraft.constants import Item
    from gridcraft.entities import AgentState, ItemDrop, MobState
    from gridcraft.render import PygameRenderer

    return {
        "GridcraftConfig": GridcraftConfig,
        "Item": Item,
        "AgentState": AgentState,
        "ItemDrop": ItemDrop,
        "MobState": MobState,
        "PygameRenderer": PygameRenderer,
    }
