from __future__ import annotations

from dataclasses import dataclass


@dataclass
class VGridcraftConfig:
    width: int = 16
    height: int = 16
    num_agents: int = 1
    view_size: int = 7
    max_steps: int = 500
    seed: int | None = None

    tree_density: float = 0.08
    stone_density: float = 0.06
    water_density: float = 0.06

    hp_max: int = 20
    hunger_max: int = 20
    hunger_decay_ticks: int = 5
    starvation_damage: int = 1
    starvation_min_hp: int = 1
    health_regen_ticks: int = 5
    move_hunger_cost_interval: int = 5
    harvest_hunger_cost_interval: int = 3
    attack_hunger_cost_interval: int = 3

    mob_spawn_rate: int = 10
    max_mobs: int = 6
    mob_damage: int = 2
    mob_hp: int = 10
    mob_aggro_radius: int = 6
    mob_move_prob: float = 0.8

    max_items: int = 32
    item_drop_chance: float = 0.2
    tree_apple_drop_chance: float = 0.5

    survival_reward: float = 0.001
    new_cell_reward: float = 0.01
    harvest_wood_reward: float = 1.0
    harvest_tree_apple_reward: float = 2.0
    harvest_stone_reward: float = 128.0
    pickup_item_reward: float = 1.0
    eat_apple_reward: float = 2.0
    health_regen_reward: float = 1.0
    attack_hit_reward: float = 32.0
    mob_kill_reward: float = 1024.0
    craft_plank_reward: float = 8.0
    craft_stick_reward: float = 16.0
    craft_wood_tool_reward: float = 64.0
    craft_stone_tool_reward: float = 512.0

    craft_anywhere: bool = True

    terrain_classes: int = 3
    block_classes: int = 4
    entity_classes: int = 4
    item_classes: int = 9
    action_size: int = 15
    tile_size: int = 48
    fps: int = 12

    @property
    def obs_size(self) -> int:
        grid_cells = self.view_size * self.view_size
        return grid_cells * (self.terrain_classes + self.block_classes + self.entity_classes) + 2 + self.item_classes
