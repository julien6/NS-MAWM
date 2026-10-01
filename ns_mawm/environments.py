from __future__ import annotations

import importlib.metadata
from pathlib import Path
import subprocess
import numpy as np
import torch
from .schema import BlockSpec, Schema
from .data import Episode

ROOT = Path(__file__).resolve().parents[1]


def commit(path):
    result = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True)
    if result.returncode:
        import json
        for root in (Path(path), ROOT):
            manifest = root / "benchmark_commits.json"
            if manifest.exists():
                pins = json.loads(manifest.read_text())
                key = {"Overcooked_AI":"overcooked", "Gridcraft":"gridcraft"}.get(Path(path).name, "ns_mawm")
                if key in pins:
                    return pins[key]
        raise RuntimeError(f"Cannot establish benchmark commit: {path}")
    return result.stdout.strip()


class SchemaBuilder:
    def __init__(self, agents):
        self.agents, self.blocks = agents, []
        self.offsets = [0] * agents

    def add(self, name, agent, categories=None, *, shared=False, tolerance=1e-5, unit="native"):
        width = len(categories) if categories else 1
        start = self.offsets[agent]
        self.blocks.append(BlockSpec(name, "shared" if shared else agent,
            "categorical" if categories else "scalar", tuple(categories) if categories else None,
            (start, start + width), unit, (0., 1.), tolerance, 0., agent))
        self.offsets[agent] += width

    def build(self):
        if len(set(self.offsets)) != 1:
            raise ValueError("All agent encodings must have the same width")
        return Schema(self.blocks, (self.offsets[0], self.agents))


class Gridcraft:
    action_size = 15
    def __init__(self, agents=2, max_steps=500, view_size=7, **kwargs):
        from vGridcraft.vgridcraft.config import VGridcraftConfig
        from vGridcraft.vgridcraft.env import VectorizedGridcraftEnv
        self.agents, self.view_size = agents, view_size
        self.env = VectorizedGridcraftEnv(1, config=VGridcraftConfig(num_agents=agents, max_steps=max_steps, view_size=view_size, **kwargs))
        builder = SchemaBuilder(agents)
        self.numeric = ("health", "hunger", "wood", "plank", "stick", "stone", "wood_sword", "stone_sword", "wood_pickaxe", "stone_pickaxe", "apple")
        for a in range(agents):
            for kind, categories in (("terrain", ("grass", "water", "dirt")), ("block", ("empty", "tree", "stone", "unused")), ("entity", ("none", "agent", "mob", "item"))):
                for y in range(view_size):
                    for x in range(view_size):
                        builder.add(f"agent{a}.{kind}[{y},{x}]", a, categories)
            for name in self.numeric:
                builder.add(f"agent{a}.{name}", a, unit="count")
        self.schema = builder.build()
        self.metadata = {"environment": "gridcraft", "commit": commit(ROOT), "upstream_commit": "fb00bb4f6229dfc1d1857b939f5cb42ee50096a2",
            "scenario": "survival", "wrapper_version": 2, "view_size": view_size,
            "semantics": {"source": "simulator_source", "text": "Agents move sequentially by index; occupied destinations block movement. Terrain is static. Crafting is anywhere, consumes one wood and produces two planks. Native inventory counts are not clipped."}}

    def encode(self, native):
        vector = native["vector"][0].T.clone()
        vector[-11:] = native["self"][0].T.float()
        return vector

    def reset(self, seed):
        return self.encode(self.env.reset(seed=seed))

    def step(self, actions):
        obs, reward, terminated, truncated, _ = self.env.step(torch.tensor([actions]))
        return self.encode(obs), float(reward.sum()), bool(terminated.item()), bool(truncated.item())

    def close(self):
        pass


class Overcooked:
    action_size = 6
    def __init__(self, agents=2, max_steps=400, layout="cramped_room", **kwargs):
        if agents != 2:
            raise ValueError("Pinned Overcooked encoding supports two players")
        if not hasattr(np, "Inf"):
            np.Inf = np.inf  # Pinned upstream still uses the NumPy 1.x spelling.
        if commit(ROOT / "Overcooked_AI") != "3935fd4d4362502b45e38e1a5a54bfdad06e32ed":
            raise RuntimeError("Overcooked checkout differs from the pinned benchmark")
        from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
        from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
        from overcooked_ai_py.mdp.actions import Action
        self.actions = Action.ALL_ACTIONS
        self.mdp = OvercookedGridworld.from_layout_name(layout)
        self.env = OvercookedEnv.from_mdp(self.mdp, horizon=max_steps)
        self.agents = 2
        self.pots = tuple(self.mdp.get_pot_locations())
        self.width, self.height = self.mdp.shape
        builder = SchemaBuilder(2)
        for a in range(2):
            for axis in ("x", "y", "dx", "dy"):
                builder.add(f"agent{a}.{axis}", a, unit="cell")
            builder.add(f"agent{a}.held", a, ("none", "onion", "tomato", "dish", "soup"))
            for x in range(self.width):
                for y in range(self.height):
                    p = f"agent{a}.cell[{x},{y}]"
                    builder.add(p + ".terrain", a, (" ", "X", "P", "O", "T", "D", "S"))
                    builder.add(p + ".object", a, ("none", "onion", "tomato", "dish", "soup"))
                    for k in ("onions", "tomatoes", "timer", "cooking", "ready"):
                        builder.add(p + "." + k, a, unit="count")
        self.schema = builder.build()
        self.metadata = {"environment": "overcooked", "commit": commit(ROOT / "Overcooked_AI"), "scenario": layout,
            "wrapper_version": 2, "semantics": {"source": "simulator_source", "text": "Fully observable public OvercookedState is encoded as player and object blocks. Interactions resolve in player order, then simultaneous movement (same destination and swaps block both), then cooking timers advance. Terrain is fixed."}}

    def encode(self, obs):
        # The environment's returned OvercookedState IS its public observation.
        values = {}
        for a, player in enumerate(obs.players):
            prefix = f"agent{a}."
            values.update({prefix + "x": player.position[0], prefix + "y": player.position[1],
                           prefix + "dx": player.orientation[0], prefix + "dy": player.orientation[1],
                           prefix + "held": player.get_object().name if player.has_object() else "none"})
            for x in range(self.width):
                for y in range(self.height):
                    p = prefix + f"cell[{x},{y}]"
                    obj = obs.objects.get((x, y))
                    values[p + ".terrain"] = self.mdp.terrain_mtx[y][x]
                    values[p + ".object"] = obj.name if obj else "none"
                    for k in ("onions", "tomatoes", "timer", "cooking", "ready"):
                        values[p + "." + k] = 0
                    if obj and obj.name == "soup":
                        values[p + ".onions"] = obj.ingredients.count("onion")
                        values[p + ".tomatoes"] = obj.ingredients.count("tomato")
                        values[p + ".timer"] = obj.cook_time_remaining
                        values[p + ".cooking"] = int(obj.is_cooking)
                        values[p + ".ready"] = int(obj.is_ready)
        return self.schema.encode(values)

    def reset(self, seed):
        np.random.seed(seed)
        self.env.reset()
        return self.encode(self.env.state)

    def step(self, actions):
        obs, reward, done, _ = self.env.step(tuple(self.actions[a] for a in actions))
        return self.encode(obs), float(reward), False, bool(done)

    def close(self):
        pass


class PredatorPrey:
    action_size = 5
    def __init__(self, agents=3, max_steps=100, landmarks=2, **kwargs):
        from .verification import benchmark_identity
        identity = benchmark_identity("mpe2", "147a8e88b669de9f0b06d1947c44d5c380c570bb", "1.1.1")
        if identity["status"] != "passed":
            raise RuntimeError("Install the pinned MPE2 source from requirements-ns-mawm.txt: " + str(identity))
        from mpe2 import simple_tag_v3
        self.env = simple_tag_v3.parallel_env(num_good=1, num_adversaries=agents,
            num_obstacles=landmarks, max_cycles=max_steps, continuous_actions=False, **kwargs)
        self.names = [f"adversary_{i}" for i in range(agents)]
        self.prey = "agent_0"
        self.agents, self.landmarks = agents, landmarks
        builder = SchemaBuilder(agents)
        # simple_tag observations: self velocity, position, landmarks, other
        # positions, good-agent velocity. No simulator state is consulted.
        for a in range(agents):
            names = ["vx", "vy", "x", "y"]
            names += [f"landmark{j}.{axis}" for j in range(landmarks) for axis in ("x", "y")]
            others = [i for i in range(agents) if i != a]
            names += [f"ally{j}.{axis}" for j in others for axis in ("x", "y")]
            names += ["prey.x", "prey.y", "prey.vx", "prey.vy"]
            for name in names:
                builder.add(f"agent{a}." + name, a, tolerance=1e-4, unit="world")
        self.schema = builder.build()
        self.metadata = {"environment": "predator_prey", "package_version": importlib.metadata.version("mpe2"),
            "commit": "147a8e88b669de9f0b06d1947c44d5c380c570bb", "scenario": "simple_tag_v3", "wrapper_version": 2,
            "prey_controller": "fixed_observation_away_v1", "semantics": {"source": "simulator_source",
            "text": "dt=.1, damping=.25, predator acceleration=3, mass=1, speed cap=1. Predator radius .075, prey .05, landmarks .2. Soft contact has a nonzero tail; exact rules abstain within .5 distance. Prey deterministically moves away from nearest observed predator."}}
        self.last_native = None

    def encode(self, obs):
        return torch.tensor(np.stack([obs[n] for n in self.names], axis=1), dtype=torch.float32)

    def reset(self, seed):
        self.last_native, _ = self.env.reset(seed=seed)
        return self.encode(self.last_native)

    def step(self, actions):
        prey_obs = self.last_native[self.prey]
        start = 4 + 2 * self.landmarks
        rel = prey_obs[start:start + 2 * self.agents].reshape(-1, 2)
        nearest = rel[np.argmin((rel ** 2).sum(1))]
        axis = int(np.argmax(np.abs(nearest)))
        prey_action = (1 if nearest[0] > 0 else 2) if axis == 0 else (3 if nearest[1] > 0 else 4)
        joint = dict(zip(self.names, map(int, actions)))
        joint[self.prey] = prey_action
        obs, rewards, term, trunc, _ = self.env.step(joint)
        self.last_native = obs
        return self.encode(obs), sum(rewards[n] for n in self.names), all(term.values()), all(trunc.values())

    def close(self):
        self.env.close()


class SMACv2:
    def __init__(self, agents=5, max_steps=120, map_name="10gen_terran", capability_config=None,
                 benchmark_commit="577ab5a2cff2391f8df582da5731ea9cd6adf3c6", **kwargs):
        from .verification import benchmark_identity
        identity = benchmark_identity("smacv2", benchmark_commit)
        if identity["status"] != "passed":
            raise RuntimeError("Install the pinned SMACv2 source from requirements-ns-mawm.txt: " + str(identity))
        from smacv2.env.starcraft2.wrapper import StarCraftCapabilityEnvWrapper
        from .smac import codec
        if not benchmark_commit or len(benchmark_commit) != 40:
            raise ValueError("SMACv2 requires its exact 40-character benchmark commit")
        capability = capability_config or {"n_units": agents, "n_enemies": agents,
            "team_gen": {"dist_type": "weighted_teams", "unit_types": ["marine"],
                         "weights": [1.0], "observe": True},
            "start_positions": {"dist_type": "surrounded_and_reflect", "p": .5, "map_x": 32, "map_y": 32}}
        if capability.get("team_gen", {}).get("unit_types") != ["marine"] or kwargs.get("step_mul", 8) != 8:
            raise ValueError("The bundled verified rule semantics require marine-only teams and step_mul=8; supply a new adapter/library for other races")
        if kwargs.get("fully_observable", False):
            raise ValueError("Hidden global-state observation is forbidden")
        self.constructor = StarCraftCapabilityEnvWrapper
        self.options = dict(map_name=map_name, capability_config=capability,
            obs_own_pos=True, obs_all_health=True, obs_timestep_number=False,
            use_unit_ranges=False, conic_fov=False, **kwargs)
        self.env = self.constructor(**self.options)
        info = self.env.get_env_info()
        self.agents, self.action_size = info["n_agents"], info["n_actions"]
        self.max_steps, self.steps = max_steps, 0
        self.schema, self.mappings = codec(self.env.get_obs_feature_names(), self.agents)
        self.metadata = {"environment": "smacv2", "commit": benchmark_commit, "scenario": map_name,
            "wrapper_version": 2, "semantics": {"source": "simulator_source", "text":
            "Public get_obs only. Missing unit types become the explicit unobserved category. Default scenario is Terran marines (no regeneration/healing), circular sight range 9; rules abstain unless continued visibility and survival follow from observed bounds."}}

    def _encode(self):
        from .smac import encode
        return encode(self.schema, self.mappings, self.env.get_obs())

    def reset(self, seed):
        # Upstream seed() is a getter. Reconstructing pins the game RNG as well
        # as the capability-distribution streams on each recorded episode.
        import random
        random.seed(seed)
        np.random.seed(seed)
        self.env.close()
        from copy import deepcopy
        self.env = self.constructor(**deepcopy(self.options), seed=seed)
        def seed_distribution(distribution, value):
            if hasattr(distribution, "rng"):
                distribution.rng = np.random.default_rng(value)
            # Composite start-position distributions contain child samplers.
            for index, child in enumerate(vars(distribution).values()):
                if child is not distribution and hasattr(child, "generate"):
                    seed_distribution(child, value + index + 1)
        for index, distribution in enumerate(self.env.env_key_to_distribution_map.values()):
            seed_distribution(distribution, seed + index)
        self.env.reset()
        self.steps = 0
        return self._encode()

    def available_actions(self):
        return self.env.get_avail_actions()

    def step(self, actions):
        reward, done, info = self.env.step(list(map(int, actions)))
        self.steps += 1
        truncated = bool(info.get("episode_limit", False)) or self.steps >= self.max_steps
        return self._encode(), float(reward), bool(done and not truncated), truncated

    def close(self):
        self.env.close()


ENVIRONMENTS = {"gridcraft": Gridcraft, "overcooked": Overcooked, "predator_prey": PredatorPrey, "smacv2": SMACv2}


def make_env(name, **kwargs):
    return ENVIRONMENTS[name](**kwargs)


def collect(env, episodes=1000, seed=0, policies=None):
    policies = policies or {"random": (1., None)}
    names = list(policies)
    weights = np.array([policies[n][0] for n in names], dtype=float)
    if (weights < 0).any() or not np.isclose(weights.sum(), 1):
        raise ValueError("Collection policy proportions must sum to one")
    rng = np.random.default_rng(seed)
    output = []
    for index in range(episodes):
        episode_seed = seed + index
        obs = env.reset(episode_seed)
        name = str(rng.choice(names, p=weights))
        policy = policies[name][1]
        if policy is not None and hasattr(policy, "reset"):
            policy.reset(episode_seed)
        observations, actions, rewards, terminated, truncated = [obs], [], [], [], []
        while True:
            if policy:
                action = policy(obs)
            elif hasattr(env, "available_actions"):
                action = [int(rng.choice(np.flatnonzero(mask))) for mask in env.available_actions()]
            else:
                action = rng.integers(env.action_size, size=env.agents).tolist()
            obs, reward, term, trunc = env.step(action)
            observations.append(obs)
            actions.append(action)
            rewards.append(reward)
            terminated.append(term)
            truncated.append(trunc)
            if term or trunc:
                break
        output.append(Episode(f"{env.metadata['environment']}-{episode_seed}", torch.stack(observations),
            torch.tensor(actions), torch.tensor(rewards), torch.tensor(terminated), torch.tensor(truncated), episode_seed, name))
    return output
