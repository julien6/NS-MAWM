"""Native BenchMARL MASAC/MAMBPO bridge using the shared semantic model."""
from __future__ import annotations
from copy import deepcopy
import sys
import time
import torch
from torch import nn
from .environments import ROOT, make_env

# This repository vendors its BenchMARL integration.
if str(ROOT / "BenchMARL") not in sys.path:
    sys.path.insert(0, str(ROOT / "BenchMARL"))
from tensordict import TensorDict
from torchrl.data import Composite, Categorical, Unbounded
from torchrl.envs import EnvBase
from benchmarl.environments.common import TaskClass
from benchmarl.experiment.callback import Callback
from .data import Episode
from .models import WorldModel
from .rules import Context, PSWM
from .training import train


class SemanticEnv(EnvBase):
    def __init__(self, name, options, num_envs=1, seed=0, device="cpu"):
        self.adapters = [make_env(name, **options) for _ in range(num_envs)]
        first = self.adapters[0]
        self.agent_count, self.features = first.schema.shape[1], first.schema.shape[0]
        self.group_map = {"agents": [f"agent_{i}" for i in range(self.agent_count)]}
        super().__init__(device=device, batch_size=[num_envs])
        self._seed = (seed or 0) + 20000
        self._observations = [None] * num_envs
        shape = (*self.batch_size, self.agent_count)
        self.observation_spec = Composite(agents=Composite(
            observation=Unbounded(shape=(*shape, self.features), device=self.device),
            shape=shape, device=self.device), shape=self.batch_size, device=self.device)
        if hasattr(first, "available_actions"):
            self.observation_spec["agents", "action_mask"] = Unbounded(shape=(*shape, first.action_size), dtype=torch.bool, device=self.device)
        self.action_spec = Composite(agents=Composite(
            action=Categorical(first.action_size, shape=shape, device=self.device),
            shape=shape, device=self.device), shape=self.batch_size, device=self.device)
        self.reward_spec = Composite(agents=Composite(reward=Unbounded(shape=(*shape, 1), device=self.device),
            shape=shape, device=self.device), shape=self.batch_size, device=self.device)
        self.done_spec = Composite(**{k: Unbounded(shape=(*self.batch_size, 1), dtype=torch.bool, device=self.device)
            for k in ("done", "terminated", "truncated")}, shape=self.batch_size, device=self.device)

    def _set_seed(self, seed):
        self._seed = seed + 20000
        return seed

    def _reset(self, tensordict=None):
        reset = torch.ones(self.batch_size, dtype=torch.bool) if tensordict is None else tensordict.get("_reset", torch.ones((*self.batch_size, 1), dtype=torch.bool)).reshape(*self.batch_size)
        for i, adapter in enumerate(self.adapters):
            if reset[i] or self._observations[i] is None:
                self._observations[i] = adapter.reset(self._seed)
                self._seed += 1
        return self._td()

    def _td(self, rewards=None, term=None, trunc=None):
        observations = torch.stack(self._observations).transpose(-2, -1).to(self.device)
        agents = {"observation": observations}
        if hasattr(self.adapters[0], "available_actions"):
            agents["action_mask"] = torch.tensor([e.available_actions() for e in self.adapters], dtype=torch.bool, device=self.device)
        if rewards is not None:
            agents["reward"] = torch.tensor(rewards, device=self.device)[:, None, None].expand(-1, self.agent_count, 1).float()
        terminated = torch.tensor(term or [False]*len(self.adapters), device=self.device)[:, None]
        truncated = torch.tensor(trunc or [False]*len(self.adapters), device=self.device)[:, None]
        return TensorDict({"agents": TensorDict(agents, batch_size=[len(self.adapters), self.agent_count]),
            "done": terminated | truncated, "terminated": terminated, "truncated": truncated}, batch_size=self.batch_size, device=self.device)

    def _step(self, td):
        actions = td["agents", "action"].reshape(len(self.adapters), self.agent_count)
        rewards, terminated, truncated = [], [], []
        for i, adapter in enumerate(self.adapters):
            observation, reward, term, trunc = adapter.step(actions[i].cpu().tolist())
            self._observations[i] = observation
            rewards.append(reward)
            terminated.append(term)
            truncated.append(trunc)
        return self._td(rewards, terminated, truncated)

    def close(self, **kwargs):
        for adapter in self.adapters:
            adapter.close()
        super().close(**kwargs)


class SemanticTask(TaskClass):
    def get_env_fun(self, num_envs, continuous_actions, seed, device):
        return lambda: SemanticEnv(self.config["name"], self.config["options"], num_envs, seed, device)
    def supports_continuous_actions(self): return False
    def supports_discrete_actions(self): return True
    def max_steps(self, env): return self.config["options"].get("max_steps", 500)
    def has_render(self, env): return False
    def group_map(self, env): return env.group_map
    def observation_spec(self, env):
        spec = env.full_observation_spec_unbatched.clone()
        if ("agents", "action_mask") in spec.keys(True, True):
            del spec["agents", "action_mask"]
        return spec
    def info_spec(self, env): return None
    def state_spec(self, env): return None
    def action_spec(self, env): return env.full_action_spec_unbatched
    def action_mask_spec(self, env):
        spec = env.full_observation_spec_unbatched.clone()
        if ("agents", "action_mask") not in spec.keys(True, True):
            return None
        del spec["agents", "observation"]
        return spec
    @staticmethod
    def env_name(): return "ns_mawm"


class ModelReplay:
    def __init__(self, schema, library, action_size, config):
        self.schema, self.library, self.action_size, self.config = schema, library, action_size, config
        self.network = None
        self.episodes, self.partial = [], {}
        self.real_frames = 0

    def __getstate__(self):
        state = dict(self.__dict__)
        library = state.pop("library")
        state["library_record"] = {"manifest": library.manifest(),
            "rules": [{k: (sorted(v) if isinstance(v, frozenset) else v) for k,v in vars(r).items()
                       if k not in ("guard", "effect", "derives")} for r in library.rules]}
        return state

    def __setstate__(self, state):
        from .libraries import builtin_library
        from .rules import Library, compile_rule
        record = state.pop("library_record")
        self.__dict__.update(state)
        native = {r.id:r for r in builtin_library(self.schema).rules}
        rules = [native[r["id"]] if r["source"] == "handcrafted" else compile_rule(r) for r in record["rules"]]
        m = record["manifest"]
        self.library = Library(rules, self.schema, edges=m["edges"], policy=m["policy"],
                               parent=m["parent"], creator=m["creator"], rule_filter=m["filter"])

    def process(self, algorithm, group, batch):
        # Collection batches are [environment, time]. Preserve episode order
        # before any replay flattening and carry unfinished episodes across calls.
        if batch.ndim != 2:
            raise ValueError("NS-MAWM needs an environment/time collection batch")
        self.real_frames += batch.numel()
        for i in range(batch.shape[0]):
            current = self.partial.setdefault(i, [])
            for t in range(batch.shape[1]):
                row = batch[i, t].clone()
                current.append(row)
                if row["next", "done"].any():
                    obs = [r[group, "observation"].cpu().T for r in current]
                    obs.append(current[-1]["next", group, "observation"].cpu().T)
                    self.episodes.append(Episode(f"benchmarl-{len(self.episodes)}", torch.stack(obs),
                        torch.stack([r[group, "action"].long().cpu().reshape(-1) for r in current]),
                        torch.stack([r["next", group, "reward"].mean().cpu() for r in current]),
                        torch.stack([r["next", "terminated"].any().cpu() for r in current]),
                        torch.stack([r["next", "truncated"].any().cpu() for r in current]), len(self.episodes), "masac"))
                    current.clear()
        if not self.episodes:
            return batch
        config = deepcopy(self.config)
        cfg = config.get("control", {})
        config["world_model"]["optim"]["updates"] = cfg.get("world_model_updates", 10)
        config.setdefault("evaluation", {}).pop("validation_target", None)
        episodes = self.episodes[-cfg.get("model_episode_capacity", 1000):]
        class RealOnly:
            def get(self, split, purpose="train"):
                if split != "train": raise PermissionError("Online model fitting is real-only")
                return episodes
        self.network, timing = train(RealOnly(), self.schema, self.library, self.action_size, config, control=True, network=self.network)
        model = WorldModel(self.network, PSWM(self.library), config["world_model"].get("strategy", "none"))
        policy = algorithm.get_policy_for_collection()
        branches = []
        cap = min(cfg.get("rollout_cap", 5), batch.shape[1])
        # Real prefixes warm both world-model memory and the local actors. The
        # collected policy hidden fields in the root contain the real prefix.
        for i in range(min(batch.shape[0], cfg.get("imagined_branches", 8))):
            root_index = max(0, batch.shape[1] - cap)
            # Never warm through a real episode reset.
            begin = root_index
            while begin > 0 and root_index-begin < cfg.get("burn_in", 10) and not batch[i, begin-1]["next", "done"].any():
                begin -= 1
            prefix = [(batch[i,t][group, "observation"].cpu().T, batch[i,t][group, "action"].cpu().reshape(-1).tolist()) for t in range(begin, root_index)]
            state, memory, history, previous = self.network.init_state(), {}, [], None
            for t, (obs, action) in enumerate(prefix):
                c = Context(self.schema.decode(obs), tuple(action), tuple(history), previous, memory, "real", t)
                p = model.step(state, c, diagnostics=True)
                state, memory, previous = p.state, p.symbolic.memory, tuple(action)
                history.append(c.observation)
            td = batch[i, root_index].clone()
            obs = td[group, "observation"].cpu().T
            branch = []
            for t in range(cap):
                with torch.no_grad():
                    policy_batch = td.unsqueeze(0)
                    policy(policy_batch)
                    td = policy_batch[0]
                action = td[group, "action"].long().reshape(-1).tolist()
                c = Context(self.schema.decode(obs), tuple(action), tuple(history[-50:]), previous, memory, "imagined", len(prefix)+t)
                p = model.step(state, c, diagnostics=True)
                row = td.clone()
                next_observation = self.schema.encode(self.schema.decode(p.output))
                row["next", group, "observation"] = next_observation.T.to(row.device)
                if (group, "action_mask") in row.keys(True, True):
                    from .smac import available_from_observation
                    row["next", group, "action_mask"] = torch.tensor(available_from_observation(self.schema,next_observation,self.action_size), device=row.device)
                row["next", group, "reward"] = torch.full_like(row["next", group, "reward"], p.reward)
                for key in (("next", "done"), ("next", "terminated"), ("next", group, "done"), ("next", group, "terminated")):
                    if key in row.keys(True, True):
                        row[key] = torch.full_like(row[key], p.terminated)
                if ("next", "truncated") in row.keys(True, True):
                    row["next", "truncated"] = torch.zeros_like(row["next", "truncated"])
                branch.append(row)
                if p.terminated:
                    break
                # Advance all actor recurrent fields from next to current.
                for key in list(td["next"].keys(True, True)):
                    if key in td.keys(True, True) and ("hidden" in str(key) or "is_init" in str(key)):
                        td[key] = row[("next", *key)] if isinstance(key, tuple) else row["next", key]
                if "is_init" in td.keys():
                    td["is_init"] = torch.zeros_like(td["is_init"])
                obs = self.schema.encode(self.schema.decode(p.output))
                td[group, "observation"] = obs.T.to(td.device)
                if (group, "action_mask") in td.keys(True, True):
                    td[group, "action_mask"] = row["next", group, "action_mask"]
                history.append(c.observation)
                state, memory, previous = p.state, p.symbolic.memory, tuple(action)
            if branch:
                # A variable terminal length is represented as its own replay
                # sequence; buffer insertion below groups identical lengths.
                branches.append(torch.stack(branch))
        if branches:
            common = min(len(b) for b in branches)
            fake = torch.stack([b[:common] for b in branches])
            # A new storage preserves genuine variable terminal lengths.
            algorithm._model_replay_buffers.pop(group, None)
            buffer = algorithm._get_model_replay_buffer(group)
            buffer.extend(fake.to(algorithm.buffer_device))
        algorithm.latest_metrics.update({"ns_mawm/real_frames": torch.tensor(float(self.real_frames)),
                                         "ns_mawm/model_updates": torch.tensor(float(timing["updates"]))})
        return batch


class Logging(Callback):
    def __init__(self, bridge, run):
        super().__init__()
        self.bridge, self.run = bridge, run
        self.started = time.perf_counter()
        self.values = []
    def on_setup(self):
        if self.bridge is not None:
            self.experiment.algorithm.ns_mawm_bridge = self.bridge
    def on_evaluation_end(self, rollouts):
        values = [float(r["next", "agents", "reward"].mean(-2).sum()) for r in rollouts]
        step = self.experiment.total_frames
        value = sum(values)/len(values)
        self.values.append((step, value))
        self.run.record("return", value, step, unit="reward")
        self.run.record("wall_seconds", time.perf_counter()-self.started, step, unit="seconds")
    def on_state_dict(self, state_dict):
        if self.bridge and self.bridge.network:
            state_dict["ns_mawm_network"] = self.bridge.network.state_dict()
            state_dict["ns_mawm_library_hash"] = self.bridge.library.hash


def run_control(env, dataset, schema, library, config, run, *, build_only=False):
    from benchmarl.algorithms import MambpoConfig, MasacConfig
    from benchmarl.experiment import Experiment, ExperimentConfig
    from benchmarl.models.lstm import LstmConfig
    from .statistics import control_metrics
    cfg = config.get("control", {})
    algorithm = (MasacConfig if cfg.get("model_free", False) else MambpoConfig).get_from_yaml()
    algorithm.alpha_init = cfg.get("alpha", .2)
    algorithm.fixed_alpha = True
    if not cfg.get("model_free", False):
        algorithm.imagined_rollouts.real_ratio = cfg.get("real_ratio", .5)
        algorithm.imagined_rollouts.rollout_length = cfg.get("rollout_cap", 5)
        algorithm.imagined_rollouts.model_buffer_size = cfg.get("imagined_capacity", 1000)
    experiment_config = ExperimentConfig.get_from_yaml()
    for name in ("sampling_device", "train_device", "buffer_device"):
        setattr(experiment_config, name, config.get("device", "cpu"))
    frames = cfg.get("frames_per_batch", cfg.get("fit_interval", 1000))
    experiment_config.max_n_frames = cfg.get("real_steps", 1000000)
    experiment_config.max_n_iters = None
    experiment_config.off_policy_collected_frames_per_batch = frames
    experiment_config.off_policy_n_envs_per_worker = cfg.get("num_envs", 1)
    experiment_config.off_policy_train_batch_size = cfg.get("batch_size", 32)
    experiment_config.off_policy_n_optimizer_steps = cfg.get("policy_updates", 1)
    experiment_config.off_policy_memory_size = cfg.get("real_capacity", 100000)
    experiment_config.gamma = cfg.get("gamma", .99)
    experiment_config.polyak_tau = cfg.get("tau", .005)
    experiment_config.lr = cfg.get("lr", 3e-4)
    experiment_config.prefer_continuous_actions = False
    experiment_config.evaluation_interval = cfg.get("checkpoint_interval", 25000)
    experiment_config.evaluation_episodes = cfg.get("eval_episodes", 5)
    experiment_config.render = False
    experiment_config.loggers = ["csv"]
    experiment_config.create_json = False
    (run.path / "benchmarl").mkdir(parents=True, exist_ok=True)
    experiment_config.save_folder = str(run.path.resolve() / "benchmarl")
    experiment_config.checkpoint_at_end = True
    experiment_config.checkpoint_interval = cfg.get("checkpoint_interval", 25000)
    hidden = cfg.get("hidden", 128)
    architecture = LstmConfig(hidden_size=hidden, n_layers=1, bias=True, dropout=0., compile=False,
        mlp_num_cells=[hidden], mlp_layer_class=nn.Linear, mlp_activation_class=nn.ReLU)
    bridge = None if cfg.get("model_free", False) else ModelReplay(schema, library, env.action_size, config)
    logger = Logging(bridge, run)
    experiment = Experiment(task=SemanticTask("semantic", {"name": config["run"]["env"], "options": config.get("environment", {})}),
        algorithm_config=algorithm, model_config=architecture, critic_model_config=deepcopy(architecture),
        seed=config["run"].get("seed", 0), config=experiment_config, callbacks=[logger])
    if build_only:
        return experiment
    experiment.run()
    torch.save({"format": "ns_mawm_benchmarl_policy_v1", "configuration": config, "schema": schema.to_dict(),
                "policy": experiment.algorithm.get_policy_for_collection().state_dict()}, run.path / "collection-policy.pt")
    result = {"learner": "BenchMARL_MASAC" if bridge is None else "BenchMARL_MAMBPO", "real_steps": experiment.total_frames,
              "checkpoints": logger.values}
    if len(logger.values) > 1:
        result.update(control_metrics([s for s,v in logger.values], [v for s,v in logger.values], cfg.get("return_threshold", float('inf'))))
    for metric in ("final_return", "normalized_auc", "interactions_to_threshold"):
        if metric in result:
            run.record(metric, result[metric])
    return result
