"""Collection policies use public encoded observations only."""
from __future__ import annotations
import numpy as np
import torch
from .control import RecurrentMASAC


class ScriptedPolicy:
    def __init__(self, env, seed=0):
        self.env, self.rng = env, np.random.default_rng(seed)
    def reset(self, seed):
        self.rng = np.random.default_rng(seed)
    def __call__(self, observation):
        o = self.env.schema.decode(observation)
        name = self.env.metadata["environment"]
        actions = []
        for i in range(self.env.agents):
            p = f"agent{i}."
            if name == "gridcraft":
                if o[p+"wood"] > 0:
                    action = 9
                elif any(v == "tree" for k,v in o.items() if k.startswith(p+"block[")):
                    action = 5 if self.rng.random() < .5 else int(self.rng.integers(1,5))
                else:
                    action = int(self.rng.integers(self.env.action_size))
            elif name == "predator_prey":
                x,y = o[p+"prey.x"],o[p+"prey.y"]
                action = (2 if x>0 else 1) if abs(x)>abs(y) else (4 if y>0 else 3)
            elif name == "overcooked":
                action = 5 if self.rng.random() < .5 else int(self.rng.integers(5))
            else:
                mask = self.env.available_actions()[i]
                attacks = [j for j,valid in enumerate(mask) if j>=6 and valid]
                action = attacks[0] if attacks else int(self.rng.choice(np.flatnonzero(mask)))
            actions.append(action)
        return actions


class CheckpointPolicy:
    def __init__(self, env, checkpoint):
        saved = torch.load(checkpoint, weights_only=True, map_location="cpu")
        self.external = None
        if saved.get("format") == "ns_mawm_benchmarl_policy_v1":
            from .benchmarl import run_control
            from .libraries import make_library
            from pathlib import Path
            from types import SimpleNamespace
            import tempfile
            from .schema import Schema
            self.temporary = tempfile.TemporaryDirectory(prefix="ns-mawm-policy-")
            config = saved["configuration"]
            self.external = run_control(env, None, Schema.from_dict(saved["schema"]), make_library(env), config,
                                        SimpleNamespace(path=Path(self.temporary.name)), build_only=True)
            self.policy = self.external.algorithm.get_policy_for_collection()
            self.policy.load_state_dict(saved["policy"])
            self.env = env
            self.states = None
            return
        self.env = env
        self.learner = RecurrentMASAC(env.schema, env.action_size, saved["config"].get("control", {}))
        self.learner.actors.load_state_dict(saved["actors"])
        self.states = None
    def reset(self, seed):
        torch.manual_seed(seed)
        self.states = None
    def __call__(self, observation):
        if self.external is not None:
            from torchrl.envs.utils import step_mdp
            if self.states is None:
                self.states = self.external.test_env.reset()
            self.states["agents", "observation"] = observation.T[None].to(self.states.device)
            if hasattr(self.env, "available_actions"):
                self.states["agents", "action_mask"] = torch.as_tensor(self.env.available_actions(), dtype=torch.bool, device=self.states.device)[None]
            with torch.no_grad():
                self.policy(self.states)
            actions = self.states["agents", "action"].reshape(-1).cpu().tolist()
            for key in list(self.states.get("next", {}).keys(True, True)) if "next" in self.states.keys() else []:
                if "hidden" in str(key) or "cell" in str(key):
                    self.states[key] = self.states[("next", *key)] if isinstance(key,tuple) else self.states["next",key]
            if "is_init" in self.states.keys(): self.states["is_init"].zero_()
            return actions
        masks = self.env.available_actions() if hasattr(self.env, "available_actions") else None
        actions,self.states = self.learner.act(observation,self.states, available=masks)
        return actions

    def close(self):
        if self.external is not None:
            self.external.test_env.close()
            self.external.collector.shutdown()
            self.temporary.cleanup()


def collection_policies(env, config):
    proportions = config.get("policies", {"random": 1.})
    policies = {}
    for name, fraction in proportions.items():
        if name == "random":
            policy = None
        elif name == "scripted":
            policy = ScriptedPolicy(env)
        elif name == "partially_trained_masac":
            policy = CheckpointPolicy(env, config["policy_checkpoint"])
        else:
            raise ValueError(f"Unknown collection policy: {name}")
        policies[name] = (float(fraction),policy)
    return policies
