from __future__ import annotations

from copy import deepcopy
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .data import Dataset, Episode
from .models import WorldModel
from .rules import Context, PSWM
from .training import train, seed_all, episode_contexts
from .statistics import control_metrics


class LocalActor(nn.Module):
    def __init__(self, obs_size, actions, hidden=128):
        super().__init__()
        self.cell = nn.GRUCell(obs_size, hidden)
        self.output = nn.Linear(hidden, actions)
        self.hidden = hidden

    def step(self, obs, state=None):
        if state is None:
            state = obs.new_zeros(obs.shape[0], self.hidden)
        state = self.cell(obs, state)
        return self.output(state), state


class CentralCritic(nn.Module):
    def __init__(self, obs_size, agents, actions, hidden=128):
        super().__init__()
        self.agents, self.actions, self.hidden = agents, actions, hidden
        self.cell = nn.GRUCell(obs_size * agents, hidden)
        self.head = nn.Sequential(nn.Linear(hidden + agents * actions, hidden), nn.ReLU(), nn.Linear(hidden, agents))

    def step(self, observation, state=None):
        flat = observation.flatten(1)
        return self.cell(flat, flat.new_zeros(len(flat), self.hidden) if state is None else state)

    def values(self, state, actions):
        onehot = F.one_hot(actions.long(), self.actions).float().flatten(1)
        return self.head(torch.cat((state, onehot), -1))


class RecurrentMASAC:
    """Small discrete MASAC reference with local actors and centralized twin critics.

    Explicit prefixes also make this a testable bridge contract for BenchMARL
    sequence replay; imagined rows use the same update method as real rows.
    """
    def __init__(self, schema, actions, config):
        self.schema, self.actions, self.config = schema, actions, config
        self.agents = schema.shape[1]
        hidden = config.get("hidden", 128)
        self.actors = nn.ModuleList([LocalActor(schema.shape[0], actions, hidden) for _ in range(self.agents)])
        self.critics = nn.ModuleList([CentralCritic(schema.shape[0], self.agents, actions, hidden) for _ in range(2)])
        self.targets = deepcopy(self.critics).requires_grad_(False)
        self.actor_optim = torch.optim.Adam(self.actors.parameters(), lr=config.get("lr", 3e-4))
        self.critic_optim = torch.optim.Adam(self.critics.parameters(), lr=config.get("lr", 3e-4))
        self.alpha = float(config.get("alpha", .2))
        self.gamma = float(config.get("gamma", .99))
        self.tau = float(config.get("tau", .005))

    @torch.no_grad()
    def act(self, observation, states=None, deterministic=False, available=None):
        states = states or [None] * self.agents
        actions, next_states = [], []
        for i, actor in enumerate(self.actors):
            logits, state = actor.step(observation[:, i][None], states[i])
            if available is not None:
                logits = logits.masked_fill(~torch.tensor(available[i], dtype=torch.bool)[None], -torch.inf)
            action = logits.argmax(-1) if deterministic else torch.distributions.Categorical(logits=logits).sample()
            actions.append(int(action.item()))
            next_states.append(state)
        return actions, next_states

    def _states(self, prefix, observation):
        actor_states, critic_states, target_states = [None]*self.agents, [None]*2, [None]*2
        with torch.no_grad():
            for obs in prefix[-self.config.get("burn_in", 10):]:
                for i, actor in enumerate(self.actors):
                    _, actor_states[i] = actor.step(obs[:, i][None], actor_states[i])
                for j in range(2):
                    critic_states[j] = self.critics[j].step(obs[None], critic_states[j])
                    target_states[j] = self.targets[j].step(obs[None], target_states[j])
        logits = []
        for i, actor in enumerate(self.actors):
            out, actor_states[i] = actor.step(observation[:, i][None], actor_states[i])
            logits.append(out)
        critic_states = [critic.step(observation[None], state) for critic, state in zip(self.critics, critic_states)]
        with torch.no_grad():
            target_states = [critic.step(observation[None], state) for critic, state in zip(self.targets, target_states)]
        return logits, actor_states, critic_states, target_states

    def update(self, rows):
        self.critic_optim.zero_grad()
        self.actor_optim.zero_grad()
        critic_loss, actor_loss = 0., 0.
        for row in rows:
            obs, action, reward, next_obs, terminated, prefix = row
            logits, actor_states, critic_states, target_states = self._states(prefix, obs)
            action = torch.tensor(action)[None]
            with torch.no_grad():
                next_logits = [actor.step(next_obs[:, i][None], actor_states[i])[0] for i, actor in enumerate(self.actors)]
                distributions = [torch.distributions.Categorical(logits=l) for l in next_logits]
                next_actions = torch.stack([dist.sample() for dist in distributions], -1)
                logp = torch.stack([dist.log_prob(next_actions[:, i]) for i, dist in enumerate(distributions)], -1)
                next_states = [critic.step(next_obs[None], state) for critic, state in zip(self.targets, target_states)]
                next_q = torch.minimum(*[critic.values(state, next_actions) for critic, state in zip(self.targets, next_states)])
                target = float(reward) + self.gamma * (1 - float(terminated)) * (next_q - self.alpha * logp)
            critic_loss = critic_loss + sum(F.mse_loss(critic.values(state, action), target) for critic, state in zip(self.critics, critic_states)) / len(rows)
            # Exact expectation over each local action, holding other agents'
            # sampled actions fixed. Critic values are constants for actor updates.
            sampled = torch.stack([torch.distributions.Categorical(logits=l.detach()).sample() for l in logits], -1)
            for i in range(self.agents):
                q = []
                with torch.no_grad():
                    for a in range(self.actions):
                        joint = sampled.clone()
                        joint[:, i] = a
                        q.append(torch.minimum(*[critic.values(state.detach(), joint)[:, i] for critic, state in zip(self.critics, critic_states)]))
                values = torch.stack(q, -1)
                log_probs = logits[i].log_softmax(-1)
                actor_loss = actor_loss + (log_probs.exp() * (self.alpha * log_probs - values)).sum(-1).mean() / (len(rows) * self.agents)
        critic_loss.backward()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critics.parameters(), 10)
        torch.nn.utils.clip_grad_norm_(self.actors.parameters(), 10)
        self.critic_optim.step()
        self.actor_optim.step()
        with torch.no_grad():
            for target, current in zip(self.targets.parameters(), self.critics.parameters()):
                target.lerp_(current, self.tau)
        return {"critic_loss": float(critic_loss.detach()), "actor_loss": float(actor_loss.detach())}


class BenchMARLBridge:
    """Converts prefix-backed model branches to BenchMARL/TorchRL replay keys."""
    def __init__(self, model):
        self.model = model

    def imagine(self, prefix, observation, policy, horizon=5):
        from tensordict import TensorDict
        # policy sees one local observation per agent, never symbolic memory.
        predictions = self.model.rollout(prefix, observation, [], horizon=horizon, policy=policy)
        return predictions

    @staticmethod
    def transition(observation, actions, prediction, group="agents"):
        from tensordict import TensorDict
        agents = observation.shape[1]
        if prediction.reward is None:
            raise ValueError("Imagined replay requires trained reward/termination heads")
        done = torch.full((agents, 1), prediction.terminated, dtype=torch.bool)
        return TensorDict({group: TensorDict({"observation": observation.T,
                "action": torch.tensor(actions)[:, None]}, batch_size=[agents]),
            "next": TensorDict({group: TensorDict({"observation": prediction.output.T,
                "reward": torch.full((agents, 1), prediction.reward), "done": done, "terminated": done}, batch_size=[agents]),
                "done": torch.tensor([prediction.terminated]), "terminated": torch.tensor([prediction.terminated])}, batch_size=[])}, batch_size=[])


def control(env, dataset, schema, library, config, run=None):
    cfg = config.get("control", {})
    seed = config["run"].get("seed", 0)
    seed_all(seed)
    learner = RecurrentMASAC(schema, env.action_size, cfg)
    model_free = cfg.get("model_free", False)
    network = None
    real, imagined, episodes = [], [], []
    maximum = int(cfg.get("real_steps", 1000000))
    interval = int(cfg.get("checkpoint_interval", 25000))
    fit_interval = int(cfg.get("fit_interval", 1000))
    ratio = float(cfg.get("real_ratio", .5))
    if not 0 < ratio <= 1 or min(maximum, interval, fit_interval) < 1:
        raise ValueError("Invalid control budget or replay ratio")
    rng = np.random.default_rng(seed + 10000)
    eval_steps, eval_returns, eval_seconds = [], [], []
    started = time.perf_counter()
    obs = env.reset(seed + 20000)
    actor_states = None
    current_obs, current_actions, current_rewards, current_terms, current_truncs = [obs], [], [], [], []
    episode_seed = seed + 20000
    for step in range(1, maximum + 1):
        available = env.available_actions() if hasattr(env, "available_actions") else None
        action, actor_states = learner.act(obs, actor_states, available=available)
        next_obs, reward, term, trunc = env.step(action)
        real.append((obs, action, reward, next_obs, term, list(current_obs[:-1])))
        current_actions.append(action)
        current_rewards.append(reward)
        current_terms.append(term)
        current_truncs.append(trunc)
        current_obs.append(next_obs)
        obs = next_obs
        if term or trunc:
            episodes.append(Episode(f"control-{episode_seed}", torch.stack(current_obs), torch.tensor(current_actions),
                torch.tensor(current_rewards), torch.tensor(current_terms), torch.tensor(current_truncs), episode_seed, "masac"))
            episode_seed += 1
            obs = env.reset(episode_seed)
            actor_states = None
            current_obs, current_actions, current_rewards, current_terms, current_truncs = [obs], [], [], [], []
        if not model_free and episodes and (step % fit_interval == 0 or step == maximum):
            # A dedicated training-only view avoids moving control transitions
            # into any offline evaluation split.
            class RealTraining:
                def get(self, split, purpose="train"):
                    if split != "train":
                        raise PermissionError("Online fitting only reads real training episodes")
                    return episodes
            fit_config = deepcopy(config)
            fit_config["world_model"]["optim"]["updates"] = cfg.get("world_model_updates", 10)
            fit_config.setdefault("evaluation", {}).pop("validation_target", None)
            network, _ = train(RealTraining(), schema, library, env.action_size, fit_config, control=True, network=network)
            if network is None:
                raise ValueError("Control needs a neural model with reward/termination heads")
            model = WorldModel(network, PSWM(library), config["world_model"].get("strategy", "none"))
            source = episodes[int(rng.integers(len(episodes)))]
            root = int(rng.integers(len(source.actions)))
            burn = cfg.get("burn_in", 10)
            prefix = [(source.observations[t], source.actions[t].tolist()) for t in range(max(0, root-burn), root)]
            local_states = None
            for obs_prefix, _ in prefix:
                _, local_states = learner.act(obs_prefix, local_states)
            branch_actions = []
            def policy(decoded):
                nonlocal local_states
                a, local_states = learner.act(schema.encode(decoded), local_states)
                branch_actions.append(a)
                return a
            predictions = model.rollout(prefix, source.observations[root], [], horizon=cfg.get("rollout_cap", 5), policy=policy)
            previous = source.observations[root]
            history = list(source.observations[max(0, root-burn):root])
            for a, pred in zip(branch_actions, predictions):
                imagined.append((previous, a, pred.reward, pred.output, pred.terminated, list(history)))
                history.append(previous)
                previous = schema.encode(schema.decode(pred.output))
            imagined = imagined[-cfg.get("imagined_capacity", 100000):]
        if len(real) >= cfg.get("warmup", 32):
            for _ in range(cfg.get("policy_updates", 1)):
                batch = cfg.get("batch_size", 32)
                real_count = batch if not imagined or model_free else max(1, round(batch * ratio))
                rows = [real[int(rng.integers(len(real)))] for _ in range(real_count)]
                rows += [imagined[int(rng.integers(len(imagined)))] for _ in range(batch-real_count)]
                losses = learner.update(rows)
        real = real[-cfg.get("real_capacity", 100000):]
        if step % interval == 0 or step == maximum:
            from .environments import make_env
            evaluation_env = make_env(config["run"]["env"], **config.get("environment", {}))
            returns = []
            try:
                for i in range(cfg.get("eval_episodes", 5)):
                    eval_obs = evaluation_env.reset(seed + 100000 + i)
                    states, total = None, 0.
                    while True:
                        avail = evaluation_env.available_actions() if hasattr(evaluation_env, "available_actions") else None
                        a, states = learner.act(eval_obs, states, deterministic=True, available=avail)
                        eval_obs, r, d, t = evaluation_env.step(a)
                        total += r
                        if d or t:
                            break
                    returns.append(total)
            finally:
                evaluation_env.close()
            value, elapsed = float(np.mean(returns)), time.perf_counter()-started
            eval_steps.append(step)
            eval_returns.append(value)
            eval_seconds.append(elapsed)
            if run:
                run.record("return", value, step, unit="reward")
                run.record("wall_seconds", elapsed, step, unit="seconds")
                torch.save({"actors": learner.actors.state_dict(), "critics": learner.critics.state_dict(),
                    "targets": learner.targets.state_dict(), "real_steps": step, "config": config}, run.path / f"policy-{step}.pt")
    result = {"real_steps": maximum, "imagined_replay": len(imagined), "checkpoints": eval_steps,
              "returns": eval_returns, "seconds": eval_seconds, "learner": "recurrent_masac_reference"}
    if len(eval_steps) >= 2:
        result.update(control_metrics(eval_steps, eval_returns, cfg.get("return_threshold", float('inf'))))
    if run:
        for key in ("final_return", "normalized_auc", "interactions_to_threshold"):
            if key in result:
                run.record(key, result[key])
    return result
