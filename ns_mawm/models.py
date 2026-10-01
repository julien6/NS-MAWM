from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from .schema import Schema
from .rules import Context, PSWM, PSWMOutput
from .extensions import STRATEGY_PLUGINS


class Backbone(Protocol):
    def init_state(self, prefix=()): ...
    def step(self, state, obs: Tensor, joint_action: Tensor, extra=None): ...


def mlp(inputs, sizes, outputs):
    layers = []
    for size in sizes:
        layers.extend([nn.Linear(inputs, size), nn.ReLU()])
        inputs = size
    return nn.Sequential(*layers, nn.Linear(inputs, outputs))


class JOPM(nn.Module):
    def __init__(self, schema: Schema, action_size: int, *, enc=(256, 256), lstm=256,
                 g=(256, 256), dec=(256, 256), residual=False, control=False):
        super().__init__()
        self.schema, self.action_size = schema, action_size
        self.width = schema.shape[0] * schema.shape[1]
        self.hidden_size, self.residual, self.control = lstm, residual, control
        self.encoder = mlp(self.width * (3 if residual else 1), enc, lstm)
        self.recurrent = nn.LSTMCell(lstm + action_size * schema.shape[1], lstm)
        self.predictor = mlp(lstm, g, lstm)
        self.decoder = mlp(lstm, dec, self.width)
        if control:
            self.reward_head = mlp(lstm, (lstm,), 1)
            self.done_head = mlp(lstm, (lstm,), 1)

    def init_state(self, prefix=(), batch_size=1):
        p = next(self.parameters())
        state = (p.new_zeros(batch_size, self.hidden_size), p.new_zeros(batch_size, self.hidden_size))
        for obs, action, extra in prefix:
            _, state = self.step(state, obs, action, extra)
        return state

    def inputs(self, obs, extra=None):
        x = obs.flatten(-2)
        if self.residual:
            if extra is None:
                raise ValueError("Residual requires both mask and zero-padded target")
            mask, target = extra
            x = torch.cat((x, mask.flatten(-2), target.flatten(-2)), -1)
        return x

    def step(self, state, obs, joint_action, extra=None):
        x = self.encoder(self.inputs(obs, extra))
        a = F.one_hot(joint_action.long(), self.action_size).float().flatten(-2)
        state = self.recurrent(torch.cat((x, a), -1), state)
        output = self.decoder(self.predictor(state[0])).reshape(*obs.shape[:-2], *self.schema.shape)
        return output, state

    def heads(self, state):
        if not self.control:
            raise RuntimeError("Reward/termination heads require control=True")
        return self.reward_head(state[0]).squeeze(-1), self.done_head(state[0]).squeeze(-1)


class DreamerWM(JOPM):
    """Centralized Gaussian RSSM-style baseline, with schema-level decoding.

    A compact architectural baseline, not a reproduction of DreamerV3's full
    policy learner or its published hyperparameters.
    """
    def __init__(self, schema, action_size, **kwargs):
        super().__init__(schema, action_size, **kwargs)
        h = self.hidden_size
        self.posterior = mlp(2*h, (h,), 2*h)
        self.prior = mlp(h, (h,), 2*h)
        self.recurrent = nn.GRUCell(h + action_size*schema.shape[1], h)
        self.auxiliary_loss = None

    def init_state(self, prefix=(), batch_size=1):
        p = next(self.parameters())
        state = (p.new_zeros(batch_size, self.hidden_size), p.new_zeros(batch_size, self.hidden_size), False)
        for obs, action, extra in prefix:
            _, state = self.step(state, obs, action, extra)
        return state

    def imagine_step(self, state, obs, joint_action, extra=None):
        if not state[2]:
            return self.step(state, obs, joint_action, extra)
        action = F.one_hot(joint_action.long(), self.action_size).float().flatten(-2)
        hidden = self.recurrent(torch.cat((state[1], action), -1), state[0])
        distribution = self.distribution(self.prior(hidden))
        latent = distribution.rsample() if self.training else distribution.mean
        output = self.decoder(self.predictor(latent)).reshape(*obs.shape[:-2], *self.schema.shape)
        return output, (hidden, latent, True)

    def distribution(self, parameters):
        mean, log_scale = parameters.chunk(2, -1)
        return torch.distributions.Normal(mean, log_scale.clamp(-5, 2).exp())

    def step(self, state, obs, joint_action, extra=None):
        encoded = self.encoder(self.inputs(obs, extra))
        prior = self.distribution(self.prior(state[0]))
        posterior = self.distribution(self.posterior(torch.cat((state[0], encoded), -1)))
        latent = posterior.rsample() if self.training else posterior.mean
        self.auxiliary_loss = torch.distributions.kl_divergence(posterior, prior).mean()
        action = F.one_hot(joint_action.long(), self.action_size).float().flatten(-2)
        hidden = self.recurrent(torch.cat((latent, action), -1), state[0])
        next_distribution = self.distribution(self.prior(hidden))
        next_latent = next_distribution.rsample() if self.training else next_distribution.mean
        output = self.decoder(self.predictor(next_latent)).reshape(*obs.shape[:-2], *self.schema.shape)
        return output, (hidden, next_latent, True)


from .mamba import MambaWM

BACKBONES = {"jopm_lstm": JOPM, "mamba_wm": MambaWM, "dreamer_central": DreamerWM}
STRATEGIES = {"none", "projection", "residual", "regularization", "feature_weighting"}


def block_mean(loss, weights=None):
    if weights is None:
        return loss.mean()
    # Normalize separately per context, including fully covered residual cases.
    return ((loss * weights).sum(-1) / weights.sum(-1).clamp_min(1)).mean()


def observation_loss(schema, logits, target, symbolic, strategy, coefficient=1.):
    if strategy in STRATEGY_PLUGINS:
        return STRATEGY_PLUGINS[strategy]["loss"](schema, logits, target, symbolic, coefficient)
    if strategy not in STRATEGIES or coefficient < 0:
        raise ValueError("Invalid integration strategy or lambda")
    data = schema.losses(logits, target, training=True)
    mask, symbolic_target = symbolic
    blocks = schema.block_mask(mask).detach()
    if strategy == "residual":
        return block_mean(data, 1 - blocks)
    loss = block_mean(data)
    if strategy in ("regularization", "feature_weighting") and coefficient:
        constraint = (data if strategy == "feature_weighting" else
                      schema.losses(logits, symbolic_target.detach(), training=True))
        loss = loss + coefficient * block_mean(constraint, blocks)
    return loss


@dataclass
class Prediction:
    output: Tensor
    pre: Tensor
    state: object
    symbolic: PSWMOutput | None
    reward: float | None = None
    terminated: bool = False


class WorldModel:
    def __init__(self, network, engine: PSWM, strategy="none", fallback=None):
        if strategy not in STRATEGIES | {"pswm_only", "llm_code"} | STRATEGY_PLUGINS.keys():
            raise ValueError("Unknown strategy")
        self.network, self.engine, self.strategy = network, engine, strategy
        self.schema, self.fallback = engine.schema, fallback

    @torch.no_grad()
    def step(self, state, context: Context, diagnostics=False):
        obs = self.schema.encode(context.observation)
        symbolic = self.engine(context) if diagnostics or self.strategy in ("projection", "residual", "pswm_only", "llm_code") or self.strategy in STRATEGY_PLUGINS else None
        if self.strategy in ("pswm_only", "llm_code"):
            if self.strategy == "llm_code" and not symbolic.mask.all():
                raise ValueError("Full code world models must assign every block")
            base = obs if self.fallback is None else self.fallback
            out = torch.where(symbolic.mask.bool(), symbolic.target, base)
            return Prediction(out, out, state, symbolic)
        device = next(self.network.parameters()).device
        extra = None if symbolic is None else (symbolic.mask[None].to(device), symbolic.target[None].to(device))
        step_function = self.network.step
        if context.origin == "imagined" and self.strategy in {"none", "regularization", "feature_weighting"} and hasattr(self.network, "imagine_step"):
            step_function = self.network.imagine_step
        logits, state = step_function(state, obs[None].to(device), torch.tensor([context.joint_action], device=device), extra)
        pre = self.schema.probabilities(logits)[0].cpu()
        out = pre
        if self.strategy in ("projection", "residual"):
            out = torch.where(symbolic.mask.bool(), symbolic.target, pre)
        if self.strategy in STRATEGY_PLUGINS:
            out = STRATEGY_PLUGINS[self.strategy]["predict"](pre, symbolic)
        reward, terminated = None, False
        if getattr(self.network, "control", False):
            r, d = self.network.heads(state)
            reward, terminated = float(r.item()), bool(d.sigmoid().item() >= .5)
        return Prediction(out, pre, state, symbolic, reward, terminated)

    def rollout(self, prefix, observation, actions, *, horizon=25, policy=None):
        state = self.network.init_state() if self.network is not None else None
        memory, history, previous = {}, [], None
        for t, (obs, action) in enumerate(prefix):
            c = Context(self.schema.decode(obs), tuple(action), tuple(history), previous, memory, "real", t)
            # Warm-up always reconstructs symbolic memory from the same prefix.
            p = self.step(state, c, diagnostics=True)
            state, memory, previous = p.state, p.symbolic.memory, tuple(action)
            history = (history + [c.observation])[-50:]
        current = self.schema.decode(observation)
        result = []
        for k in range(horizon):
            if policy is None and k >= len(actions):
                break
            action = policy(current) if policy else actions[k]
            c = Context(current, tuple(action), tuple(history), previous, memory, "imagined", len(prefix) + k)
            p = self.step(state, c)
            result.append(p)
            state = p.state
            # For neural-only inference no symbolic execution is required.
            if p.symbolic is not None:
                memory = p.symbolic.memory
            history = (history + [current])[-50:]
            previous = tuple(action)
            current = self.schema.decode(p.output)
            if p.terminated:
                break
        return result
