"""Schema adapter for the pinned, published multi-agent MAMBA RSSM."""
from types import SimpleNamespace
import torch
from torch import nn
from torch.nn import functional as F
from .vendor.mamba.configs.dreamer.DreamerAgentConfig import DreamerConfig
from .vendor.mamba.agent.models.DreamerModel import DreamerModel
from .vendor.mamba.environments import Env

UPSTREAM_COMMIT = '2c97258f71bf1c421c40ce14fd2f7cc3fe7fe19f'


def configuration(schema, action_size, width=256, categories=32, classes=32):
    cfg = DreamerConfig()
    cfg.IN_DIM, cfg.ACTION_SIZE = schema.shape[0], action_size
    cfg.HIDDEN = cfg.MODEL_HIDDEN = cfg.EMBED = cfg.DETERMINISTIC = width
    cfg.REWARD_HIDDEN = cfg.PCONT_HIDDEN = cfg.ACTION_HIDDEN = cfg.VALUE_HIDDEN = width
    cfg.N_CATEGORICALS, cfg.N_CLASSES = categories, classes
    cfg.STOCHASTIC = categories * classes
    cfg.FEAT = cfg.STOCHASTIC + width
    cfg.GLOBAL_FEAT = cfg.FEAT + width
    cfg.ENV_TYPE = Env.STARCRAFT
    return cfg


class MambaWM(nn.Module):
    """Published discrete RSSM and inter-agent attention, semantic decoder loss."""
    def __init__(self, schema, action_size, *, enc=(256,256), lstm=256, g=(256,256),
                 dec=(256,256), residual=False, control=False, categories=32, classes=32):
        super().__init__()
        if lstm % 8:
            raise ValueError('MAMBA width must be divisible by its eight attention heads')
        self.schema, self.action_size = schema, action_size
        self.residual, self.control = residual, control
        self.config = configuration(schema, action_size, lstm, categories, classes)
        self.model = DreamerModel(self.config)
        self.extra_encoder = nn.Linear(schema.shape[0]*3, schema.shape[0]) if residual else None
        # Shared reward/termination architecture across integration strategies.
        self.auxiliary_loss = None

    def init_state(self, prefix=(), batch_size=1):
        p = next(self.parameters())
        state = (self.model.representation.initial_state(batch_size, self.schema.shape[1], device=p.device),
                 p.new_zeros(batch_size, self.schema.shape[1], self.action_size), None)
        for obs, action, extra in prefix:
            _, state = self.step(state, obs, action, extra)
        return state

    def step(self, state, obs, joint_action, extra=None):
        previous, previous_action, _ = state
        x = obs.transpose(-2,-1)
        if self.residual:
            if extra is None:
                raise ValueError('Residual needs mask and target')
            x = self.extra_encoder(torch.cat((obs, *extra), -2).transpose(-2,-1))
        embed = self.model.observation_encoder(x)
        prior, posterior = self.model.representation(embed, previous_action, previous)
        shape = (*posterior.logits.shape[:-1], self.config.N_CATEGORICALS, self.config.N_CLASSES)
        q = torch.distributions.Categorical(logits=posterior.logits.reshape(shape))
        p = torch.distributions.Categorical(logits=prior.logits.reshape(shape))
        self.auxiliary_loss = torch.distributions.kl_divergence(q,p).mean()
        action = F.one_hot(joint_action.long(), self.action_size).float()
        predicted = self.model.transition(action, posterior)
        output, _ = self.model.observation_decoder(predicted.get_features())
        return output.transpose(-2,-1), (posterior, action, predicted)

    def heads(self, state):
        features = state[2].get_features()
        reward = self.model.reward_model(features).mean(-2).squeeze(-1)
        # Upstream pcont predicts continuation, not termination.
        probability = self.model.pcont(features).mean.mean(-2).squeeze(-1).clamp(1e-6,1-1e-6)
        return reward, torch.logit(1-probability)
