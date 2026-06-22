import numpy as np

from ns_symbolic import apply_symbolic_projection
from rnn.rnn import GridcraftRNN, rnn_init_state
from vae.vae import GridcraftVAE


class WorldModelAdapter:
  def reset(self, seed=None):
    raise NotImplementedError

  def encode(self, obs_vector):
    raise NotImplementedError

  def decode(self, latent):
    raise NotImplementedError

  def predict_next(self, obs_vector, current_tabular_obs, action, mode="mean", ns_variant="neural", rng=None):
    raise NotImplementedError


class VaeMdnRnnAdapter(WorldModelAdapter):
  def __init__(self, vae=None, rnn=None):
    self.vae = vae or GridcraftVAE()
    self.rnn = rnn or GridcraftRNN()
    self.rnn_state = rnn_init_state(self.rnn)

  def reset(self, seed=None):
    self.rnn_state = rnn_init_state(self.rnn)

  def encode(self, obs_vector):
    mu, _ = self.vae.encode_mu_logvar(np.asarray(obs_vector, dtype=np.float32).reshape(1, -1))
    return mu[0]

  def decode(self, latent):
    return self.vae.decode_tabular(latent)

  def predict_next(self, obs_vector, current_tabular_obs, action, mode="mean", ns_variant="neural", rng=None):
    latent = self.encode(obs_vector)
    predicted_latent, self.rnn_state = predict_next_latent(
      self.rnn,
      self.rnn_state,
      latent,
      action,
      rng=rng,
      mode=mode,
    )
    predicted_obs = self.decode(predicted_latent)
    projected_obs, symbolic_info = apply_symbolic_projection(
      predicted_obs,
      current_tabular_obs,
      action,
      ns_variant,
    )
    return projected_obs, predicted_latent, symbolic_info


def predict_next_latent(rnn, rnn_state, latent, action, rng=None, mode="mean"):
  logmix, mean, logstd, reward, done_logit, next_state = rnn.step(latent, action, rnn_state)
  mix = np.exp(logmix - np.max(logmix, axis=1, keepdims=True))
  mix = mix / np.sum(mix, axis=1, keepdims=True)
  if mode == "mean":
    next_latent = np.sum(mix * mean, axis=1).astype(np.float32)
  elif mode == "mode":
    component = np.argmax(mix, axis=1)
    next_latent = mean[np.arange(rnn.z_size), component].astype(np.float32)
  elif mode == "sample":
    if rng is None:
      rng = np.random.default_rng()
    next_latent = np.zeros((rnn.z_size,), dtype=np.float32)
    for j in range(rnn.z_size):
      component = int(rng.choice(np.arange(rnn.num_mixture), p=mix[j]))
      next_latent[j] = mean[j, component] + np.exp(logstd[j, component]) * rng.standard_normal()
  else:
    raise ValueError(f"unknown imagination mode: {mode}")
  return next_latent, next_state
