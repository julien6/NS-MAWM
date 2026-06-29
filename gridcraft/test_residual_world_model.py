import sys

import torch

sys.path.insert(0, ".")

from run_benchmarl_dyna_gridcraft import apply_ns_mawm_to_latent_step
from torch_world_model import TorchGridcraftRNN, TorchGridcraftVAE


def test_residual_observation_head_projects_and_reencodes():
  vae = TorchGridcraftVAE()
  rnn = TorchGridcraftRNN()
  z = torch.randn(3, vae.z_size)
  action = torch.tensor([8, 5, 8])

  next_z, _reward, _done, _state, residual_obs = rnn.step_with_observation(
    z,
    action,
    None,
    deterministic=True,
  )
  corrected_z, memory, metrics = apply_ns_mawm_to_latent_step(
    vae=vae,
    current_z=z,
    predicted_z=next_z,
    action=action,
    ns_memory=[None],
    ns_variant="residual",
    ns_coverage=0.0,
    num_agents=3,
    device=torch.device("cpu"),
    predicted_obs_vector=residual_obs,
  )

  assert residual_obs.shape == (3, 550)
  assert corrected_z.shape == (3, vae.z_size)
  assert len(memory) == 1
  assert metrics["projected_observation"].shape == (3, 550)
