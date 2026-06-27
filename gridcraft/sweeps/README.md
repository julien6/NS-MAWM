# Gridcraft W&B Sweeps

Run commands from `gridcraft/`.

## Smoke Sweep

Use this first to verify the agent, W&B project, metric sections, and `General`
panels:

```bash
./create_sweep.bash sweeps/smoke.yaml
```

W&B prints an agent command:

```bash
wandb agent ENTITY/ns-mawm-gridcraft/SWEEP_ID
```

Run that command from `gridcraft/`. The sweep launches `sweep_agent.py`, which
calls `run_baseline.py` with the sampled parameters.

## World Model Baselines

```bash
./create_sweep.bash sweeps/world_model_baselines.yaml
wandb agent ENTITY/ns-mawm-gridcraft/SWEEP_ID
```

This grid compares the classic neural baseline and the requested NS-MAWM
coverage/strategy baselines across seeds:

```text
B10 neural VAE+MDN-RNN
B24 regularization coverage 0.3
B25 projection coverage 0.3
B26 residual coverage 0.3
B27 regularization coverage 0.6
B28 projection coverage 0.6
B29 residual coverage 0.6
```

The objective is:

```text
World Model evaluation/grid_mismatch
```

Lower is better.

## Policy Baselines

```bash
./create_sweep.bash sweeps/policy_baselines.yaml
wandb agent ENTITY/ns-mawm-gridcraft/SWEEP_ID
```

This grid compares downstream-control baselines with one W&B run per baseline:

```text
B00 model-free MASAC in real vGridcraft
B24 model-based MAMBPO + optional MPC-CEM
B25 model-based MAMBPO + optional MPC-CEM
B26 model-based MAMBPO + optional MPC-CEM
B27 model-based MAMBPO + optional MPC-CEM
B28 model-based MAMBPO + optional MPC-CEM
B29 model-based MAMBPO + optional MPC-CEM
```

The objective is:

```text
MARL evaluation/eval_real_reward
```

Higher is better.

## Notes

- `sweep_agent.py` reuses the active W&B sweep run; one sampled baseline creates
  one W&B run and no nested stage runs.
- Videos are enabled in the smoke sweep and log to
  `World Model evaluation/video_real_vs_imagined` or
  `MARL evaluation/video_policy_rollout`. Set `no_wandb_videos: true` in a
  sweep config to disable them for large runs.
- Baseline outputs are stored under `runs/sweeps/...`.
- Set `WANDB_ENTITY` and `WANDB_PROJECT` before creating the sweep if needed.
- Use `WANDB_MODE=offline` for local debugging, but create real sweeps online.
