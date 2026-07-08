# Gridcraft GPU/CPU BenchMARL Path

Recommended smoke command:

```bash
../.venv/bin/python run_benchmarl_gridcraft.py \
  --baseline-id B25_projection_k0.3 \
  --phase all \
  --num-envs 8 \
  --episodes 16 \
  --max-steps 20 \
  --vae-steps 5 \
  --rnn-steps 5 \
  --eval-every 5 \
  --device cpu
```

Recommended serious command on a CUDA machine:

```bash
../.venv/bin/python run_benchmarl_gridcraft.py \
  --baseline-id B25_projection_k0.3 \
  --phase all \
  --num-envs 256 \
  --episodes 5000 \
  --max-steps 500 \
  --vae-steps 10000 \
  --rnn-steps 10000 \
  --wm-batch-size 1024 \
  --wm-num-workers 8 \
  --eval-every 1000 \
  --device cuda \
  --wandb
```

Datasets are cached under `datasets/gridcraft/` using a deterministic key from the environment config, number of episodes, max steps, and seed. Use `--force-recollect` to rebuild the dataset.

Single-command pipeline: World Model train/eval first, then downstream MARL
train/eval. By default, `B00` uses MASAC in real vGridcraft and model-based
baselines use MAMBPO:

```bash
cd gridcraft
./run_full_benchmarl_baseline.bash
```

Smoke version of the same pipeline:

```bash
cd gridcraft
DEVICE=cpu WANDB_FLAG= WM_NUM_ENVS=8 WM_EPISODES=16 WM_MAX_STEPS=20 \
VAE_STEPS=5 RNN_STEPS=5 WM_BATCH_SIZE=32 WM_NUM_WORKERS=0 WM_EVAL_EVERY=5 \
MARL_NUM_ENVS=4 MARL_MAX_STEPS=20 MARL_MAX_ITERS=1 MARL_FRAMES_PER_BATCH=64 \
./run_full_benchmarl_baseline.bash
```

vGridcraft rendering uses the existing Gridcraft renderer:

```python
from vgridcraft import VGridcraftConfig, VectorizedGridcraftEnv

env = VectorizedGridcraftEnv(num_envs=8, config=VGridcraftConfig(num_agents=1))
frame = env.render(env_index=0, mode="rgb_array")
env.render(env_index=0, mode="human")
env.close()
```

## Automatic Spark resource profile

For long campaigns and HPO on Spark, enable the automatic resource profile. It
detects CUDA, free VRAM, CPU count and RAM, then exports aggressive but bounded
values for vectorized environments, world-model batches, DataLoader workers,
BenchMARL frame batches and replay memory.

```bash
cd gridcraft
AUTO_RESOURCE_PROFILE=1 RESOURCE_PROFILE=spark_max \
./run_benchmarl_requested_baselines_3agents_fast_scientific.bash
```

The same mechanism is available for HPO:

```bash
cd gridcraft
AUTO_RESOURCE_PROFILE=1 RESOURCE_PROFILE=spark_max HPO_MODE=serious \
./run_world_model_hpo_pipeline.bash

AUTO_RESOURCE_PROFILE=1 RESOURCE_PROFILE=spark_max MARL_HPO_MODE=serious \
./run_marl_hpo_pipeline.bash
```

Inspect the selected settings without launching a run:

```bash
../.venv/bin/python resource_profile.py --profile spark_max --target all --format summary
../.venv/bin/python resource_profile.py --profile spark_max --target all --format shell
```

Explicit environment variables still win. For example, setting
`WM_NUM_ENVS=1024` before the command keeps that value even when
`AUTO_RESOURCE_PROFILE=1` is enabled.
