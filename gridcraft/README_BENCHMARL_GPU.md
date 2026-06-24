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
