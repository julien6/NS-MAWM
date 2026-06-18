# Gridcraft World Models

This folder mirrors the `carracing/` and `doomrnn/` experiments with a mono-agent
Gridcraft task. The observation model is structured, not pixel based: the local
grid planes are one-hot encoded and concatenated with the agent state vector.

The initial objective is the native Gridcraft survival reward with
`GridcraftConfig(num_agents=1)`.

## Layout

```text
env.py              real Gridcraft mono-agent wrapper
model.py            controller evaluation in the real environment
train.py            ES/CMA controller training
extract.py          random rollout collection
series.py           rollout encoding into latent sequences
vae_train.py        structured VAE training
rnn_train.py        MDN-RNN training
dream_env.py        learned Gridcraft world model environment
dream_model.py      controller evaluation inside the dream environment
render_env.py       real environment render wrapper
render_model.py     rendered controller evaluation
config.py           World Models-style game registry
es.py               local ES optimizers
vae/vae.py          structured VAE implementation
rnn/rnn.py          MDN-RNN implementation
initial_z/          initial latent states for dream rollouts
```

No `gridcraft/__init__.py` is provided on purpose, because it would shadow the
editable `Gridcraft/gridcraft` package.

## Install

From the repository root:

```bash
./scripts/setup_gridcraft_env.sh
```

The setup script installs the local Gridcraft checkout with:

```bash
pip install -e Gridcraft
```

## Smoke Pipeline

Run from this directory:

```bash
cd gridcraft

../.venv/bin/python env.py --steps 10
../.venv/bin/python extract.py --episodes 2 --max-steps 20
../.venv/bin/python vae_train.py --steps 5 --batch-size 8
../.venv/bin/python series.py --limit 2
../.venv/bin/python rnn_train.py --steps 5
../.venv/bin/python train.py --generations 1 --max_len 5
../.venv/bin/python model.py gridcraftrnn render log/gridcraftrnn.cma.16.64.best.json --max-steps 5
../.venv/bin/python model.py gridcraftreal render log/gridcraftrnn.cma.16.64.best.json --max-steps 5
../.venv/bin/python model.py gridcraftreal norender log/gridcraftrnn.cma.16.64.best.json --episodes 3 --max-steps 5
```

## Full Workflow

The default commands are configured for a first serious run:

```bash
../.venv/bin/python extract.py
../.venv/bin/python vae_train.py
../.venv/bin/python series.py
../.venv/bin/python rnn_train.py
../.venv/bin/python train.py
```

Those defaults currently mean:

- `extract.py`: 5000 random episodes, 500 max steps.
- `vae_train.py`: 10000 VAE optimization steps, batch size 256.
- `series.py`: encode all recorded episodes.
- `rnn_train.py`: 10000 MDN-RNN steps, batch size 64, sequence length 32.
- `train.py`: CMA-ES controller, 16 evaluation episodes per candidate, 64 candidates, 100 generations.

The same training-only workflow can be launched with one script:

```bash
./train_world_model.bash
```

Its parameters can be overridden with environment variables, for example:

```bash
EXTRACT_EPISODES=1000 VAE_STEPS=2000 RNN_STEPS=2000 ES_GENERATIONS=20 ./train_world_model.bash
```

Evaluate in the real environment:

```bash
../.venv/bin/python model.py gridcraftreal norender log/gridcraftrnn.cma.16.64.best.json
../.venv/bin/python model.py gridcraftreal render log/gridcraftrnn.cma.16.64.best.json
```

Evaluate in the learned environment:

```bash
../.venv/bin/python model.py gridcraftrnn norender log/gridcraftrnn.cma.16.64.best.json
../.venv/bin/python model.py gridcraftrnn render log/gridcraftrnn.cma.16.64.best.json
```

## Convenience Scripts

```bash
./extract.bash --episodes 1000 --max-steps 500
./gpu_jobs.bash --steps 5000 --batch-size 128
./gce_train.bash --generations 100 --max_len 500
```

Generated rollouts, encoded series, logs, and TensorFlow checkpoint-style
directories are ignored by git. Baseline JSON model files under `vae/`, `rnn/`,
and `initial_z/` are intentionally kept in the experiment layout.
