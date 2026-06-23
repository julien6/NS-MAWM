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
evaluate_world_model.py
                    VAE reconstruction and RNN one-step diagnostics
compare_ns_variants.py
                    NS-MAWM batch comparison across variants
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
../.venv/bin/python evaluate_world_model.py --vae-only --episodes 2 --max-steps 20
../.venv/bin/python series.py --limit 2
../.venv/bin/python rnn_train.py --steps 5
../.venv/bin/python evaluate_world_model.py --rnn-one-step --episodes 2 --max-steps 20
../.venv/bin/python train.py --generations 1 --max_len 5
../.venv/bin/python model.py gridcraftrnn render log/gridcraftrnn.cma.16.64.best.json --max-steps 5
../.venv/bin/python model.py gridcraftreal render log/gridcraftrnn.cma.16.64.best.json --max-steps 5
../.venv/bin/python model.py gridcraftreal norender log/gridcraftrnn.cma.16.64.best.json --episodes 3 --max-steps 5
```

## Full Workflow

The default commands for a first serious world-model run are:

```bash
../.venv/bin/python extract.py
../.venv/bin/python vae_train.py
../.venv/bin/python series.py
../.venv/bin/python rnn_train.py
```

Those defaults currently mean:

- `extract.py`: 5000 random episodes, 500 max steps.
- `vae_train.py`: 10000 VAE optimization steps, batch size 256.
- `series.py`: encode all recorded episodes.
- `rnn_train.py`: 10000 MDN-RNN steps, batch size 64, sequence length 32.

The same world-model-only workflow can be launched with one script:

```bash
./train_world_model.bash
```

This script runs world-model diagnostics:

- after VAE training: `trainlog/vae_eval.json`
- after MDN-RNN training: `trainlog/rnn_eval.json`

Both files report `grid_mismatch`, per-plane mismatch for `terrain`, `block`,
and `entity`, plus `self_mse`. Lower is better. The RNN diagnostic is one-step:
it compares the real next observation with the next observation imagined by the
MDN-RNN under a random policy.

`rnn_train.py` also reports `mean_mse`. This is an auxiliary loss that directly
trains the deterministic MDN-RNN mean used by `--imagination-mode mean`, so it
is the most relevant training signal for visual real-vs-imagined comparison.

Controller training is skipped by default. Enable it explicitly when needed:

```bash
TRAIN_CONTROLLER=1 ./train_world_model.bash
```

That optional controller step runs `train.py`: CMA-ES controller, 16 evaluation
episodes per candidate, 64 candidates, 100 generations by default.

To remove generated artifacts and restart from a clean experiment state:

```bash
./clean_experiment.bash --yes
./train_world_model.bash
```

Its parameters can be overridden with environment variables, for example:

```bash
EXTRACT_EPISODES=1000 VAE_STEPS=2000 RNN_STEPS=2000 ./train_world_model.bash
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

Compare real and imagined observations under a random policy. The render shows
the real full grid, the real observation panel, then the imagined observation
panel on the far right:

```bash
../.venv/bin/python model.py gridcraftcompare render --max-steps 500 --render-delay 0.1 --render-hold 20 --imagination-mode mean
../.venv/bin/python model.py gridcraftcompare norender --episodes 100 --max-steps 500 --imagination-mode mean
```

Use a manual policy to inspect specific transitions. Hold `q`, `z`, `d`, or `s`
to move left, up, right, or down; no key means `stay`:

```bash
../.venv/bin/python model.py gridcraftcompare render --policy manual --max-steps 500 --render-delay 0.2 --render-hold 20 --imagination-mode mean
```

`--imagination-mode mean` is deterministic and best for debugging fidelity.
`--imagination-mode mode` uses the most likely mixture component. Use
`--imagination-mode sample` only when you want to inspect stochastic rollouts.

If the VAE metrics are already acceptable but the imagined next observation is
still inconsistent, retrain only the latent dynamics:

```bash
../.venv/bin/python series.py
../.venv/bin/python rnn_train.py
../.venv/bin/python evaluate_world_model.py --rnn-one-step --episodes 100 --max-steps 500 --imagination-mode mean
```

Run the diagnostics directly:

```bash
../.venv/bin/python evaluate_world_model.py --vae-only --episodes 100 --max-steps 500
../.venv/bin/python evaluate_world_model.py --rnn-one-step --episodes 100 --max-steps 500 --imagination-mode mean
```

## NS-MAWM

NS-MAWM is implemented as an architecture-agnostic symbolic layer around the
existing VAE + MDN-RNN world model. The default `neural` variant is the original
pipeline. The symbolic variants operate in decoded tabular observation space:

- `regularization`: adds symbolic consistency loss during RNN training.
- `projection`: applies symbolic projection at inference on top of `neural`.
- `residual`: trains the RNN with an auxiliary loss focused on non-symbolic
  components, then assembles symbolic and neural observations at inference.

Train and compare the NS-MAWM variants:

```bash
./train_ns_world_model.bash
```

For a short test run:

```bash
RNN_STEPS=5 RNN_BATCH_SIZE=4 RNN_SEQ_LEN=5 EVAL_EPISODES=2 EVAL_MAX_STEPS=20 EVAL_HORIZON=5 ./train_ns_world_model.bash
```

Evaluate one variant:

```bash
../.venv/bin/python evaluate_world_model.py --rnn-one-step --ns-variant projection --episodes 100 --max-steps 500 --horizon-steps 50
```

Render one variant:

```bash
../.venv/bin/python model.py gridcraftcompare render --ns-variant projection --policy manual --max-steps 500 --render-delay 0.2 --render-hold 20 --imagination-mode mean
```

Batch comparison outputs:

```text
trainlog/ns_eval_neural.json
trainlog/ns_eval_regularization.json
trainlog/ns_eval_projection.json
trainlog/ns_eval_residual.json
trainlog/ns_mawm_summary.json
```

## Reproducible Baselines And W&B

The reproducibility runner is `run_baseline.py`. It supports three operational
families on Gridcraft:

- `B10`: classic VAE + MDN-RNN world model.
- `B24` / `B25` / `B26`: VAE + MDN-RNN with NS-MAWM regularization, projection, or residual behavior at symbolic coverage `0.3`.
- `B27` / `B28` / `B29`: the same NS-MAWM variants at symbolic coverage `0.6`.
- `B00`: model-free policy trained directly in the real Gridcraft environment.

World-model baselines periodically checkpoint and evaluate one-step and
multi-horizon compounding metrics:

```bash
../.venv/bin/python run_baseline.py --baseline-id B10 --phase world_model --eval-every 1000 --horizons 1 5 10 25 50 --wandb
../.venv/bin/python run_baseline.py --baseline-id B25 --phase world_model --eval-every 1000 --horizons 1 5 10 25 50 --wandb
```

Outputs are written under:

```text
runs/<baseline_slug>_seed<seed>/
  baseline_config.json
  wandb_panels.json
  series/
  rnn/checkpoints/checkpoint_<step>.json
  eval/world_model_step_<step>.json
  eval.json
```

Policy baselines are launched through the same runner:

```bash
# model-free policy in real Gridcraft
../.venv/bin/python run_baseline.py --baseline-id B00 --phase policy --policy-baseline real_mappo --wandb

# policy trained only in the imagined environment, then evaluated in real Gridcraft
../.venv/bin/python run_baseline.py --baseline-id B25 --phase policy --policy-baseline imagined_mappo --wandb

# MPC-CEM planning in real Gridcraft using the trained world model
../.venv/bin/python run_baseline.py --baseline-id B25 --phase policy --policy-baseline mpc_cem --wandb
```

For a complete one-run-per-baseline execution, use `--phase all` with
`--policy-baseline all`. This creates exactly one W&B run for the baseline and
seed, containing world-model extraction/training/evaluation plus the downstream
policy evaluations:

```bash
../.venv/bin/python run_baseline.py \
  --baseline-id B25 \
  --phase all \
  --policy-baseline all \
  --eval-every 1000 \
  --horizons 1 5 10 25 50 \
  --wandb
```

For model-based baselines, `--policy-baseline all` runs both `imagined_mappo`
and `mpc_cem` inside the same parent run. Their W&B metrics are prefixed under
the MARL sections, for example `MARL evaluation/mpc_cem/eval_real_reward` and
`MARL evaluation/imagined_mappo/eval_real_reward`. For `B00`, `all` resolves to
the real-environment model-free policy only.

The policy implementation is currently a local mono-agent actor-critic runner
with MAPPO-compatible logging keys. It is intentionally small so the protocol is
operational with the current `.venv`; a strict BenchMARL/TorchRL MAPPO runner can
replace `policy_baselines.py` once those dependencies are installed.

Convenience scripts:

```bash
# B10 plus B24-B29 world-model baselines
WANDB=1 ./run_world_model_baselines.bash

# B00 real policy plus B24-B29 imagined-only and MPC-CEM model-based policies
WANDB=1 ./run_policy_baselines.bash

# Complete Gridcraft protocol
WANDB=1 ./run_repro_gridcraft.bash
```

Requested NS-MAWM protocol with serious default budgets:

```bash
WANDB=1 ./run_requested_baselines_serious.bash
```

`run_requested_baselines_serious.bash` is the recommended script when you want
`1 run = 1 baseline`. It runs `B00` once per seed, then each requested
model-based baseline once per seed with `--phase all --policy-baseline all`.

This runs `B00` as the model-free real-environment MARL baseline, then
`B24/B25/B26` for regularization/projection/residual at symbolic coverage `0.3`
and `B27/B28/B29` for the same strategies at symbolic coverage `0.6`. Defaults
use `SEEDS="1 2 3"`, `EXTRACT_EPISODES=20000`, `VAE_STEPS=50000`,
`RNN_STEPS=100000`, world-model evaluation every `5000` RNN steps,
`EVAL_EPISODES=200`, `POLICY_UPDATES=2000`, and `POLICY_EVAL_EPISODES=50`.
Override these variables from the shell to scale the run up or down.

`MPC-CEM` can also batch its imagined rollouts through TensorFlow while keeping
the real Gridcraft environment scalar. This is opt-in:

```bash
WANDB=1 \
BATCHED_CEM=1 \
CEM_SAMPLES=256 \
PLANNING_HORIZON=25 \
./run_policy_baselines.bash
```

The batched path uses deterministic latent predictions by default. Use
`BATCHED_CEM_SAMPLE_Z=1` only when you explicitly want stochastic latent
sampling. For `projection` and `residual`, symbolic correction remains the same
CPU observation-space projection as the scalar path. W&B logs
`planning_step_time_ms`, `planning_batch_size`, `planning_horizon`, and
`planning_device` under `MARL evaluation`.

W&B sweeps are available under `sweeps/`. Start with the smoke sweep:

```bash
./create_sweep.bash sweeps/smoke.yaml
```

Then run the `wandb agent ...` command printed by W&B from this `gridcraft/`
directory. See `sweeps/README.md` for the world-model and policy sweep configs.

For smoke tests:

```bash
RNN_STEPS=5 EVAL_EVERY=1 EVAL_EPISODES=1 EVAL_MAX_STEPS=5 EVAL_HORIZONS="1 3" BASELINES="B10 B25" ./run_world_model_baselines.bash
POLICY_UPDATES=1 EPISODES_PER_UPDATE=1 POLICY_EVAL_EVERY=1 POLICY_EVAL_EPISODES=1 MAX_STEPS=5 MODEL_BASELINES="B25" ./run_policy_baselines.bash
```

W&B defaults:

```text
project: ns-mawm-gridcraft
group: baseline id, for example B10 or B25
name: <baseline_slug>_seed<seed>
```

Each `run_baseline.py` invocation creates at most one W&B run. Stage scripts
stream progress through local JSONL files, and `run_baseline.py` relays those
metrics, evaluations, and videos into the single baseline run. Running 6
baselines therefore creates 6 W&B runs.

`run_baseline.py --list` prints the deterministic `baseline_slug` for every
baseline. The slug keeps the `BXX` prefix and appends the main keywords that
define the baseline, for example:

```text
B10_neural-wm_vae-mdn-rnn_random_gridcraft_none_sv-sv_neural_k0.0
B25_ns-mawm-strategy_vae-mdn-rnn_random_gridcraft_projection_uv-mv_projection_k0.3
B00_model-free-control_none_real_gridcraft_none_sv-sv_neural_k0.0
```

Logged world-model metrics include training losses, `grid_mismatch`,
`terrain_mismatch`, `block_mismatch`, `entity_mismatch`, `self_mse`, RVR,
determinable/undeterminable mismatch, and horizon-prefixed compounding metrics.
Policy metrics include real and imagined rewards, real-imagined reward gap,
episode length, planning imagined return, and planning action entropy.
When `--wandb` is active, short RGB videos are logged by default:

```text
World Model evaluation/video_real_vs_imagined
MARL evaluation/video_policy_rollout
```

Model-based videos show the full real grid, the real local observation, and the
imagined local observation generated by the world model. Model-free B00 videos
show only the full real grid and real local observation. Use
`--video-max-steps`, `--video-episodes`, and `--video-fps` to control clip size,
or `--no-wandb-videos` to disable video generation for long sweeps.

In W&B, metrics are routed into four stable sections:

```text
General
World Model training
World Model evaluation
MARL training
MARL evaluation
```

`General` contains run documentation panels: a PSTR rule catalog with explicit
rule identifiers and English descriptions, the baseline configuration, and the
architecture/algorithm configuration for the VAE, MDN-RNN, NS-MAWM integration,
and downstream MARL/control algorithm.

Each active section also receives an HTML `00 Information - ...` panel with a
collapsible `Information` block explaining what the panel shows, how to
interpret it, the expected trend, and common failure modes. Disable these cards
with `--no-wandb-info-panels` if you only want scalar curves.

## Baseline Experiments And W&B

The NeurIPS-style baseline runner stores each baseline in an independent
directory under `runs/` and can optionally log to Weights & Biases.

List available Gridcraft baselines:

```bash
../.venv/bin/python run_baseline.py --list
```

Run one baseline locally without W&B:

```bash
../.venv/bin/python run_baseline.py --baseline-id B25 --steps 10000 --episodes 100 --horizons 1 5 10 25 50
```

Run a short smoke baseline:

```bash
../.venv/bin/python run_baseline.py --baseline-id B25 --steps 5 --batch-size 4 --seq-len 5 --episodes 2 --max-steps 20 --horizons 1 5 --series-limit 2
```

Run the targeted Gridcraft baseline suite:

```bash
./run_baselines.bash
```

Enable W&B:

```bash
WANDB=1 WANDB_PROJECT=ns-mawm-gridcraft ./run_baselines.bash
```

For offline cluster runs:

```bash
WANDB=1 WANDB_MODE=offline WANDB_PROJECT=ns-mawm-gridcraft ./run_baselines.bash
```

Useful overrides:

```bash
BASELINES="B10 B24 B25 B26" SEEDS="1 2 3" RNN_STEPS=20000 EVAL_EPISODES=200 EVAL_HORIZONS="1 5 10 25 50" ./run_baselines.bash
```

Logged quantities include WM losses, one-step prediction fidelity,
long-horizon/compounding metrics, RVR, determinable and residual mismatch, and
run metadata matching the sampled baseline IDs from the paper.

The current VAE latent size is `64`, so older `vae/vae.json`, `rnn/rnn.json`,
and controller logs trained with `z_size=32` are no longer compatible. Re-run
`series.py`, `rnn_train.py`, and `train.py` after retraining the VAE.

## Convenience Scripts

```bash
./extract.bash --episodes 1000 --max-steps 500
./gpu_jobs.bash --steps 5000 --batch-size 128
./gce_train.bash --generations 100 --max_len 500
```

Generated rollouts, encoded series, logs, and TensorFlow checkpoint-style
directories are ignored by git. Baseline JSON model files under `vae/`, `rnn/`,
and `initial_z/` are intentionally kept in the experiment layout.
