# NS-MAWM World Models Experiments

This repository is currently organized around reproducing and extending the
World Models experiments from Ha and Schmidhuber, 2018. It contains modernized
CarRacing and DoomRNN experiments plus a new mono-agent Gridcraft experiment.

The code is intentionally close to the historical experiment layout:

```text
carracing/   CarRacing World Models experiment
doomrnn/     DoomTakeCover World Models experiment
gridcraft/   mono-agent Gridcraft World Models experiment
scripts/     environment setup scripts
```

The original reference is the Otoro tutorial:

https://blog.otoro.net/2018/06/09/world-models-experiments/

## Environment

Use the existing virtual environment from the repository root:

```bash
python3.10 -m venv .venv
./scripts/setup_worldmodels_env.sh
./scripts/setup_gridcraft_env.sh
```

The World Models dependencies are pinned in:

```text
requirements-worldmodels.txt
```

`Gridcraft/` is an editable external checkout installed with:

```bash
pip install -e Gridcraft
```

That external checkout is ignored by the root git repository. If you change
Gridcraft itself, commit or inspect those changes from inside `Gridcraft/`.

## CarRacing

From `carracing/`, pretrained controller evaluation follows the historical
style:

```bash
cd carracing
../.venv/bin/python model.py render log/carracing.cma.16.64.best.json
../.venv/bin/python model.py norender log/carracing.cma.16.64.best.json
```

See [carracing/README.md](carracing/README.md) for experiment-specific notes.

## DoomRNN

From `doomrnn/`, use the same modernized environment and run the DoomRNN entry
points from that directory:

```bash
cd doomrnn
../.venv/bin/python model.py render log/doomrnn.cma.16.64.best.json
../.venv/bin/python model.py norender log/doomrnn.cma.16.64.best.json
```

See [doomrnn/README.md](doomrnn/README.md) for Doom-specific commands.

## Gridcraft

Gridcraft is a structured-observation World Models experiment. The VAE sees a
fixed vector built from the local tabular observation, not pixels:

- local grid planes: terrain, block, entity
- agent state vector: hp, hunger, inventory counts
- mono-agent setup: `GridcraftConfig(num_agents=1)`

Install Gridcraft support:

```bash
./scripts/setup_gridcraft_env.sh
```

Run a short smoke pipeline:

```bash
cd gridcraft
../.venv/bin/python env.py --steps 10
../.venv/bin/python extract.py --episodes 2 --max-steps 20
../.venv/bin/python vae_train.py --steps 5 --batch-size 8
../.venv/bin/python series.py --limit 2
../.venv/bin/python rnn_train.py --steps 5
../.venv/bin/python train.py --generations 1 --max_len 5
```

Launch the full training-only workflow:

```bash
cd gridcraft
./train_world_model.bash
```

Evaluate after training:

```bash
../.venv/bin/python model.py gridcraftreal norender log/gridcraftrnn.cma.16.64.best.json
../.venv/bin/python model.py gridcraftreal render log/gridcraftrnn.cma.16.64.best.json
../.venv/bin/python model.py gridcraftrnn render log/gridcraftrnn.cma.16.64.best.json
```

See [gridcraft/README.md](gridcraft/README.md) for all Gridcraft commands and
tunable training parameters.

## Generated Files

Experiment outputs are intentionally ignored:

- rollout archives: `record/`
- latent series: `series/`
- TensorFlow/checkpoint-style folders: `tf_*`
- controller logs: `log/`
- training logs: `trainlog/`

Baseline model JSON files under Gridcraft `vae/`, `rnn/`, and `initial_z/` are
not ignored, so a small runnable baseline can be kept with the experiment code.

## Citation

If you use the original World Models work in an academic setting, cite:

```latex
@incollection{ha2018worldmodels,
  title = {Recurrent World Models Facilitate Policy Evolution},
  author = {Ha, David and Schmidhuber, J{\"u}rgen},
  booktitle = {Advances in Neural Information Processing Systems 31},
  pages = {2451--2463},
  year = {2018},
  publisher = {Curran Associates, Inc.},
  url = {https://papers.nips.cc/paper/7512-recurrent-world-models-facilitate-policy-evolution},
  note = "\url{https://worldmodels.github.io}",
}
```

## License

MIT, following the original World Models Experiments repository.
