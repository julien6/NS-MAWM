# NS-MAWM

Neuro-Symbolic Multi-Agent World Model (NS-MAWM) is a small, research-friendly library for learning multi-agent world dynamics while integrating symbolic rules. It combines a neural backbone (encoder + LSTM latent dynamics + decoder) with a rule engine that can inject deterministic structure, and supports multiple neuro-symbolic strategies (regularization, projection, residual replacement).

## Why NS-MAWM

- **Multi-agent world modeling**: predict next observations for multiple agents given current observations and actions.
- **Neuro-symbolic integration**: enforce or regularize predictions against hand-crafted rules.
- **Simple, hackable core**: minimal abstractions, clean PyTorch modules, easy-to-extend rule interface.
- **Metrics & training utilities**: built-in rule-violation metric and Lightning trainer.

## Key Concepts

### Model

`NSMAWM` wraps a neural backbone (`MAWMBackbone`) and a symbolic `RuleEngine`. At each step:

1. The rule engine produces a partial prediction `omega_d` and a mask indicating where rules apply.
2. The neural backbone predicts the full next observation.
3. A strategy combines neural and symbolic outputs.

Supported strategies:

- **`reg`**: regularize neural predictions to match symbolic values on masked features (`lambda_symb` controls weight).
- **`reg+proj`**: regularize during training, and project to symbolic values at inference.
- **`proj`**: project final predictions to symbolic values where rules apply.
- **`residual`**: replace masked elements with symbolic values.

### Rules

Rules are lightweight classes that map a `RuleContext` to a `RuleResult`:

- `RuleContext` contains `obs_t`, `act_t`, optional `prev_hidden`, and optional `meta`.
- `RuleResult` provides `values` and a boolean `mask` with the same shape as observations.

### Rule Engine

`RuleEngine` aggregates multiple rules and resolves collisions:

- `collision="last"` (default): later rules overwrite earlier ones.
- `collision="error"`: raise an error if two rules write to the same masked element.

### Metric

**RVR (Rule Violation Rate)** measures the fraction of masked elements where predictions violate the symbolic rule beyond `eps`.

## Installation

From a local checkout:

```bash
pip install -e .
```

Dependencies are declared in `pyproject.toml` (Python >= 3.10).

## Quickstart

### 1) Run the toy demo

```bash
python examples/toy_demo.py
```

The toy demo trains on synthetic data with a simple "stay" rule, then reports RVR under different strategies.

### 2) (Optional) Overcooked-AI minimal demo

```bash
python examples/overcooked_demo.py
```

Requires `overcooked_ai_py` (Overcooked-AI). The demo collects random transitions and trains with simple position-based rules.

### 2) Train via CLI (Hydra)

```bash
nsmawm
```

Override settings with Hydra-style config arguments:

```bash
nsmawm model.n_agents=4 training.strategy=proj training.max_epochs=10
```

## Usage

### Define a rule

```python
import torch
from nsmawm.symbolic.rule import Rule, RuleContext, RuleResult


class StayRule(Rule):
    """If action=stay, positions remain unchanged."""

    def apply(self, context: RuleContext) -> RuleResult:
        obs = context.obs_t
        act = context.act_t
        stay = act[..., 0] > 0.5
        mask = stay.unsqueeze(-1).expand_as(obs)
        values = torch.where(mask, obs, torch.zeros_like(obs))
        return RuleResult(values=values, mask=mask)
```

### Build and train the model

```python
import torch
from torch.utils.data import DataLoader

from nsmawm.data.datasets import TransitionsDataset
from nsmawm.models.mawm_backbone import BackboneConfig
from nsmawm.models.nsmawm import NSMAWM
from nsmawm.symbolic.engine import RuleEngine
from nsmawm.training.trainer import fit

backbone_cfg = BackboneConfig(
    n_agents=2,
    n_features=2,
    action_dim=2,
    latent_dim=32,
    hidden_dim=32,
    encoder_hidden=64,
    decoder_hidden=64,
)

rule_engine = RuleEngine([StayRule()])
model = NSMAWM.from_config(backbone_cfg, rule_engine=rule_engine, strategy="reg", lambda_symb=1.0)

obs = torch.randn(512, 2, 2)
act = torch.randn(512, 2, 2)
next_obs = torch.randn_like(obs)
dataset = TransitionsDataset(obs, act, next_obs)
train_loader = DataLoader(dataset, batch_size=64, shuffle=True)

fit(model, train_loader, max_epochs=3, learning_rate=1e-3)
```

### Predict and evaluate RVR

```python
from nsmawm.metrics.rvr import compute_rvr

with torch.no_grad():
    output = model.forward(obs[:32], act[:32], apply_projection=True)
    rvr = compute_rvr(output.prediction, output.omega_d, output.mask)
    print("RVR:", rvr.item())
```

## Overcooked-AI Featurizer

The Overcooked adapter exposes a configurable featurizer for positions, orientations, and held items.

```python
from nsmawm.envs.overcooked import OvercookedAdapter, OvercookedFeatureConfig

feature_cfg = OvercookedFeatureConfig(
    include_positions=True,
    include_orientation=True,
    include_holding=True,
    holding_vocab=("onion", "tomato", "dish", "soup"),
)
adapter = OvercookedAdapter(env, feature_config=feature_cfg)

feature_indices = adapter.feature_indices()
pos_idx = feature_indices["positions"]
orient_idx = feature_indices["orientation"]
holding_idx = feature_indices["holding"]
```

`feature_indices()` returns a dictionary of indices to make symbolic rules robust to feature ordering.

Example rule using `feature_indices()`:

```python
from nsmawm.symbolic.overcooked_rules import StayPutRule

stay_idx = adapter.get_stay_action_index()
feature_indices = adapter.feature_indices()
rule = StayPutRule(stay_action_index=stay_idx, feature_indices=feature_indices)
```

Feature slices for block-wise operations:

```python
slices = adapter.feature_slices()
pos_slice = slices["positions"]
orient_slice = slices["orientation"]
holding_slice = slices["holding"]
```

Feature index layout (when `include_positions=True`, `include_orientation=True`, `include_holding=True` with default vocab):

```
positions:   x=0, y=1
orientation: NORTH=2, SOUTH=3, EAST=4, WEST=5
holding:     none=6, onion=7, tomato=8, dish=9, soup=10, unknown=11
```

You can also print the same layout with:

```python
print(adapter.feature_index_table())
```

## Configuration

`NSMAWMConfig` provides structured defaults for model, training, data, and trainer options.

Key fields (see `src/nsmawm/config/base.py`):

- `model`: number of agents/features, action dim, latent/hidden sizes.
- `training`: batch size, epochs, learning rate, `strategy`, `lambda_symb`.
- `data`: dataset size and sequence length for the CLI synthetic dataset.
- `trainer`: Lightning trainer options.

## Project Structure

```
src/nsmawm/
  cli.py                # Hydra CLI entrypoint
  config/               # Dataclass configs
  data/                 # Simple dataset utilities
  metrics/              # RVR metric
  models/               # MAWM backbone + NS-MAWM wrapper
  symbolic/             # Rule interfaces, engine, masks, corrections
  training/             # Lightning module + fit helper
examples/
tests/
```

## Tests

```bash
pytest
```

## License

MIT (see `LICENSE`).
