# NS-MAWM Reproduction

This repository is split into four concerns:

- `env_adapters`: adapters for installed external environments.
- `marl_lib`: PyTorch MARL policy/data-generator algorithms.
- `wm_lib`: PyTorch world-model architectures.
- `ns_mawm`: environment-, policy-, and WM-agnostic neuro-symbolic composition.

The root installation scripts are kept separate from the research code.

## Smoke Test

```bash
source venv/bin/activate
python -m pip install -e .[dev]
python -m pytest
python scripts/reproduce_table4.py | head
```
