# NS-MAWM

Minimal implementation of the [software requirements specification](NS-MAWM%20%E2%80%94%20Software%20Requirements%20Specification.md): observation schemas, partial symbolic rules, neural world models, rule diagnostics and refinement, and multi-agent control.

## Setup

Use Python 3.11 or 3.12. Run commands from the repository root.

```bash
git submodule update --init Overcooked_AI
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-ns-mawm.txt
```

For Gridcraft prediction alone, `python -m pip install -e '.[test]'` is sufficient. Full setup adds the vendored BenchMARL learner, the pinned Overcooked submodule, MPE2 PredatorPrey, and SMACv2. Live SMACv2 runs additionally require a separately installed StarCraft II client and maps. LLM refinement requires a configured OpenAI-compatible endpoint.

## Short checks

```bash
python -m pytest tests -q
python -m ns_mawm collect --config configs/smoke.yaml
python -m ns_mawm train --config configs/smoke.yaml
```

The smoke configuration collects eight short episodes and runs two optimizer updates. Tests also exercise short control runs when BenchMARL dependencies are installed. These commands do not launch the full experimental campaign.

Other CLI stages include `evaluate`, `control`, `generate-rules`, `refine`, `tune`, `freeze`, `plan`, `report`, and `export-anon`; use `python -m ns_mawm --help`. Larger example configurations are in `configs/`. Generated datasets, checkpoints, and reports belong under ignored `outputs/`.

## Source layout

- `ns_mawm/`: the unified framework and CLI.
- `vGridcraft/`: the vectorized Gridcraft simulator, packaged by the root `pyproject.toml`.
- `BenchMARL/`: vendored learner code, including the NS-MAWM integration; retain its license.
- `Overcooked_AI/`: pinned upstream environment submodule; retain its license.
- `tests/`, `configs/`, `.github/`: regression tests, example configurations, and CI.

Optional Gridcraft rendering uses the upstream simulator. It is not needed for training:

```bash
python -m pip install 'gridcraft @ git+https://github.com/julien6/Gridcraft.git@fb00bb4f6229dfc1d1857b939f5cb42ee50096a2'
```

Legacy World Models, VAE/MDN-RNN checkpoint integrations, and the old experiment launchers are no longer part of the supported code paths. Control uses the shared NS-MAWM semantic model through BenchMARL by default; the small reference learner remains available for isolated tests.

## Campaigns and acceptance

```bash
python -m ns_mawm campaign --config configs/campaign-smoke.yaml
python -m ns_mawm verify --config configs/verify.yaml
```

`campaign` executes a serial dependency graph and resumes only jobs whose configuration, code, and output hashes still match. `configs/campaign-smoke.yaml` runs two tiny conditions. Production prediction configs expand to the SRS conditions; `control-gridcraft.yaml` and `control-overcooked.yaml` expand to MASAC, MAMBPO, and the published MAMBA learner. These production commands are intentionally not run during code verification.

`tune` supports baseline hyperparameters, the five-value lambda sweep, and strategy selection. `generate-code` is the full-transition-code baseline and rejects partial programs. `trace` consumes archived rule files through `rules.trace_versions`. The independent documentation condition A11 requires `llm.sections` tagged `public_docs` and an `llm.independent_author` attestation; it cannot fabricate independent human authorship.

The MAMBA compatibility port is under `ns_mawm/vendor/mamba`, with its upstream license, pinned commit, original file hashes, and listed adaptations. Prediction uses its discrete RSSM; external control uses its learner and SMAC preset. Explicit `control.mamba` overrides are recorded and are intended for bounded smoke tests. The centralized Dreamer-style model is an architectural baseline, not DreamerV3 reproduction.

`verify` writes a machine-readable acceptance report and test log. Missing StarCraft II, an LLM endpoint, original regression confidence intervals, or a full performance measurement stays **unverified**. Set `verify.live: true` only to request bounded live checks. T-19/T-20 comparison input is a JSON `criteria` list with `env`, `config_hash`, `metric`, `ci95`, and `seeds`, supplied as `verify.reference_results`; `verify.baseline_results` points to the corresponding run-level CSV. No reference results are invented.

The requirement map is `ns_mawm/requirements.json`. A passing test is evidence for its exercised behavior; the report keeps scientific validation separate. Public extension contracts and adapters are checked with strict mypy. Generated artifacts remain under ignored `outputs/`.
