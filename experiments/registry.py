"""B01-B45 baseline registry.

The registry is part of the experiment layer because it names environments, policies,
and world-model architectures.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BaselineSpec:
    baseline_id: str
    family: str
    wm: str
    policy: str
    environment: str
    coverage: float
    strategy: str
    eval_regime: str
    train_regime: str


def build_baselines() -> dict[str, BaselineSpec]:
    specs: list[BaselineSpec] = []
    for i, policy in enumerate(["MAPPO"] * 3 + ["QMIX"] * 3 + ["SAC"] * 3, 1):
        regime = ["SV", "KV", "UV"][(i - 1) % 3]
        specs.append(BaselineSpec(f"B{i:02d}", "Model-free control", "MF", policy, "GridCraft", 0.0, "-", regime, "SV" if regime == "SV" else "MV"))
    for i, wm in enumerate(["RSSM"] * 3 + ["Deterministic"] * 3 + ["Transformer"] * 3, 10):
        regime = ["SV", "KV", "UV"][(i - 10) % 3]
        specs.append(BaselineSpec(f"B{i:02d}", "Neural WM", wm, "MAPPO", "GridCraft", 0.0, "-", regime, "SV" if regime == "SV" else "MV"))
    for bid, coverage in zip(range(19, 24), [0.1, 0.3, 0.5, 0.7, 1.0]):
        specs.append(BaselineSpec(f"B{bid:02d}", "NS-MAWM coverage", "RSSM", "MAPPO", "GridCraft", coverage, "regularization", "UV", "MV"))
    for bid, strategy in zip(range(24, 27), ["regularization", "projection", "residual"]):
        specs.append(BaselineSpec(f"B{bid:02d}", "NS-MAWM strategy", "RSSM", "MAPPO", "GridCraft", 0.3, strategy, "UV", "MV"))
    for bid, regime in zip(range(27, 30), ["SV", "KV", "UV"]):
        specs.append(BaselineSpec(f"B{bid:02d}", "NS-MAWM training regime", "RSSM", "MAPPO", "GridCraft", 0.3, "regularization", regime, "SV" if regime == "SV" else "MV"))
    for bid, env, cov, strategy in [
        (30, "Overcooked", 0.0, "-"), (31, "PredatorPrey", 0.0, "-"), (32, "SMACv2", 0.0, "-"),
        (33, "Overcooked", 0.3, "regularization"), (34, "PredatorPrey", 0.3, "regularization"), (35, "SMACv2", 0.3, "regularization"),
    ]:
        specs.append(BaselineSpec(f"B{bid:02d}", "Environment transfer", "RSSM", "MAPPO", env, cov, strategy, "UV", "MV"))
    for bid, policy in zip(range(36, 41), ["Random", "Scripted", "MAPPO", "QMIX", "SAC"]):
        specs.append(BaselineSpec(f"B{bid:02d}", "Data-generator ablation", "RSSM", policy, "GridCraft", 0.0, "-", "UV", "MV"))
    for bid, policy in zip(range(41, 46), ["Random", "Scripted", "MAPPO", "QMIX", "SAC"]):
        specs.append(BaselineSpec(f"B{bid:02d}", "NS-MAWM data-generator", "RSSM", policy, "GridCraft", 0.3, "regularization", "UV", "MV"))
    return {spec.baseline_id: spec for spec in specs}


BASELINES = build_baselines()
