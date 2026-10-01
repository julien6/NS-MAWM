from __future__ import annotations
from copy import deepcopy
import csv
import itertools
import json
from pathlib import Path
import numpy as np
import yaml
from .statistics import summarize, compare, holm, wording
from .rules import digest


def plan(base, *, seconds_per_update=None):
    if base.get("run", {}).get("experiment") == "control":
        runs=[]
        for label, learner, model_free in (("MASAC","benchmarl",True),("MAMBPO","benchmarl",False),("MAMBA","mamba",False)):
            for seed in range(base.get("seeds",10)):
                cfg=deepcopy(base);cfg["run"].update(seed=seed,condition=label)
                cfg.setdefault("control",{}).update(learner=learner,model_free=model_free)
                runs.append(cfg)
        return {"runs":runs,"total_real_steps":sum(c["control"].get("real_steps",1000000) for c in runs),
                "estimated_gpu_hours":None,"estimate_requires_measured_update_latency":True,"primary_conditions":["MASAC","MAMBPO","MAMBA"]}
    conditions = [("B1", {"backbone": "jopm_lstm", "strategy": "none"}),
                  ("B2", {"backbone": "mamba_wm", "strategy": "none"}),
                  ("B3", {"backbone": "dreamer_central", "strategy": "none"}),
                  ("B4", {"backbone": "pswm_only", "strategy": "none"}),
                  ("B6", {"backbone": "mamba_wm", "strategy": "regularization"}),
                  ("A1", {"strategy": "feature_weighting"})]
    for strategy in ("projection", "residual", "regularization"):
        conditions.append((strategy, {"strategy": strategy}))
    if base["run"]["env"] in ("gridcraft", "overcooked"):
        conditions.append(("B5", {"backbone": "llm_code", "strategy": "none"}))
    variants = []
    for label, change in conditions:
        c = deepcopy(base)
        c["world_model"].update(change)
        variants.append((label, c))
    for label, scope in (("A5", ["individual"]), ("A6", ["joint"])):
        c = deepcopy(base)
        c.setdefault("rules", {})["filter"] = {"scope": scope}
        variants.append((label, c))
    for coefficient in (0, .1, .3, 1, 3):
        c = deepcopy(base)
        c["world_model"].update(strategy="regularization", **{"lambda": coefficient})
        variants.append((f"A7-{coefficient}", c))
    for label, feedback in (("refined", "full"), ("A2", "no_rvr"), ("A3", "rvr_only"), ("A4", "full")):
        if label == "A3" and base["run"]["env"] != "gridcraft":
            continue
        c = deepcopy(base)
        c["run"]["experiment"] = "refinement"
        c.setdefault("llm", {}).update(feedback=feedback, max_candidates=1 if label == "A4" else 4)
        variants.append((label, c))
    if base["run"]["env"] in ("gridcraft", "overcooked"):
        for corruption, fraction in itertools.product(("random_values", "wrong_guards", "overclaimed_masks"), (0., .1, .2, .3, .4)):
            c = deepcopy(base)
            c.setdefault("rules", {})["corruption"] = {"kind": corruption, "fraction": fraction}
            variants.append((f"A8-{corruption}-{fraction}", c))
    c = deepcopy(base)
    c.setdefault("evaluation", {})["fixed_block_set"] = "union_of_versions"
    variants.append(("A9", c))
    if base["run"]["env"] in ("gridcraft", "predator_prey"):
        for agents in (2, 4, 6):
            c = deepcopy(base)
            c.setdefault("environment", {})["agents"] = agents
            variants.append((f"A10-{agents}", c))
    if base["run"]["env"] == "gridcraft":
        c = deepcopy(base)
        c.setdefault("llm", {})["allowed_sources"] = ["public_docs"]
        c["run"]["experiment"] = "refinement"
        variants.append(("A11", c))
    runs = []
    for label, variant in variants:
        for seed in range(base.get("seeds", 10)):
            config = deepcopy(variant)
            config["run"]["seed"] = seed
            config["run"]["condition"] = label
            config.setdefault("tuning", {})["configurations"] = 12
            llm_seeds = base.get("llm", {}).get("seeds", [0, 1, 2]) if config["run"].get("experiment") == "refinement" or label == "B5" else [None]
            for llm_seed in llm_seeds:
                member = deepcopy(config)
                if llm_seed is not None:
                    member.setdefault("llm", {})["seed"] = llm_seed
                runs.append(member)
    updates = 0
    for c in runs:
        if c["world_model"].get("backbone") in ("pswm_only", "llm_code"):
            continue
        tuning = c.get("tuning", {})
        fits = len(tuning.get("learning_rates", [1e-4,3e-4,1e-3])) * len(tuning.get("widths", [128,256])) * len(tuning.get("sequence_lengths", [25,50])) + 1
        label = c["run"]["condition"]
        if label == "B6": fits += 7
        elif c["world_model"].get("strategy") == "regularization" and not label.startswith("A7"): fits += 5
        if c["run"].get("experiment") == "refinement": fits += 2 + c.get("llm",{}).get("max_candidates",4)
        updates += fits * c["world_model"].get("optim",{}).get("updates",20000)

    return {"runs": runs, "total_updates": updates,
            "estimated_gpu_hours": updates * seconds_per_update / 3600 if seconds_per_update is not None else None,
            "budget_gpu_hours": 2000, "estimate_requires_measured_update_latency": seconds_per_update is None,
            "primary_conditions": ["A1", "A2", "A9", "A5"]}


def report(csv_path, output, comparisons=(), resamples=10000, extension=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with Path(csv_path).open() as f:
        rows = list(csv.DictReader(f))
    manifests = [json.loads(p.read_text()) for p in Path(csv_path).parent.rglob("manifest.json")]
    for manifest in manifests:
        if manifest.get("evaluation_split") == "test":
            frozen_stats = manifest.get("configuration",{}).get("stats",{})
            if digest(comparisons) != digest(frozen_stats.get("comparisons",[])) or digest(extension or {}) != digest(frozen_stats.get("extension",{})):
                raise PermissionError("Statistical comparisons or extension rules changed after test freeze")
    condition_hashes = {}
    for manifest in manifests:
        cfg=manifest.get("configuration", {})
        label=cfg.get("run",{}).get("condition")
        key=(cfg.get("run",{}).get("env"),label,cfg.get("llm",{}).get("seed"))
        if label: condition_hashes.setdefault(key,set()).add(manifest["config_hash"])
    resolved=[]
    for declaration in comparisons:
        if "left_condition" in declaration:
            for llm_seed in sorted({key[2] for key in condition_hashes if key[0]==declaration["env"]},key=str):
                left=condition_hashes.get((declaration["env"],declaration["left_condition"],llm_seed),set())
                right=condition_hashes.get((declaration["env"],declaration["right_condition"],llm_seed),set())
                if not right: right=condition_hashes.get((declaration["env"],declaration["right_condition"],None),set())
                if len(left)==1 and len(right)==1:
                    resolved.append({**declaration,"left":next(iter(left)),"right":next(iter(right)),"llm_seed":llm_seed})
        else: resolved.append(declaration)
    groups = {}
    for row in rows:
        if row["value"] in ("", "None") or row["checkpoint"] != "final":
            continue
        key = (row["env"], row["config_hash"], row["metric"])
        values = groups.setdefault(key, {})
        seed = int(row["seed"])
        if seed in values:
            raise ValueError(f"Duplicate seed-level metric: {key}, {seed}")
        values[seed] = float(row["value"])
    summary = [{"env": k[0], "config_hash": k[1], "metric": k[2], **summarize(list(v.values()), resamples=resamples)} for k, v in groups.items()]
    tests = []
    extension_decisions = []
    for declaration in resolved:
        env, metric = declaration["env"], declaration["metric"]
        a = groups[(env, declaration["left"], metric)]
        b = groups[(env, declaration["right"], metric)]
        if declaration.get("paired", True) and set(a) != set(b):
            raise ValueError("Paired comparisons require the same seed set in both arms")
        if extension and extension.get("enabled"):
            from .statistics import extension_decision
            extension_decisions.append({"comparison": declaration, **extension_decision({declaration["left"]:a,declaration["right"]:b},extension)})
        result = compare([a[s] for s in sorted(a)], [b[s] for s in sorted(b)], paired=declaration.get("paired", True), resamples=resamples)
        tests.append({**declaration, **result})
    for family in {t["family"] for t in tests}:
        members = [t for t in tests if t["family"] == family]
        for t, p in zip(members, holm([t["p_primary"] for t in members])):
            t["adjusted_p"] = p
            t["wording"] = wording(t["difference"], p, lower_is_better=t.get("lower_is_better", True))
    from .reporting import metadata_notes, development_cost, figures
    attainment = {}
    for row in rows:
        if "to_threshold" in row["metric"] or "to_target" in row["metric"]:
            key = row["config_hash"] + ":" + row["metric"]
            counts = attainment.setdefault(key, {"attained": 0, "total": 0})
            counts["total"] += 1
            counts["attained"] += row["value"] not in ("", "None")
    variance = []
    for env,label in sorted({(k[0],k[1]) for k in condition_hashes},key=str):
        means=[]
        for (e,l,llm_seed), hashes in condition_hashes.items():
            if (e,l)!=(env,label) or llm_seed is None: continue
            values=[v for h in hashes for v in groups.get((env,h,"mse"),{}).values()]
            if values: means.append({"llm_seed":llm_seed,"mean_mse":float(np.mean(values)),"neural_seeds":len(values)})
        if means: variance.append({"env":env,"condition":label,"libraries":means,"between_library_sd":float(np.std([m["mean_mse"] for m in means],ddof=1)) if len(means)>1 else None})
    result = {"library_variance": variance, "summary": summary, "comparisons": tests, "comparison_protocol_hash": digest(comparisons),
              "notes": metadata_notes(Path(csv_path).parent), "attainment": attainment,
              "development_cost": development_cost(Path(csv_path).parent), "extension_decisions": extension_decisions}
    (output / "report.json").write_text(json.dumps(result, indent=2))
    with (output / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, ["env", "config_hash", "metric", "n", "mean", "sd", "ci95"])
        writer.writeheader()
        writer.writerows(summary)
    from .reporting import paper_tables
    paper_tables(rows, summary, tests, output)
    figures(csv_path, output)
    return result
