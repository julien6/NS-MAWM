from __future__ import annotations

import argparse
import json
from pathlib import Path
import torch
import yaml
from .schema import Schema
from .data import Dataset
from .environments import ROOT, make_env, collect
from .libraries import make_library, load_library, save_library, corrupt
from .rules import Library, PSWM
from .training import train, evaluate, build_network
from .artifacts import Run, code_identity, export_anonymous
from .diagnostics import dashboard
from .experiments import plan, report
from .refinement import RuleWorkflow, refine, leakage_audit, PAPER_DESCRIPTIONS
from .validation import library_card
from .policies import collection_policies
from .provenance import RuleArchive


def setup(config):
    env = make_env(config["run"]["env"], **config.get("environment", {}))
    expected = config.get("run", {}).get("env_commit")
    if expected and env.metadata.get("commit") != expected:
        env.close()
        raise ValueError("Configured benchmark commit differs from installed environment")
    return env


def load_components(config):
    dataset = Dataset.load(config["data"])
    schema = Schema.from_dict(json.loads((Path(config["data"]) / "schema.json").read_text()))
    env = setup(config)
    if env.schema.shape != schema.shape or tuple(env.schema.by_name) != tuple(schema.by_name):
        raise ValueError("Dataset schema does not match configured environment")
    if config.get("run",{}).get("split_manifest"):
        manifest=json.loads(Path(config["run"]["split_manifest"]).read_text())
        if manifest.get("hash") != dataset.hash:
            env.close()
            raise ValueError("Configured split manifest differs from loaded dataset")
    rule_config = config.get("rules", {})
    lib = load_library(rule_config["library"], schema) if rule_config.get("library") else make_library(env, schema)
    if rule_config.get("conflict_policy"):
        lib = Library(list(lib.rules), schema, edges=lib.edges, policy=rule_config["conflict_policy"],
                      parent=lib.parent, creator=lib.creator, rule_filter=lib.rule_filter)
    # Validate all handcrafted rules against revision data before enforcement.
    card = library_card(lib, dataset, max_rdd=rule_config.get("max_rdd", .1),
                        minimum_support=rule_config.get("minimum_support", 1))
    if not card["validation"]["valid"]:
        raise ValueError("Library validation failed: " + json.dumps(card["validation"]))
    filters = dict(rule_config.get("filter", {}))
    filters["disable_ids"] = sorted(set(filters.get("disable_ids", [])) | set(card["excluded"]))
    lib = lib.filtered(**filters)
    card_dir = Path(config.get("output", "outputs/ns_mawm"))
    card_dir.mkdir(parents=True, exist_ok=True)
    (card_dir / "library-card.json").write_text(json.dumps(card, indent=2))
    if rule_config.get("corruption"):
        lib = corrupt(lib, **rule_config["corruption"], seed=config["run"].get("seed", 0))
    return env, dataset, schema, lib


def main(argv=None):
    from .extensions import load_plugins
    load_plugins()
    parser = argparse.ArgumentParser(prog="ns-mawm")
    parser.add_argument("stage", choices=["collect", "generate-rules", "refine", "train", "evaluate", "control", "report", "export-anon", "plan", "freeze", "tune", "generate-code", "campaign", "verify", "trace"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--split", choices=["rev", "select", "test"], default="select")
    args = parser.parse_args(argv)
    from .configuration import validate_config
    config = validate_config(yaml.safe_load(Path(args.config).read_text()), args.stage)
    torch.set_num_threads(config.get("threads", 1))
    output = Path(config.get("output", "outputs/ns_mawm"))
    output.mkdir(parents=True, exist_ok=True)
    if args.stage == "campaign":
        from .campaign import run_campaign
        result = run_campaign(config)
        print(json.dumps({k:v["status"] for k,v in result.items()}, indent=2))
        if any(v["status"] != "completed" for v in result.values()):
            raise RuntimeError("Campaign contains failed or blocked jobs; inspect campaign.json")
        return
    if args.stage == "verify":
        from .verification import verify
        result = verify(config)
        print(json.dumps(result, indent=2))
        if any(result.get(key,{}).get("status") == "failed" for key in ("tests", "types")):
            raise RuntimeError("Acceptance checks failed; inspect acceptance.json")
        return
    if args.stage == "collect":
        env = setup(config)
        try:
            cfg = config.get("collection", {})
            policies = collection_policies(env, cfg)
            try:
                episodes = collect(env, cfg.get("episodes", 1000), cfg.get("seed", 0), policies)
            finally:
                for _, policy in policies.values():
                    if hasattr(policy, "close"): policy.close()
            dataset = Dataset.split(episodes, cfg.get("split_seed", 0), metadata=env.metadata)
            schema = env.schema.fit(torch.cat([e.observations for e in dataset.get("train")]))
            dataset.save(config["data"])
            (Path(config["data"]) / "schema.json").write_text(json.dumps(schema.to_dict(), indent=2))
            print(json.dumps(dataset.summary(), indent=2))
        finally:
            env.close()
        return
    if args.stage == "plan":
        result = plan(config, seconds_per_update=config.get("seconds_per_update"))
        (output / "plan.json").write_text(json.dumps(result, indent=2))
        print(json.dumps({k: v for k, v in result.items() if k != "runs"}, indent=2))
        return
    if args.stage == "report":
        report(output / "raw_runs.csv", output / "report", config.get("stats", {}).get("comparisons", []), config.get("stats", {}).get("resamples", 10000), config.get("stats", {}).get("extension"))
        return
    if args.stage == "export-anon":
        export_anonymous(**config["export"])
        return
    env, dataset, schema, library = load_components(config)
    try:
        if args.stage in ("generate-rules", "refine", "generate-code"):
            cfg = dict(config.get("llm", {}))
            if "source_tree" not in cfg:
                if config["run"]["env"] == "gridcraft":
                    cfg["source_tree"] = str(ROOT / "vGridcraft" / "vgridcraft")
                elif config["run"]["env"] == "overcooked":
                    cfg["source_tree"] = str(ROOT / "Overcooked_AI" / "src")
                else:
                    import importlib
                    module_name = {"predator_prey":"mpe2", "smacv2":"smacv2"}.get(config["run"]["env"], type(env).__module__)
                    package = importlib.import_module(module_name)
                    cfg["source_tree"] = str(Path(package.__file__).parent)
            if config["run"].get("condition") == "A11":
                if not cfg.get("independent_author") or not cfg.get("sections") or any(s["source"] != "public_docs" for s in cfg["sections"]):
                    raise ValueError("A11 requires an independently authored public-docs prompt and author attestation")
            workflow = RuleWorkflow(schema, dataset, output / "llm", cfg)
            sections = cfg.get("sections", [{"source": "paper_description", "reference": "NS-MAWM SRS section 8.2",
                                              "text": PAPER_DESCRIPTIONS.get(config["run"]["env"], "Use the supplied public schema and documented action semantics; abstain when premises are unknown.")}])
            cost_config = json.loads(json.dumps(config))
            cost_config["run"]["experiment"] = "llm_cost"
            cost_run = Run(output, cost_config, library, dataset, config["run"].get("seed",0), code_identity(ROOT))
            try:
                if args.stage == "generate-code":
                    from .refinement import generate_code
                    selected = generate_code(workflow, config, env.action_size, sections)
                else:
                    selected = refine(workflow, library, config, env.action_size, sections,
                                      rounds=1 if args.stage == "generate-rules" else None, source_tree=cfg.get("source_tree"))
                if selected:
                    save_library(selected, output / "selected_library.json")
            finally:
                from .reporting import record_llm_cost
                record_llm_cost(cost_run, workflow)
                cost_run.finish()
            return
        if args.stage == "trace":
            from .tuning import version_trace
            from .rules import compile_rule
            records = [json.loads(Path(p).read_text()) for p in config.get("rules",{}).get("trace_versions",[])]
            if not records:
                raise ValueError("trace requires archived rule versions")
            versions = [compile_rule(r) for r in records]
            result = version_trace(dataset, schema, library, versions, env.action_size, config)
            (output / "version-trace.json").write_text(json.dumps(result, indent=2))
            from .reporting import record_numbers
            for version, row in zip(versions, result):
                trace_run = Run(output, {**config, "trace_version": row["version"]}, library.swapped(version), dataset, config["run"].get("seed",0), code_identity(ROOT))
                record_numbers(trace_run, row["result"]["metrics"], "trace."+row["rule"]+".v"+str(row["version"]))
                record_numbers(trace_run, row["result"]["diagnostics"], "trace_diagnostics."+row["rule"]+".v"+str(row["version"]))
                trace_run.finish()
            return
        if args.stage == "tune":
            from .tuning import tune
            result = tune(dataset, schema, library, env.action_size, config, output)
            print(json.dumps({k:v for k,v in result.items() if k not in ("curve", "configuration")}))
            return
        if args.stage == "freeze":
            dataset.freeze(library.hash, config)
            dataset.save(config["data"])
            print(json.dumps(dataset.frozen))
            return
        run = Run(output, config, library, dataset, config["run"].get("seed", 0), code_identity(ROOT))
        save_library(library, run.path / "library.json")
        archive = RuleArchive(output / "rules")
        for rule in library.rules:
            archive.add(rule)
        if args.stage == "control":
            if config.get("control", {}).get("learner") == "mamba":
                from .mamba_control import run_control as control
            elif config.get("control", {}).get("learner", "benchmarl") == "benchmarl":
                from .benchmarl import run_control as control
            else:
                from .control import control
            result = control(env, dataset, schema, library, config, run)
            (run.path / "control.json").write_text(json.dumps(result, indent=2))
            run.finish()
            print(run.path)
            return
        if args.stage == "train":
            if dataset.frozen is not None:
                raise PermissionError("Training after test protocol freeze is disabled")
            network, timing = train(dataset, schema, library, env.action_size, config, run)
            run.manifest["training"] = timing
        else:
            if config["world_model"].get("backbone") in ("pswm_only", "llm_code"):
                network = None
            else:
                if not args.checkpoint:
                    raise ValueError("evaluate requires --checkpoint")
                checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
                if checkpoint["library_hash"] != library.hash or checkpoint["split_hash"] != dataset.hash:
                    raise ValueError("Checkpoint library/split provenance mismatch")
                if checkpoint["config"]["world_model"] != config["world_model"] or checkpoint["schema"] != schema.to_dict():
                    raise ValueError("Checkpoint architecture/training configuration or schema mismatch")
                network = build_network(schema, env.action_size, config, checkpoint.get("control", False))
                network.load_state_dict(checkpoint["network"])
        if config.get("rules", {}).get("diagnostic_library"):
            reference = load_library(config["rules"]["diagnostic_library"], schema)
        else:
            reference = make_library(env, schema)
            reference_card = library_card(reference, dataset, max_rdd=config.get("rules", {}).get("max_rdd", .1),
                                          minimum_support=config.get("rules", {}).get("minimum_support", 1))
            reference = reference.filtered(disable_ids=reference_card["excluded"])
        compared = [load_library(p, schema) for p in config.get("evaluation", {}).get("compared_libraries", [])]
        result = evaluate(dataset, schema, library, network, config, split=args.split, diagnostic_library=reference, compared_libraries=compared)
        (run.path / f"evaluation-{args.split}.json").write_text(json.dumps(result, indent=2))
        dashboard({config["world_model"].get("strategy", "none"): result["diagnostics"]}, run.path / "diagnostics.html")
        from .reporting import record_numbers
        record_numbers(run, result["diagnostics"], "diagnostic")
        record_numbers(run, result.get("imagined_diagnostics", {}), "imagined")
        record_numbers(run, run.manifest.get("training", {}), "training")
        for metric, value in result["metrics"].items():
            run.record(metric, value)
        for metric in ("proposed_coverage", "accepted_coverage", "conflict_fraction"):
            run.record(metric, result["diagnostics"][metric], unit="fraction")
        for step, error in result["per_step_error"].items():
            run.record(f"mse_step_{step}", error)
        run.manifest["evaluation_split"] = args.split
        run.manifest["diagnostic_library_hash"] = result["diagnostics"]["library_hash"]
        for rule in result["diagnostics"]["rules"]:
            for metric in ("rvr", "rvr_pre", "rdd"):
                run.record(f"{metric}.{rule['rule_id']}.v{rule['version']}", rule[metric], support=rule["support"], unit="structural" if metric == "rvr" and rule["structural"] else "fraction")
        run.finish()
        print(run.path)
    finally:
        env.close()
