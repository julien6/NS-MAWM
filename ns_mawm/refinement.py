from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import re
import time
import urllib.request
from .artifacts import now
from .rules import Library, compile_rule, digest
from .libraries import save_library
from .training import train, evaluate
from .provenance import RuleArchive

P0 = """You author partial symbolic transition rules, not predictors that copy a neural model.
Read only declared immutable observation blocks, previous actions and explicit history/memory.
Never read simulator state, files, network, future observations or held-out test transitions.
UNKNOWN premises require abstention. Effects assign exact finite scalar values or a complete
category by name. Keep rule ids stable, increment versions on any edit and cite evidence ids.
Return one JSON object with a patches list. Each patch includes base_library_version,
candidate_version, rule_id, rule_version, operation (add/revise/remove/retain), read_blocks,
write_blocks, dependencies, guard_code, effect_code, assumptions, evidence_ids,
change_summary, test_cases. Code defines guard(c) returning Tri and effect(c) returning
list[Assignment(block, value, rule_id, rule_version)]. No imports or I/O. Tests contain
context, guard (TRUE/FALSE/UNKNOWN) and expected assignments. Never lower RVR by copying
neural outputs or by dropping coverage without diagnostic evidence. Explicit retain/no-change
is permitted. Use only ordinary bounded Python expressions and comprehensions."""


PAPER_DESCRIPTIONS = {
    "gridcraft": "SRS 8.2: water preserves known static terrain when movement is blocked; terrain.shift translates known stable terrain by established displacement; plank consumes one wood and produces two planks when crafting prerequisites hold; collision needs known contenders and resolution semantics; map requires aligned fresh observations. Abstain without these premises.",
    "overcooked": "SRS 8.2: empty-handed agents can pick up from dispensers; pot.add adds one held ingredient to a non-full pot and clears hands; pot.timer advances cooking below completion; delivering soup clears hands; blocked moves preserve position and update orientation; same-target or swapping players may collide under documented resolution semantics.",
    "predator_prey": "SRS 8.2: cooperative predators have a fixed prey controller. Velocity may be determined by damping and action forces only when no contact forces apply; relative landmarks follow own displacement; relative allies require both displacements and known joint actions. Abstain unless physical constants and contact margins are justified by supplied sources.",
    "smacv2": "SRS 8.2: unit types persist only with established continued visibility; dead agents produce zero native observations; enemy health may persist only without attacks, regeneration or other damage; ally visibility requires observed positions and known sight range. Never consult future visibility or hidden unit state."
}


def ngrams(text, n=8):
    tokens = re.findall(r"\w+|[^\w\s]", text)
    return {tuple(tokens[i:i+n]) for i in range(max(0, len(tokens)-n+1))}


def leakage_audit(sections, source_tree):
    matches = []
    prompt = set().union(*(ngrams(s["text"]) for s in sections)) if sections else set()
    for path in Path(source_tree).rglob("*.py"):
        overlap = prompt & ngrams(path.read_text(errors="replace"))
        if overlap:
            matches.append({"file": str(path.relative_to(source_tree)), "matches": [" ".join(g) for g in sorted(overlap)]})
    return {"flagged": bool(matches), "ngram_size": 8, "matches": matches,
            "source_tags": [s["source"] for s in sections]}


class RuleWorkflow:
    def __init__(self, schema, dataset, directory, config, transport=None):
        self.schema, self.dataset, self.config = schema, dataset, config
        self.path = Path(directory)
        self.path.mkdir(parents=True, exist_ok=True)
        self.transport = transport or self._http
        self.calls, self.tokens, self.latency = 0, 0, 0.
        self.last_response = None
        ledger = self.path / "calls.jsonl"
        if ledger.exists():
            for line in ledger.read_text().splitlines():
                record = json.loads(line)
                self.calls += 1
                self.tokens += record["prompt_tokens"] + record["completion_tokens"]
                self.latency += record["latency_seconds"]

    def _append(self, name, record):
        with (self.path / name).open("a") as f:
            f.write(json.dumps(record, default=str) + "\n")

    def _http(self, payload):
        endpoint = self.config.get("endpoint", "http://localhost:8000/v1").rstrip("/") + "/chat/completions"
        request = urllib.request.Request(endpoint, json.dumps(payload).encode(), {"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=float(self.config.get("timeout", 120))) as response:
            return json.load(response)

    def call(self, stage, sections, *, evidence=(), library=None, validation=None):
        if self.dataset.frozen is not None:
            raise PermissionError("No rule generation or refinement after test protocol freeze")
        if stage not in ("P1", "P2", "P3"):
            raise ValueError("Invalid prompt stage")
        allowed = set(self.config.get("allowed_sources", ["public_docs", "expert_statement", "paper_description"]))
        if any(s["source"] not in allowed for s in sections):
            raise PermissionError("Forbidden prompt knowledge source")
        revision_ids = set(self.dataset.manifest["splits"]["rev"])
        for e in evidence:
            if e.get("split") != "rev" or str(e.get("evidence_id", "")).rsplit(":", 1)[0] not in revision_ids:
                raise PermissionError("Only D_rev evidence may enter revision prompts")
        content = {"stage": stage, "schema": self.schema.to_dict(), "sections": sections,
                   "evidence": evidence, "library": library, "validation": validation}
        text = json.dumps(content, default=str)
        # Exact episode ids are checked as JSON tokens, avoiding substring collisions.
        test_ids = set(self.dataset.manifest["splits"]["test"])
        strings = re.findall(r'"([^"\\]*(?:\\.[^"\\]*)*)"', text)
        if any(s in test_ids or s.rsplit(":", 1)[0] in test_ids for s in strings):
            raise PermissionError("Test episode identifier detected in prompt")
        max_calls = int(self.config.get("max_calls", self.config.get("max_candidates", 4)))
        if self.calls >= max_calls or self.tokens >= self.config.get("token_budget", 400000):
            raise RuntimeError("LLM budget exhausted")
        payload = {"model": self.config.get("model", "qwen3.6-35b-a3b-nvfp4"),
                   "temperature": self.config.get("temperature", .2), "seed": self.config.get("seed", 0),
                   "max_tokens": min(self.config.get("max_tokens", 8192), self.config.get("token_budget", 400000) - self.tokens),
                   "messages": [{"role": "system", "content": P0}, {"role": "user", "content": text}],
                   "response_format": {"type": "json_object"}}
        audit = leakage_audit([{"source": "assembled_prompt", "text": P0 + text}], self.config["source_tree"]) if self.config.get("source_tree") else {"audited": False, "reason": "No source tree configured"}
        start = time.perf_counter()
        self.calls += 1
        raw, error = {}, None
        self.last_response = None
        try:
            raw = self.transport(payload)
            response = raw["choices"][0]["message"]["content"]
            self.last_response = response
            parsed = json.loads(response)
            if not isinstance(parsed.get("patches"), list):
                raise ValueError("Response must contain a patches list")
        except Exception as exc:
            error = str(exc)
            raise
        finally:
            elapsed = time.perf_counter() - start
            usage = raw.get("usage", {})
            # Missing usage is conservatively charged against the budget, never zero.
            prompt_tokens = usage.get("prompt_tokens", len(text) + len(P0))
            completion_tokens = usage.get("completion_tokens", payload["max_tokens"])
            self.tokens += prompt_tokens + completion_tokens
            self.latency += elapsed
            self._append("calls.jsonl", {"call_id": self.calls, "stage": stage, "date": now(), "payload": payload,
                "response": raw, "error": error, "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                "latency_seconds": elapsed, "gpu_seconds": elapsed * self.config.get("serving_gpus", 1),
                "model_version": self.config.get("model_version", payload["model"]), "serving_format": "NVFP4",
                "source_tags": [s["source"] for s in sections], "leakage_audit": audit})
        if self.tokens > self.config.get("token_budget", 400000):
            raise RuntimeError("LLM response exceeded the total token budget")
        return parsed

    def candidate(self, base, response):
        rules = {r.id: r for r in base.rules}
        dependencies = {}
        required = {"base_library_version", "candidate_version", "rule_id", "rule_version", "operation", "read_blocks",
                    "write_blocks", "dependencies", "guard_code", "effect_code", "assumptions", "evidence_ids", "change_summary", "test_cases"}
        for patch in response["patches"]:
            if not required <= patch.keys():
                raise ValueError("Incomplete patch response schema")
            if not isinstance(patch['rule_version'],int) or patch['rule_version'] < 0 or not isinstance(patch['rule_id'],str):
                raise ValueError('Rule id/version have invalid types')
            for name in ('read_blocks','write_blocks','dependencies','assumptions','evidence_ids','test_cases'):
                if not isinstance(patch[name],list): raise ValueError(f'{name} must be a list')
            revision_ids=set(self.dataset.manifest['splits']['rev'])
            if any(str(e).rsplit(':',1)[0] not in revision_ids for e in patch['evidence_ids']):
                raise ValueError('Candidate evidence must identify a revision episode')
            if patch["base_library_version"] != base.hash:
                raise ValueError("Patch refers to a different base library")
            rid, op = patch["rule_id"], patch["operation"]
            if op == "retain":
                if rid not in rules:
                    raise ValueError("Cannot retain an unknown rule")
                continue
            if op == "remove":
                if not patch["evidence_ids"]:
                    raise ValueError("Coverage removal requires evidence")
                del rules[rid]
                continue
            if op not in ("add", "revise") or (op == "add") == (rid in rules):
                raise ValueError("Invalid patch operation")
            prior = rules.get(rid)
            if prior and patch["rule_version"] <= prior.version:
                raise ValueError("Any changed rule needs a new version")
            record = {"id": rid, "version": patch["rule_version"], "stage": patch.get("stage", 0),
                      "scope": patch.get("scope", "joint" if len({self.schema.by_name[n].owner for n in patch["read_blocks"] if n in self.schema.by_name}) > 1 else "individual"),
                      "reads": patch["read_blocks"], "writes": patch["write_blocks"],
                      "code": patch["guard_code"] + "\n" + patch["effect_code"],
                      "assumptions": patch["assumptions"], "test_cases": patch["test_cases"],
                      "base_version": prior.version if prior else None, "source": "llm_refined" if prior else "llm_initial"}
            rules[rid] = compile_rule(record)
            dependencies[rid] = patch["dependencies"]
        pending, resolved = set(rules), {}
        while pending:
            ready = sorted(rid for rid in pending if all(dep in resolved for dep in dependencies.get(rid, [])))
            if not ready:
                raise ValueError("Cyclic or unknown rule dependencies")
            for rid in ready:
                r = rules[rid]
                stage = max([r.stage] + [resolved[d].stage + 1 for d in dependencies.get(rid, [])])
                resolved[rid] = replace(r, stage=stage)
                pending.remove(rid)
        edges = set(base.edges) | {(resolved[d].stage, resolved[r].stage) for r, ds in dependencies.items() for d in ds}
        library = Library(list(resolved.values()), self.schema, edges=sorted(edges), parent=base.hash, creator=f"llm:{self.calls}")
        archive = RuleArchive(self.path / "rules")
        for rule in library.rules:
            archive.add(rule, [e for p in response["patches"] if p["rule_id"] == rule.id for e in p["evidence_ids"]], decision="candidate")
        result = library.validate()
        for rule in library.rules:
            if rule.source != "handcrafted" and {case["guard"] for case in rule.test_cases} != {"TRUE", "FALSE", "UNKNOWN"}:
                result["valid"] = False
                result["failures"].append({"rule": rule.id, "error": "Generated rules require applicable, inapplicable and unknown tests"})
        self._append("candidates.jsonl", {"library_hash": library.hash, "response": response, "validation": result})
        save_library(library, self.path / f"candidate-{library.hash}.json")
        if not result["valid"]:
            raise ValueError(json.dumps(result))
        return library


def admissibility(report, handcrafted_coverage, config):
    minimum = config.get("min_support", 50)
    rdd = config.get("max_rdd", .1)
    enforced = [r for r in report["rules"] if r.get("enabled", True)]
    unvalidated = [r["rule_id"] for r in enforced if r["support"] < minimum]
    bad = [r["rule_id"] for r in enforced if r["support"] >= minimum and r["rdd"] > rdd]
    valid = (bool(enforced) and not bad and not unvalidated and (report["conflict_fraction"] or 0) <= config.get("max_conflict", .05)
             and (report["accepted_coverage"] or 0) >= config.get("coverage_floor_fraction", .5) * handcrafted_coverage)
    return {"admissible": valid, "excluded_unvalidated": unvalidated, "bad_rules": bad}


def refine(workflow, initial, config, action_size, sections, *, rounds=None, source_tree=None):
    if source_tree is not None:
        audit = leakage_audit(sections, source_tree)
        (workflow.path / "leakage_audit.json").write_text(json.dumps(audit, indent=2))
    budget = min(rounds or workflow.config.get("rounds", workflow.config.get("max_candidates", 4)), workflow.config.get("max_candidates", 4))
    regular = json.loads(json.dumps(config))
    regular["world_model"]["strategy"] = "regularization"
    baseline_network, baseline_timing = train(workflow.dataset, workflow.schema, initial, action_size, regular)
    workflow._append("candidates.jsonl", {"condition": "initial_reference", "training": baseline_timing})
    baseline = evaluate(workflow.dataset, workflow.schema, initial, baseline_network, regular, split="select")
    handcrafted_coverage = baseline["diagnostics"]["accepted_coverage"] or 0
    current, report, validation = initial, None, None
    candidates = []
    produced = 0
    rejected = None
    workflow.config.setdefault("max_calls", budget * 3)
    for index in range(workflow.config["max_calls"]):
        if produced >= budget or workflow.calls >= workflow.config["max_calls"]:
            break
        candidate_start_cost = workflow.latency
        stage = "P2" if validation else ("P1" if index == 0 else "P3")
        evidence = [e for r in (report or {}).get("rules", []) for e in r["counterexamples"]]
        feedback = {"library_hash": current.hash, "manifest": current.manifest(), "rule_code": {r.id: r.code for r in current.rules if r.source != "handcrafted" or "simulator_source" in workflow.config.get("allowed_sources", [])}, "diagnostics": report}
        if validation is not None:
            feedback["rejected_candidate"] = rejected
        ablation = workflow.config.get("feedback", "full")
        if report is not None and ablation in ("no_rvr", "rvr_only"):
            feedback = json.loads(json.dumps(feedback))
            feedback["diagnostics"].pop("by_scope", None)
            for key in list(feedback["diagnostics"]):
                if (ablation == "no_rvr" and "rvr" in key):
                    del feedback["diagnostics"][key]
            for row in feedback["diagnostics"]["rules"]:
                row.pop("interpretation", None)
                row.pop("owners", None)
                row.pop("structural", None)
                row.pop("application_violations", None)
                for key in list(row):
                    if (ablation == "no_rvr" and ("rvr" in key or key in ("violations", "pre_violations"))) or (ablation == "rvr_only" and ("rdd" in key or key == "data_violations")):
                        del row[key]
        try:
            response = workflow.call(stage, sections, evidence=evidence, library=feedback, validation=validation)
            rejected = response
            candidate = workflow.candidate(current, response)
            produced += 1
            # RDD validation is performed on D_rev before any enforcement.
            provisional = evaluate(workflow.dataset, workflow.schema, candidate, baseline_network, regular, split="rev")
            check = admissibility(provisional["diagnostics"], 0., workflow.config)
            preselect = evaluate(workflow.dataset, workflow.schema, candidate, baseline_network, regular, split="select")
            select_check = admissibility(preselect["diagnostics"], 0., workflow.config)
            excluded = sorted(set(check["excluded_unvalidated"] + check["bad_rules"] + select_check["excluded_unvalidated"] + select_check["bad_rules"]))
            candidate = candidate.filtered(disable_ids=excluded)
            network, timing = train(workflow.dataset, workflow.schema, candidate, action_size, regular)
            selection = evaluate(workflow.dataset, workflow.schema, candidate, network, regular, split="select")
            decision = admissibility(selection["diagnostics"], handcrafted_coverage, workflow.config)
            score = selection["metrics"]["mse"]
            record = {"candidate": candidate.hash, "mse25_select": score, "decision": decision,
                      "training": timing, "cost_seconds": workflow.latency - candidate_start_cost, "excluded_rules": excluded}
            workflow._append("candidates.jsonl", record)
            if decision["admissible"]:
                candidates.append((score, sum(candidate.enabled(r) for r in candidate.rules), workflow.latency - candidate_start_cost, candidate))
            current = candidate
            report = evaluate(workflow.dataset, workflow.schema, candidate, network, regular, split="rev")["diagnostics"]
            validation = None
        except (ValueError, RuntimeError, KeyError) as exc:
            rejected = workflow.last_response if workflow.last_response is not None else rejected
            validation = {"error": str(exc), "kind": type(exc).__name__}
            workflow._append("candidates.jsonl", {"round": index, "validation": validation, "response": rejected, "decision": "rejected"})
    selected = min(candidates, key=lambda c: c[:3])[-1] if candidates else None
    fallback_record = {}
    if selected is None:
        import torch
        fallback_config = json.loads(json.dumps(config))
        fallback_config["world_model"].update(backbone="jopm_lstm", strategy="none")
        fallback, timing = train(workflow.dataset, workflow.schema, initial, action_size, fallback_config)
        checkpoint = workflow.path / "neural-fallback.pt"
        torch.save({"network": fallback.state_dict(), "config": fallback_config, "schema": workflow.schema.to_dict(),
                    "library_hash": initial.hash, "split_hash": workflow.dataset.hash, "control": False}, checkpoint)
        workflow._append("candidates.jsonl", {"condition": "neural_fallback", "training": timing})
        fallback_record = {"fallback_checkpoint": str(checkpoint), "fallback_configuration": fallback_config}
    (workflow.path / "selection.json").write_text(json.dumps({**fallback_record, "library_hash": selected.hash if selected else None,
        "fallback": None if selected else "purely_neural", "calls": workflow.calls, "tokens": workflow.tokens,
        "llm_seconds": workflow.latency, "candidates": produced}, indent=2))
    return selected


def generate_code(workflow, config, action_size, sections):
    """Generate a total transition program; partial programs are rejected."""
    from .rules import Context, PSWM
    from .training import episode_contexts
    empty = Library([], workflow.schema)
    rejected, validation = None, None
    candidates = []
    workflow.config.setdefault('max_calls', 3 * workflow.config.get('max_candidates',4))
    code_section = {'source':'expert_statement','text':
        'Generate a full transition program as rules: collectively assign EVERY schema block in EVERY context. No neural model or partial fallback is available. Use only public observations and explicit history. Include TRUE/FALSE/UNKNOWN tests for every rule.'}
    # Put the instruction in the contract, not in a forged documentation source.
    for index in range(workflow.config['max_calls']):
        if len(candidates) >= workflow.config.get('max_candidates',4):
            break
        try:
            response = workflow.call('P2' if validation else 'P1', sections,
                library={'manifest':empty.manifest(),'instruction':code_section['text'],'rejected_candidate':rejected}, validation=validation)
            rejected=response
            candidate=workflow.candidate(empty,response)
            for split in ('train','rev','select'):
                for episode in workflow.dataset.get(split,purpose='train' if split=='train' else 'evaluate'):
                    for _,output in episode_contexts(episode,workflow.schema,PSWM(candidate)):
                        if not output.mask.all():
                            raise ValueError('Full transition code left uncovered blocks')
            cfg=json.loads(json.dumps(config)); cfg['world_model'].update(backbone='llm_code',strategy='none')
            result=evaluate(workflow.dataset,workflow.schema,candidate,None,cfg,split='select')
            candidates.append((result['metrics']['mse'],candidate))
            validation=None
        except (ValueError,RuntimeError,KeyError) as error:
            rejected=workflow.last_response or rejected
            validation={'error':str(error)}
            workflow._append('candidates.jsonl',{'decision':'rejected','response':rejected,'validation':validation})
    if not candidates:
        (workflow.path/'selection.json').write_text(json.dumps({'library_hash':None,'fallback':None,'status':'failed_full_code'}))
        raise RuntimeError('No complete code world model passed validation; no symbolic fallback was used')
    selected=min(candidates,key=lambda item:item[0])[1]
    (workflow.path/'selection.json').write_text(json.dumps({'library_hash':selected.hash,'status':'selected','fallback':None}))
    return selected
