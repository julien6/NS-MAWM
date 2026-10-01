from __future__ import annotations

from dataclasses import asdict
from collections import defaultdict
import html
import json
import torch

from .rules import Context, PSWM


def violation(schema, tensor, assignment):
    b = schema.by_name[assignment.block]
    v = schema.get(tensor, b)
    return (b.categories[int(v.argmax())] != assignment.value if b.kind == "categorical"
            else abs(float(v.item()) - assignment.value) > b.eval_tol)


def error_split(schema, prediction, target, reference_mask, fixed_blocks=None):
    errors = schema.losses(prediction, target, training=False)
    covered = schema.block_mask(reference_mask).bool()
    fixed = torch.tensor([b.name in fixed_blocks for b in schema.blocks], device=errors.device) if fixed_blocks is not None else torch.ones_like(covered)
    def average(mask):
        return float(errors[mask].mean()) if mask.any() else None
    return {"mse": float(errors.mean()), "coverage": float(covered.float().mean()),
            "covered_error": average(covered), "uncovered_error": average(~covered),
            "covered_contribution": float((errors * covered).mean()),
            "uncovered_contribution": float((errors * ~covered).mean()),
            "fixed_error": average(fixed)}


class Diagnostics:
    def __init__(self, reference: PSWM, *, enforced_hash=None, strategy="none", threshold=.1, max_examples=5, focal_agent=0):
        self.engine, self.schema = reference, reference.schema
        self.enforced_hash, self.strategy = enforced_hash, strategy
        self.threshold, self.max_examples, self.focal_agent = threshold, max_examples, focal_agent
        self.rows = {}
        for r in reference.library.rules:
            self.rows[(r.id, r.version)] = {"rule_id": r.id, "version": r.version, "scope": r.scope,
                "enabled": reference.library.enabled(r), "support": 0, "violations": 0, "pre_violations": 0, "data_violations": 0,
                "accepted": 0, "conflicted": 0, "abstained": 0, "applications": 0,
                "application_violations": 0, "counterexamples": [], "owners": {}}
        self.contexts = self.proposed = self.accepted = self.conflicted = 0
        self.scope_coverage = defaultdict(lambda: defaultdict(int))
        self.owner_coverage = defaultdict(lambda: defaultdict(int))
        self.pre_available = False
        self.union_support = self.union_violations = 0

    def add(self, context: Context, prediction, observed, *, pre=None, split=None, evidence_id=None):
        out = self.engine(context)
        self.contexts += 1
        self.proposed += len({a.block for a in out.proposals})
        self.accepted += len(out.provenance)
        self.conflicted += len(out.conflicts)
        self.pre_available |= pre is not None
        for scope in ("individual", "joint"):
            proposed = {a.block for a in out.proposals if self.rows[(a.rule_id,a.rule_version)]["scope"] == scope}
            self.scope_coverage[scope]["proposed"] += len(proposed)
            self.scope_coverage[scope]["accepted"] += len(proposed & out.provenance.keys())
        for owner in ("own", "other", "shared"):
            names = {b.name for b in self.schema.blocks if ("shared" if b.owner == "shared" else "own" if b.owner == self.focal_agent else "other") == owner}
            proposed = {a.block for a in out.proposals} & names
            self.owner_coverage[owner]["total"] += len(names)
            self.owner_coverage[owner]["proposed"] += len(proposed)
            self.owner_coverage[owner]["accepted"] += len(names & out.provenance.keys())
        self.union_support += len(out.provenance)
        accepted_proposals = {a.block: a for a in out.proposals if a.block in out.provenance}
        self.union_violations += sum(violation(self.schema, prediction, a) for a in accepted_proposals.values())
        grouped = defaultdict(list)
        for a in out.proposals:
            grouped[(a.rule_id, a.rule_version)].append(a)
        for key, counts in out.counters.items():
            row = self.rows[key]
            for kind in counts.values():
                for name in ("accepted", "conflicted", "abstained"):
                    row[name] += kind[name]
        for key, assignments in grouped.items():
            row = self.rows[key]
            row["applications"] += 1
            application_bad = False
            for a in assignments:
                vr, rd = violation(self.schema, prediction, a), violation(self.schema, observed, a)
                vp = violation(self.schema, pre, a) if pre is not None else False
                row["support"] += 1
                row["violations"] += int(vr)
                row["pre_violations"] += int(vp)
                row["data_violations"] += int(rd)
                application_bad |= vr
                owner = self.schema.by_name[a.block].owner
                group = "shared" if owner == "shared" else ("own" if owner == self.focal_agent else "other")
                stats = row["owners"].setdefault(group, {"support": 0, "violations": 0, "data_violations": 0})
                stats["support"] += 1
                stats["violations"] += int(vr)
                stats["data_violations"] += int(rd)
                if split == "rev" and (vr or rd) and len(row["counterexamples"]) < self.max_examples:
                    row["counterexamples"].append({"evidence_id": evidence_id, "split": "rev",
                        "rule_id": a.rule_id, "rule_version": a.rule_version, "context_origin": context.origin,
                        "guard": "TRUE", "assigned_blocks": [a.block], "symbolic": a.value,
                        "neural": self.schema.get(pre if pre is not None else prediction, self.schema.by_name[a.block]).tolist(),
                        "observed": self.schema.get(observed, self.schema.by_name[a.block]).tolist(),
                        "action": context.joint_action, "context": dict(context.observation)})
            row["application_violations"] += int(application_bad)
        return out

    def report(self):
        rows = []
        for raw in self.rows.values():
            r = dict(raw)
            n = r["support"]
            r.update(rvr=r["violations"] / n if n else None,
                     rdd=r["data_violations"] / n if n else None,
                     rvr_pre=r["pre_violations"] / n if n and self.pre_available and self.strategy != "residual" else None,
                     application_rvr=r["application_violations"] / r["applications"] if r["applications"] else None,
                     conflict_rate=r["conflicted"] / n if n else None)
            # Conflict-rejected proposals and approximate scalar composition can
            # disagree even under enforcement. Never falsely label these zeros.
            r["structural"] = bool(n and r["violations"] == 0 and r["accepted"] == n and
                                   self.strategy in ("projection", "residual") and
                                   self.enforced_hash == self.engine.library.hash)
            if n:
                high_v, high_d = r["rvr"] > self.threshold, r["rdd"] > self.threshold
                r["interpretation"] = {(False, False): "Model and rule agree on observed cases",
                    (True, False): "Model error, optimization difficulty or tolerance mismatch",
                    (False, True): "Model may have absorbed an unsuitable rule",
                    (True, True): "Inspect both model error and rule premises/effects"}[high_v, high_d]
            else:
                r["interpretation"] = "N/A: no support"
            rows.append(r)
        total = self.contexts * len(self.schema.blocks)
        support = sum(r["support"] for r in rows)
        supported = [r for r in rows if r["support"]]
        return {"library_hash": self.engine.library.hash, "contexts": self.contexts, "rules": rows,
                "proposed_coverage": self.proposed / total if total else None,
                "accepted_coverage": self.accepted / total if total else None,
                "conflict_fraction": self.conflicted / total if total else None,
                "micro_rvr": sum(r["violations"] for r in rows) / support if support else None,
                "macro_rvr": sum(r["rvr"] for r in supported) / len(supported) if supported else None,
                "union_mask_rvr": self.union_violations / self.union_support if self.union_support else None,
                "coverage_by_scope": {key: {k: v/total if total else None for k,v in counts.items()} for key,counts in self.scope_coverage.items()},
                "coverage_by_owner": {key: {k: v/counts["total"] if counts["total"] else None for k,v in counts.items() if k != "total"} for key,counts in self.owner_coverage.items()},
                "by_scope": {scope: {"support": sum(r["support"] for r in rows if r["scope"] == scope),
                    "violations": sum(r["violations"] for r in rows if r["scope"] == scope),
                    "accepted": sum(r["accepted"] for r in rows if r["scope"] == scope),
                    "conflicted": sum(r["conflicted"] for r in rows if r["scope"] == scope)}
                    for scope in ("individual", "joint")}}


def union_blocks(outputs):
    return {name for output in outputs for name in output.provenance}


def dashboard(reports, path):
    payload = html.escape(json.dumps(reports, indent=2))
    rows = []
    for model, report in reports.items():
        for r in report["rules"]:
            rows.append("<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in
                (model, r["rule_id"], r["version"], r["support"], r["rvr"], r["rdd"], r["conflict_rate"], r["structural"])) + "</tr>")
    path.write_text("<!doctype html><meta charset='utf-8'><title>Rule diagnostics</title>"
                    "<table><tr><th>Model</th><th>Rule</th><th>Version</th><th>Support</th><th>RVR</th><th>RDD</th><th>Conflicts</th><th>Structural</th></tr>"
                    + "".join(rows) + "</table><details><summary>Evidence</summary><pre>" + payload + "</pre></details>")
