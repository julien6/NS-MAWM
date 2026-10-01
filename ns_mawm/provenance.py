from __future__ import annotations
from dataclasses import replace
import json
from pathlib import Path
from .rules import compile_rule, digest


class RuleArchive:
    def __init__(self, directory):
        self.path = Path(directory)
        self.path.mkdir(parents=True, exist_ok=True)

    def add(self, rule, evidence_ids=(), decision="accepted"):
        record = {k: (sorted(v) if isinstance(v, frozenset) else v) for k,v in vars(rule).items()
                  if k not in ("guard", "effect", "derives")}
        fingerprint = digest({k: record[k] for k in ("code_hash", "reads", "writes", "stage", "scope")})
        directory = self.path / digest(rule.id)
        directory.mkdir(exist_ok=True)
        path = directory / f"v{rule.version}.json"
        if path.exists():
            old = json.loads(path.read_text())
            if old["fingerprint"] != fingerprint:
                raise ValueError("Changed guard/effect/reads/writes requires a new rule version")
        record.update(fingerprint=fingerprint, evidence_ids=list(evidence_ids), decision=decision)
        path.write_text(json.dumps(record, indent=2))
        return path

    def human_edit(self, rule, code, *, reads=None, writes=None, evidence_ids=()):
        record = {k:v for k,v in vars(rule).items() if k not in ("guard", "effect", "derives")}
        record.update(code=code, version=rule.version+1, base_version=rule.version, source="human_edited")
        if reads is not None: record["reads"] = reads
        if writes is not None: record["writes"] = writes
        edited = compile_rule(record)
        self.add(edited, evidence_ids)
        return edited
