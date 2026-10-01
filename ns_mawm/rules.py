from __future__ import annotations

import ast
import fnmatch
import hashlib
import json
import math
import random
import re
import signal
import threading
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from types import MappingProxyType
from typing import Callable, Mapping
import torch

from .schema import Schema


def matches(name, pattern):
    return re.fullmatch(re.escape(pattern).replace(r"\*", ".*"), name) is not None


def digest(obj: object) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


class Tri(Enum):
    TRUE = 1
    FALSE = 0
    UNKNOWN = -1


class FrozenMap(Mapping):
    def __init__(self, values):
        self._values = MappingProxyType({k: freeze(v) for k, v in values.items()})
    def __getitem__(self, key): return self._values[key]
    def __iter__(self): return iter(self._values)
    def __len__(self): return len(self._values)


def freeze(value):
    if type(value) in (str, int, float, bool, type(None), Fact):
        return value
    if isinstance(value, FrozenMap):
        return value
    if isinstance(value, Mapping):
        return FrozenMap(value)
    if isinstance(value, (list, tuple)):
        return tuple(freeze(v) for v in value)
    if isinstance(value, (str, int, float, bool, type(None), Fact)):
        return value
    raise TypeError("Only schema-level immutable values may enter rule contexts")


@dataclass(frozen=True)
class Fact:
    key: str
    value: float | str
    frame: str
    age: int
    source_step: int
    source_rule: str
    persistent: bool = False


@dataclass(frozen=True)
class Context:
    observation: Mapping[str, float | str]
    joint_action: tuple[int, ...]
    history: tuple = ()
    prev_joint_action: tuple[int, ...] | None = None
    memory: Mapping[str, Fact] = field(default_factory=dict)
    origin: str = "real"
    step: int = 0

    def __post_init__(self):
        if self.origin not in ("real", "imagined"):
            raise ValueError("Unknown context origin")
        if len(self.history) > 50:
            raise ValueError("Explicit history exceeds 50-step window")
        for k in ("observation", "joint_action", "history", "prev_joint_action", "memory"):
            object.__setattr__(self, k, freeze(getattr(self, k)))


@dataclass(frozen=True)
class Assignment:
    block: str
    value: float | str
    rule_id: str
    rule_version: int


@dataclass(frozen=True)
class RuleSpec:
    id: str
    version: int
    stage: int
    scope: str
    reads: frozenset[str]
    writes: frozenset[str]
    guard: Callable[[Context], Tri]
    effect: Callable[[Context], list[Assignment]]
    derives: Callable[[Context], list[Fact]] | None = None
    assumptions: tuple[str, ...] = ()
    source: str = "handcrafted"
    base_version: int | None = None
    code_hash: str = ""
    test_cases: tuple[dict, ...] = ()
    tags: tuple[str, ...] = ()
    code: str = ""


@dataclass
class PSWMOutput:
    target: torch.Tensor
    mask: torch.Tensor
    provenance: dict
    conflicts: list[dict]
    proposals: list[Assignment]
    derived_facts: list[Fact]
    memory: Mapping[str, Fact]
    counters: dict


@contextmanager
def time_limit(seconds: float):
    # Rule evaluation is a main-thread operation; refusing threads avoids silently
    # losing the CPU deadline. The namespace blocks I/O and allocation gadgets.
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError("Rule execution requires the main thread")
    old_handler = signal.getsignal(signal.SIGALRM)
    def expired(*_):
        raise TimeoutError("Rule exceeded its context time budget")
    signal.signal(signal.SIGALRM, expired)
    old_timer = signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, *old_timer)
        signal.signal(signal.SIGALRM, old_handler)


SAFE_CALLS = {"abs": abs, "min": min, "max": max, "len": len, "float": float,
              "int": int, "bool": bool, "enumerate": enumerate, "zip": zip,
              "all": all, "any": any, "sum": sum, "Assignment": Assignment,
              "Fact": Fact, "Tri": Tri}
SAFE_ATTRS = {"observation", "joint_action", "history", "prev_joint_action", "memory",
              "origin", "step", "get", "items", "keys", "values", "TRUE", "FALSE",
              "UNKNOWN", "value", "frame", "age", "source_step", "source_rule", "persistent"}


def compile_rule(record: dict) -> RuleSpec:
    """Load a deliberately small Python subset; no imports, mutation or indirect calls."""
    code = record["code"]
    tree = ast.parse(code)
    banned = (ast.Import, ast.ImportFrom, ast.While, ast.ClassDef, ast.With, ast.Try,
              ast.Raise, ast.Global, ast.Nonlocal, ast.Lambda, ast.Delete, ast.AsyncFunctionDef,
              ast.Await, ast.Yield, ast.YieldFrom, ast.NamedExpr)
    for node in ast.walk(tree):
        if isinstance(node, banned):
            raise ValueError(f"Forbidden rule construct: {type(node).__name__}")
        if isinstance(node, ast.Name) and node.id.startswith("_"):
            raise ValueError("Private names are forbidden")
        if isinstance(node, ast.Attribute) and node.attr not in SAFE_ATTRS:
            raise ValueError(f"Forbidden attribute: {node.attr}")
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(not isinstance(t, (ast.Name, ast.Tuple)) for t in targets):
                raise ValueError("Rule contexts cannot be mutated")
        if isinstance(node, ast.Call):
            f = node.func
            if not ((isinstance(f, ast.Name) and f.id in SAFE_CALLS) or
                    (isinstance(f, ast.Attribute) and f.attr in {"get", "items", "keys", "values"})):
                raise ValueError("Only allow-listed calls are permitted")
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Pow, ast.MatMult)):
            raise ValueError("Unbounded arithmetic is forbidden")
        if isinstance(node, ast.Constant) and isinstance(node.value, (str, int)) and len(str(node.value)) > 10000:
            raise ValueError("Oversized constant")
    if any(not isinstance(n, ast.FunctionDef) or n.name not in {"guard", "effect", "derives"} for n in tree.body):
        raise ValueError("Rule source contains only guard/effect/derives functions")
    from .sandbox import bind
    namespace = {n.name: bind(n, SAFE_CALLS, SAFE_ATTRS) for n in tree.body}
    if not {"guard", "effect"} <= namespace.keys():
        raise ValueError("Exact guard and effect required")
    kwargs = {k: v for k, v in record.items() if k in RuleSpec.__dataclass_fields__}
    kwargs.update(guard=namespace["guard"], effect=namespace["effect"], derives=namespace.get("derives"), code_hash=digest(code))
    for key in ("reads", "writes"):
        kwargs[key] = frozenset(kwargs[key])
    for key in ("assumptions", "test_cases", "tags"):
        kwargs[key] = tuple(kwargs.get(key, ()))
    return RuleSpec(**kwargs)


class Library:
    def __init__(self, rules: list[RuleSpec], schema: Schema, *, edges=(), policy="block_rejection", parent=None, creator="human", rule_filter=None):
        self.schema, self.parent, self.creator = schema, parent, creator
        self.policy, self.edges = policy, tuple(tuple(e) for e in edges)
        self.rule_filter = {"scope": ["individual", "joint"], "disable_ids": [], **(rule_filter or {})}
        if policy not in ("block_rejection", "reject_library"):
            raise ValueError("Unknown conflict policy")
        self.rules = tuple(sorted(rules, key=lambda r: r.id))
        self.write_sets = {r.id: frozenset(n for n in schema.by_name if n in r.writes or any(matches(n, p) for p in r.writes if "*" in p)) for r in rules}
        if len({r.id for r in rules}) != len(rules):
            raise ValueError("A library contains one version per rule id")
        for r in rules:
            if r.scope not in ("individual", "joint") or r.version < 0:
                raise ValueError("Invalid rule metadata")
            if not r.writes:
                raise ValueError("Inequality-only rules are not PSTRs")
            for name in r.reads:
                if not name.startswith("memory:") and name not in schema.by_name:
                    raise ValueError(f"Unknown read block {name}")
            for pattern in r.writes:
                if pattern not in schema.by_name and not any(matches(n, pattern) for n in schema.by_name):
                    raise ValueError(f"Unknown write pattern {pattern}")
        pending = {r.stage for r in rules} | {v for e in self.edges for v in e}
        self.stages = []
        while pending:
            ready = sorted(s for s in pending if not any(b == s and a in pending for a, b in self.edges))
            if not ready:
                raise ValueError("Cyclic stage DAG")
            self.stages.extend(ready)
            pending.difference_update(ready)
        self.hash = digest(self.manifest())

    def enabled(self, r):
        f = self.rule_filter
        return ((not f.get("ids") or r.id in f["ids"]) and r.id not in f.get("disable_ids", ()) and r.scope in f.get("scope", ("individual", "joint"))
                and (not f.get("tags") or bool(set(r.tags) & set(f["tags"]))))

    def manifest(self):
        return {"rules": [{"id": r.id, "version": r.version, "code_hash": r.code_hash or digest(r.code),
                           "reads": sorted(r.reads), "writes": sorted(r.writes), "stage": r.stage,
                           "scope": r.scope, "tags": r.tags} for r in self.rules],
                "edges": self.edges, "policy": self.policy, "parent": self.parent,
                "creator": self.creator, "filter": self.rule_filter, "schema": self.schema.to_dict()}

    def filtered(self, **rule_filter):
        return Library(list(self.rules), self.schema, edges=self.edges, policy=self.policy,
                       parent=self.parent, creator=self.creator, rule_filter=rule_filter)

    def swapped(self, rule: RuleSpec):
        return Library([r for r in self.rules if r.id != rule.id] + [rule], self.schema,
                       edges=self.edges, policy=self.policy, parent=self.hash, creator=self.creator)

    def validate(self, contexts=(), *, require_cases=False):
        failures = []
        engine = PSWM(self)
        for r in self.rules:
            if require_cases and {c["guard"] for c in r.test_cases} != {"TRUE", "FALSE", "UNKNOWN"}:
                failures.append({"rule": r.id, "error": "Applicable, inapplicable and unknown cases required"})
            for case in r.test_cases:
                try:
                    c = Context(**case["context"])
                    with time_limit(.05):
                        if r.guard(c).name != case["guard"]:
                            raise ValueError("Guard test failed")
                    result = PSWM(Library([r], self.schema))(c)
                    if "assignments" in case and {a.block: a.value for a in result.proposals} != case["assignments"]:
                        raise ValueError("Effect test failed")
                except Exception as exc:
                    failures.append({"rule": r.id, "error": str(exc)})
        for c in contexts:
            try:
                expected = engine(c)
                for seed in range(3):
                    order = list(self.rules)
                    random.Random(seed).shuffle(order)
                    actual = engine(c, order=order)
                    if not torch.equal(expected.target, actual.target) or not torch.equal(expected.mask, actual.mask):
                        raise ValueError("Rule order changed acceptance")
            except Exception as exc:
                failures.append({"error": str(exc)})
        return {"valid": not failures, "failures": failures, "library_hash": self.hash}


class PSWM:
    def __init__(self, library: Library, max_age=25, timeout=.05):
        self.library, self.schema = library, library.schema
        self.max_age, self.timeout = max_age, timeout

    def __call__(self, context: Context, *, order=None) -> PSWMOutput:
        with time_limit(self.timeout):
            return self._execute(context, order=order)

    def _execute(self, context: Context, *, order=None) -> PSWMOutput:
        memory = {k: replace(f, age=f.age + 1) for k, f in context.memory.items()
                  if f.persistent and f.age + 1 <= self.max_age and
                  (f.frame == "world" or context.prev_joint_action is None or not any(context.prev_joint_action)) and
                  (k not in context.observation or context.observation[k] == f.value)}
        for name, value in context.observation.items():
            memory[name] = Fact(name, value, "observation", 0, context.step, "observation", False)
        proposals, derived, counters = [], [], {}
        rules = self.library.rules if order is None else order
        for stage in self.library.stages:
            stage_facts = []
            snapshots = {}
            for r in rules:
                if r.stage != stage or not self.library.enabled(r):
                    continue
                # Read restrictions are enforced, not just documented.
                snapshot = snapshots.get(r.reads)
                if snapshot is None:
                    obs = {k: v for k, v in context.observation.items() if k in r.reads}
                    mem = {k: f for k, f in memory.items() if "memory:" + k in r.reads}
                    history = tuple({k: v for k, v in h.items() if k in r.reads} for h in context.history)
                    snapshot = replace(context, observation=obs, history=history, memory=mem)
                    snapshots[r.reads] = snapshot
                counts = {kind: {k: 0 for k in ("proposed", "accepted", "conflicted", "abstained")}
                          for kind in ("scalar", "categorical")}
                counters[(r.id, r.version)] = counts
                with nullcontext():
                    guard = r.guard(snapshot)
                    if not isinstance(guard, Tri):
                        raise TypeError("Guard must return Tri")
                    effects = r.effect(snapshot) if guard is Tri.TRUE else []
                    seen = set()
                    for a in effects:
                        if not isinstance(a, Assignment) or (a.rule_id, a.rule_version) != (r.id, r.version):
                            raise ValueError("Assignment provenance mismatch")
                        if a.block in seen or a.block not in self.library.write_sets[r.id]:
                            raise ValueError("Duplicate or undeclared assignment")
                        self.schema.validate_value(a.block, a.value)
                        seen.add(a.block)
                        counts[self.schema.by_name[a.block].kind]["proposed"] += 1
                        proposals.append(a)
                    for name in self.library.write_sets[r.id] - seen:
                        counts[self.schema.by_name[name].kind]["abstained"] += 1
                    if guard is Tri.TRUE and r.derives:
                        facts = r.derives(snapshot)
                        if any(not isinstance(f, Fact) or f.source_rule != r.id or f.source_step != context.step or f.age != 0 for f in facts):
                            raise ValueError("Invalid derived-fact provenance")
                        stage_facts.extend(facts)
            for key in sorted({f.key for f in stage_facts}):
                fs = [f for f in stage_facts if f.key == key]
                if len({(f.value, f.frame) for f in fs}) == 1:
                    f = min(fs, key=lambda f: f.source_rule)
                    memory[key] = f
                    derived.append(f)
                else:
                    memory.pop(key, None)
        accepted, provenance, conflicts = {}, {}, []
        by_block = {}
        for assignment in proposals:
            by_block.setdefault(assignment.block, []).append(assignment)
        for name in sorted(by_block):
            group = sorted(by_block[name], key=lambda a: (a.rule_id, a.rule_version))
            b = self.schema.by_name[name]
            agree = (len({a.value for a in group}) == 1 if b.kind == "categorical" else
                     max(a.value for a in group) - min(a.value for a in group) <= b.comp_tol)
            for a in group:
                counters[(a.rule_id, a.rule_version)][b.kind]["accepted" if agree else "conflicted"] += 1
            if agree:
                accepted[name] = group[0].value
                provenance[name] = [(a.rule_id, a.rule_version) for a in group]
            else:
                conflicts.append({"block": name, "proposals": [asdict(a) for a in group]})
        if conflicts and self.library.policy == "reject_library":
            raise ValueError("Library rejected due to conflicting assignments")
        mask = torch.zeros(self.schema.shape)
        rows, columns = [], []
        for name in accepted:
            b = self.schema.by_name[name]
            rows.extend(range(*b.slice))
            columns.extend([b.col] * (b.slice[1] - b.slice[0]))
        if rows:
            mask[rows, columns] = 1
        return PSWMOutput(self.schema.encode(accepted, partial=True), mask, provenance, conflicts,
                          proposals, derived, freeze(memory), counters)

    def batch(self, contexts):
        return [self(c) for c in contexts]

    def predict(self, context, fallback=None):
        out = self(context)
        base = self.schema.encode(context.observation) if fallback is None else fallback
        return torch.where(out.mask.bool(), out.target, base)
