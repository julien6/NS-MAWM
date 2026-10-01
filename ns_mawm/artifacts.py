from __future__ import annotations

import csv
import fcntl
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import re
import subprocess
import tempfile
import zipfile
import torch
from .rules import digest
from .anonymize import sanitize_binary, notebook

FIELDS = "run_id experiment env config_hash library_hash split_hash seed checkpoint metric value support unit".split()


def now():
    return datetime.now(timezone.utc).isoformat()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def code_identity(root):
    commit = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    # A dirty working tree must not impersonate its parent commit.
    files = sorted({p for directory in ("ns_mawm", "vGridcraft", "BenchMARL/benchmarl", "configs")
                    for p in (Path(root)/directory).rglob("*") if p.is_file() and p.suffix in {".py", ".yaml", ".json"}} |
                   {p for p in (Path(root)/"pyproject.toml", Path(root)/"requirements-ns-mawm.txt") if p.exists()})
    import importlib.metadata
    versions = {}
    for name in ("torch", "numpy", "scipy", "torchrl", "tensordict", "mpe2", "smacv2"):
        try: versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: versions[name] = None
    return {"commit": commit, "source_hash": digest({str(p.relative_to(root)): file_hash(p) for p in files}),
            "dependencies": versions}



class Run:
    def __init__(self, directory, configuration, library, dataset, seed, code):
        group_config = json.loads(json.dumps(configuration))
        group_config.get("run", {}).pop("seed", None)
        for field in ("data", "output", "campaign", "verify"):
            group_config.pop(field, None)
        for key in ("library", "diagnostic_library"):
            group_config.get("rules", {}).pop(key, None)
        for key in ("fixed_union", "fixed_union_hash", "compared_libraries"):
            group_config.get("evaluation", {}).pop(key, None)
        self.config_hash = configuration.get("protocol_hash", digest(group_config))
        self.id = digest([configuration, library.hash, dataset.hash, seed, code])
        self.path = Path(directory) / self.id
        self.path.mkdir(parents=True, exist_ok=True)
        self.manifest = {"run_id": self.id, "configuration": configuration, "config_hash": self.config_hash,
            "library_hash": library.hash, "split_hash": dataset.hash, "seeds": {"initialization": seed, "sampling": seed + 10000, "environment": seed + 20000},
            "code": code, "benchmarks": dataset.manifest.get("metadata"), "hardware": {
                "platform": platform.platform(), "torch": torch.__version__, "cuda": torch.version.cuda,
                "device": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu"},
            "start": now(), "dataset": dataset.summary()}
        existing = self.path / "manifest.json"
        if existing.exists():
            previous = json.loads(existing.read_text())
            if previous["run_id"] != self.id:
                raise ValueError("Run manifest identity mismatch")
            self.manifest = previous
        self.csv = Path(directory) / "raw_runs.csv"
        self._recorded = set()
        if self.csv.exists():
            with self.csv.open() as f:
                self._recorded = {(r["run_id"], r["checkpoint"], r["metric"]) for r in csv.DictReader(f)}
        self.write_manifest()

    def record(self, metric, value, checkpoint="final", support=None, unit="normalized"):
        m = self.manifest
        row = {"run_id": self.id, "experiment": m["configuration"].get("run", {}).get("experiment", "prediction"),
               "env": m["configuration"]["run"]["env"], "config_hash": self.config_hash, "library_hash": m["library_hash"],
               "split_hash": m["split_hash"], "seed": m["seeds"]["initialization"], "checkpoint": checkpoint,
               "metric": metric, "value": value, "support": support, "unit": unit}
        key = (self.id, str(checkpoint), metric)
        with self.csv.open("a+", newline="") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            if key in self._recorded:
                f.seek(0)
                rows = [r for r in csv.DictReader(f) if (r["run_id"], r["checkpoint"], r["metric"]) != key]
                f.seek(0)
                f.truncate()
                writer = csv.DictWriter(f, FIELDS)
                writer.writeheader()
                writer.writerows(rows)
                writer.writerow(row)
            else:
                f.seek(0, 2)
                writer = csv.DictWriter(f, FIELDS)
                if f.tell() == 0:
                    writer.writeheader()
                writer.writerow(row)
            f.flush()
            fcntl.flock(f, fcntl.LOCK_UN)
        self._recorded.add(key)

    def write_manifest(self):
        (self.path / "manifest.json").write_text(json.dumps(self.manifest, indent=2))

    def finish(self):
        self.manifest["end"] = now()
        self.manifest["outputs"] = {str(p.relative_to(self.path)): file_hash(p)
            for p in self.path.rglob("*") if p.is_file() and p.name != "manifest.json"}
        self.write_manifest()


def export_anonymous(source, destination, *, deny, replacements, benchmark_commits):
    if not benchmark_commits or any(not re.fullmatch(r"[0-9a-f]{40}", str(v)) for v in benchmark_commits.values()):
        raise ValueError("Every external benchmark must have a pinned commit")
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source == destination or source in destination.parents:
        raise ValueError("Export destination must be outside the source tree")
    if not deny or any(not s for s in deny):
        raise ValueError("An explicit nonempty anonymity deny-list is required")
    files = []
    failures = []
    if (source / "pyproject.toml").exists():
        # A source export must retain the package, not just its result tables.
        required = ["pyproject.toml", "ns_mawm/__init__.py", "ns_mawm/cli.py"]
        missing = [p for p in required if not (source / p).is_file()]
        if missing:
            raise ValueError("Source export is not runnable; missing: " + ", ".join(missing))
        if "ns_mawm" not in benchmark_commits:
            raise ValueError("Source export requires the NS-MAWM source commit as well as benchmark pins")
    skipped = {".git", ".venv", "__pycache__", ".pytest_cache", ".mypy_cache"}
    for path in source.rglob("*"):
        if not path.is_file() or skipped.intersection(path.relative_to(source).parts):
            continue
        if path.is_symlink():
            raise ValueError(f"Symlinks are not exported: {path.relative_to(source)}")
        relative = str(path.relative_to(source))
        raw = path.read_bytes()
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            raw, evidence = sanitize_binary(path, replacements)
            for word in deny:
                if word.casefold() in (relative + evidence).casefold():
                    failures.append({"path": relative, "match": word})
            files.append((relative, raw))
            continue
        if path.suffix == ".ipynb":
            text = notebook(text)
        for old, new in replacements.items():
            text, relative = text.replace(old, new), relative.replace(old, new)
        text = re.sub(r"/home/[^/\s\"']+", "/home/anonymous", text)
        text = re.sub(r"(?im)^.*(?:git@github\.com|remote\.origin\.url|^Author:).*$", "", text)
        for word in [*deny, "commit remains to be pinned", "<pinned>"]:
            if word.casefold() in (relative + text).casefold():
                failures.append({"path": relative, "match": word})
        files.append((relative, text.encode()))
    if failures:
        raise ValueError("Anonymity audit failed: " + json.dumps(failures))
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".zip", delete=False) as tmp:
        temp = Path(tmp.name)
    try:
        with zipfile.ZipFile(temp, "w", zipfile.ZIP_DEFLATED) as archive:
            for name, raw in files:
                archive.writestr(name, raw)
            archive.writestr("benchmark_commits.json", json.dumps(benchmark_commits))
        temp.replace(destination)
    finally:
        temp.unlink(missing_ok=True)
