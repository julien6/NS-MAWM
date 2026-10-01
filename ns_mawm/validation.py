from __future__ import annotations
from dataclasses import asdict
from .rules import PSWM
from .training import episode_contexts
from .diagnostics import Diagnostics


def library_card(library, dataset, *, max_rdd=.1, minimum_support=1):
    engine = PSWM(library)
    diagnostic = Diagnostics(engine)
    contexts = []
    for episode in dataset.get("rev", purpose="revision"):
        for t, (context, symbolic) in enumerate(episode_contexts(episode, library.schema, engine)):
            observed = episode.observations[t+1]
            diagnostic.add(context, observed, observed, split="rev", evidence_id=f"{episode.id}:{t}")
            if len(contexts) < 3:
                contexts.append(context)
    report = diagnostic.report()
    # This stage measures data consistency, not the RVR of any trained model.
    for row in report["rules"]:
        for key in ("rvr", "rvr_pre", "violations", "pre_violations", "structural", "interpretation", "application_rvr", "application_violations"):
            row.pop(key, None)
        row["validated"] = row["support"] >= minimum_support and row["rdd"] <= max_rdd
    for key in ("micro_rvr", "macro_rvr"):
        report.pop(key, None)
    report.update(scope_counts={scope: sum(r.scope == scope for r in library.rules) for scope in ("individual", "joint")},
                  blocks_written=sorted(set().union(*library.write_sets.values()) if library.write_sets else set()),
                  semantics=dataset.manifest.get("metadata", {}).get("semantics"),
                  validation=library.validate(contexts), split_hash=dataset.hash,
                  rdd_threshold=max_rdd, minimum_support=minimum_support)
    report["excluded"] = [r["rule_id"] for r in report["rules"] if not r["validated"]]
    return report
