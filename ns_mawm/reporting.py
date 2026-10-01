from __future__ import annotations
import csv
import json
from pathlib import Path


def metadata_notes(directory):
    directory = Path(directory)
    notes = []
    manifests = list(directory.rglob("manifest.json"))
    for manifest_path in manifests:
        m = json.loads(manifest_path.read_text())
        strategy = m["configuration"].get("world_model", {}).get("strategy")
        if strategy == "projection":
            notes.append({"run_id": m["run_id"], "note": "Projection trains the same neural architecture on the same data loss and initialization streams as the purely neural condition; pre-enforcement equality requires matching configuration, data and seed."})
        for path in manifest_path.parent.glob("evaluation-*.json"):
            evaluation = json.loads(path.read_text())
            for rule in evaluation["diagnostics"]["rules"]:
                if rule["structural"]:
                    notes.append({"run_id": m["run_id"], "rule_id": rule["rule_id"], "note": "RVR_out = 0 (structural: accepted assignments enforced by this library)."})
        notes.append({"run_id":m["run_id"],"architecture":m["configuration"].get("world_model",{}),
                      "hardware":m.get("hardware"),"benchmark_semantics":m.get("benchmarks")})
    for path in directory.rglob("calls.jsonl"):
        for line in path.read_text().splitlines():
            call=json.loads(line)
            notes.append({"call_id":call["call_id"],"source_tags":call["source_tags"],
                          "leakage_audit":call.get("leakage_audit"),"model_version":call.get("model_version")})
    return notes


def development_cost(directory):
    path = Path(directory) / "raw_runs.csv"
    groups = {}
    if path.exists():
        with path.open() as handle:
            for row in csv.DictReader(handle):
                if row["checkpoint"] != "final" or row["value"] in ("", "None"):
                    continue
                field = {"training.seconds":"final_training_seconds", "candidate_training_seconds":"candidate_training_seconds",
                         "llm_seconds":"llm_seconds", "llm_gpu_seconds":"llm_gpu_seconds", "llm_tokens":"tokens"}.get(row["metric"])
                if field:
                    group = groups.setdefault(row["env"], {})
                    group[field] = group.get(field, 0.) + float(row["value"])
    for group in groups.values():
        group["total_development_seconds"] = sum(group.get(k,0) for k in ("final_training_seconds","candidate_training_seconds","llm_seconds"))
    return groups


def figures(csv_path, output):
    # Figures, like tables, are derived exclusively from raw_runs.csv.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output = Path(output)
    with Path(csv_path).open() as f:
        rows = list(csv.DictReader(f))
    by_run = {}
    for row in rows:
        if row["checkpoint"] != "final" or row["value"] in ("", "None"):
            continue
        by_run.setdefault(row["run_id"], {})[row["metric"]] = row
    fig, axis = plt.subplots(figsize=(6, 5))
    for run, metrics in by_run.items():
        for name, row in metrics.items():
            if not name.startswith("rvr."):
                continue
            rule = name[4:]
            if "rdd."+rule not in metrics:
                continue
            support = int(row["support"] or 0)
            axis.scatter(float(metrics["rdd."+rule]["value"]), float(row["value"]),
                         s=max(8,min(200,support**.5)), alpha=.6)
    axis.set(xlabel="Rule–data disagreement (RDD)", ylabel="Rule violation rate (RVR)", xlim=(-.02,1.02), ylim=(-.02,1.02))
    fig.tight_layout()
    fig.savefig(output / "rule_diagnostics.svg")
    plt.close(fig)


def record_numbers(run, value, prefix):
    """Store diagnostic counts/timings as ordinary auditable measurement rows."""
    if isinstance(value,dict):
        for key,item in value.items():
            if key not in {'counterexamples','context','action','symbolic','neural','observed'}:
                record_numbers(run,item,prefix+'.'+str(key))
    elif isinstance(value,list):
        for index,item in enumerate(value):
            name=(item.get('rule_id',str(index))+'.v'+str(item.get('version',0))) if isinstance(item,dict) else str(index)
            record_numbers(run,item,prefix+'.'+name)
    elif value is None or isinstance(value,(int,float)):
        run.record(prefix,value,unit='count' if isinstance(value,int) else 'normalized')


def paper_tables(rows, summaries, comparisons, output):
    """All numeric table cells originate in raw_runs.csv seed-level records."""
    output=Path(output)
    index={(r['env'],r['config_hash'],r['metric']):r for r in summaries}
    groups=sorted({(r['env'],r['config_hash']) for r in summaries})
    metrics=['mse','accepted_coverage','proposed_coverage','covered_error','uncovered_error',
             'fixed_error','other_shared_error','final_return','normalized_auc','interactions_to_threshold']
    table=[]
    for env,condition in groups:
        row={'env':env,'config_hash':condition}
        for metric in metrics:
            record=index.get((env,condition,metric))
            row[metric]=record['mean'] if record else None
        row['significant_comparisons']=';'.join(t['right'] for t in comparisons if t['env']==env and t['left']==condition and t['adjusted_p']<.05)
        table.append(row)
    with (output/'main_table.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,['env','config_hash',*metrics,'significant_comparisons']);writer.writeheader();writer.writerows(table)
    return table


def record_llm_cost(run, workflow):
    training=0.
    path=workflow.path/'candidates.jsonl'
    if path.exists():
        training=sum(json.loads(line).get('training',{}).get('seconds',0) for line in path.read_text().splitlines())
    run.record('candidate_training_seconds',training,unit='seconds')
    run.record('llm_seconds',workflow.latency,unit='seconds')
    run.record('llm_tokens',workflow.tokens,unit='tokens')
    path=workflow.path/'calls.jsonl'
    gpu=sum(json.loads(line).get('gpu_seconds',0) for line in path.read_text().splitlines()) if path.exists() else 0
    run.record('llm_gpu_seconds',gpu,unit='gpu-seconds')
    flagged=sum(bool(json.loads(line).get('leakage_audit',{}).get('flagged')) for line in path.read_text().splitlines()) if path.exists() else 0
    run.record('leakage_flagged_calls',flagged,unit='count')
    human=sum(json.loads(p.read_text()).get('source')=='human_edited' for p in (workflow.path/'rules').rglob('v*.json'))
    run.record('human_edits',human,unit='count')
