from __future__ import annotations
from copy import deepcopy
import json
from pathlib import Path
from .rules import digest
from .training import train, evaluate


def select_lambda(dataset, schema, library, action_size, config, directory):
    if dataset.frozen is not None:
        raise PermissionError("Hyperparameters cannot change after test freeze")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    curve = []
    choices = (0., .1, .3, 1., 3.)
    best = None
    for value in choices:
        candidate = deepcopy(config)
        candidate["world_model"].update(strategy="regularization", **{"lambda": value})
        network, timing = train(dataset, schema, library, action_size, candidate)
        result = evaluate(dataset, schema, library, network, candidate, split="select")
        record = {"lambda": value, "mse25": result["metrics"]["mse"], "rvr": result["diagnostics"]["micro_rvr"],
                  "rule_rvr": {r["rule_id"]: r["rvr"] for r in result["diagnostics"]["rules"]}, "training": timing}
        curve.append(record)
        if best is None or record["mse25"] < best[0]:
            best = (record["mse25"], candidate)
    selected = {"curve": curve, "selected_lambda": best[1]["world_model"]["lambda"], "split_hash": dataset.hash,
                "library_hash": library.hash, "configuration": best[1]}
    (directory / "lambda-sweep.json").write_text(json.dumps(selected, indent=2))
    return selected


def version_trace(dataset, schema, final_library, versions, action_size, config):
    # All other final-library rules stay fixed in every version row.
    results = []
    libraries = [final_library.swapped(r) for r in versions]
    for rule, library in zip(versions, libraries):
        network, timing = train(dataset, schema, library, action_size, config)
        result = evaluate(dataset, schema, library, network, config, compared_libraries=libraries)
        results.append({"rule": rule.id, "version": rule.version, "library_hash": library.hash,
                        "note": "Full final library with only the traced rule swapped to this version",
                        "training": timing, "result": result})
    return results


def tune(dataset, schema, library, action_size, config, directory):
    """Identical selection allowance for every trainable prediction baseline."""
    from itertools import product
    if dataset.frozen is not None:
        raise PermissionError('Tuning is disabled after protocol freeze')
    path = Path(directory); path.mkdir(parents=True, exist_ok=True)
    options = config.get('tuning', {})
    mode = options.get('mode', 'baseline')
    if mode == 'lambda':
        return select_lambda(dataset, schema, library, action_size, config, path)
    if config['world_model'].get('backbone') in {'pswm_only','llm_code'}:
        result = {'applicable': False, 'reason': 'No trainable parameters', 'configuration': deepcopy(config), 'curve': []}
    else:
        if mode not in {'baseline','strategy'}:
            raise ValueError('tuning.mode must be baseline, lambda or strategy')
        changes = []
        if mode == 'baseline':
            for lr, width, length in product(options.get('learning_rates',[1e-4,3e-4,1e-3]), options.get('widths',[128,256]), options.get('sequence_lengths',[25,50])):
                changes.append({'sizes': {'enc':[width,width], 'lstm':width, 'g':[width,width], 'dec':[width,width]},
                                'optim': {'lr':lr,'seq_len':length}})
        else:
            for strategy in options.get('strategies',['projection','residual','regularization']):
                for coefficient in (options.get('lambdas',[0,.1,.3,1,3]) if strategy == 'regularization' else [0]):
                    changes.append({'strategy':strategy,'lambda':coefficient})
        curve = []
        for index, change in enumerate(changes):
            candidate = deepcopy(config)
            for key,value in change.items():
                if isinstance(value,dict):
                    candidate['world_model'].setdefault(key,{}).update(value)
                else:
                    candidate['world_model'][key]=value
            network,timing=train(dataset,schema,library,action_size,candidate)
            score=evaluate(dataset,schema,library,network,candidate,split='select')
            curve.append({'index':index,'configuration':candidate,'mse':score['metrics']['mse'],
                          'diagnostics':score['diagnostics'],'training':timing})
        winner=min(curve,key=lambda r:(r['mse'],r['index']))
        result={'applicable':True,'mode':mode,'configuration':winner['configuration'],'curve':curve,
                'split_hash':dataset.hash,'library_hash':library.hash,'configurations':len(curve)}
    (path/'selection.json').write_text(json.dumps(result,indent=2))
    return result


def freeze_union(dataset, schema, libraries, destination):
    """Declare comparison support on revision/selection contexts before test access."""
    from .training import episode_contexts
    from .rules import PSWM
    if dataset.frozen is not None:
        raise PermissionError('Comparison support must be fixed before test access')
    blocks=set()
    for library in libraries:
        for split in ('rev','select'):
            for episode in dataset.get(split,purpose='evaluate'):
                for _,out in episode_contexts(episode,schema,PSWM(library)):
                    blocks.update(out.provenance)
    result={'blocks':sorted(blocks),'libraries':sorted({lib.hash for lib in libraries}),
            'split_hash':dataset.hash,'source_splits':['rev','select']}
    result['hash']=digest(result)
    Path(destination).write_text(json.dumps(result,indent=2))
    return result
