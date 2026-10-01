"""Serial, dependency-aware execution. No scheduler or database is needed."""
from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
from .rules import digest
from .artifacts import file_hash, code_identity


def build_jobs(base):
    from .experiments import plan
    runs=plan(base)['runs']
    wanted=base.get('campaign',{}).get('conditions')
    if wanted:
        runs=[c for c in runs if c['run']['condition'] in wanted]
    root=Path(base.get('output','outputs/campaign')).resolve()
    jobs=[]; collections={}; terminals=[]; branches=[]
    for variant in runs:
        cfg=deepcopy(variant); cfg.pop('campaign',None)
        protocol=deepcopy(cfg); protocol['run'].pop('seed',None)
        protocol.pop('data',None); protocol.pop('output',None)
        cfg['protocol_hash']=digest(protocol)
        identity=digest(cfg)[:16]
        cfg['output']=str(root/'runs'/identity)
        source=digest([cfg['run']['env'],cfg.get('environment',{}),cfg.get('collection',{})])[:16]
        if source not in collections:
            collect=deepcopy(cfg); collect['data']=str(root/'datasets'/source)
            collect['output']=str(root/'collection'/source)
            jid='collect-'+source
            jobs.append({'id':jid,'stage':'collect','depends':[],'config':collect})
            collections[source]=jid
        cfg['data']=str(root/'protocols'/identity)
        prepare='prepare-'+identity
        jobs.append({'id':prepare,'stage':'prepare','depends':[collections[source]],'config':cfg,'source':str(root/'datasets'/source)})
        last=prepare
        if cfg['run'].get('experiment') == 'control':
            terminals.append(last); branches.append((identity,last)); continue
        tuning='tune-'+identity
        jobs.append({'id':tuning,'stage':'tune','depends':[last],'configuration_from':last,'config':{}})
        last=tuning
        label=cfg['run']['condition']
        if label=='B6' or (cfg['world_model'].get('strategy')=='regularization' and not label.startswith('A7')):
            mode='strategy' if label=='B6' else 'lambda'
            jid=mode+'-'+identity
            jobs.append({'id':jid,'stage':'tune','depends':[last],'configuration_from':last,'config':{'tuning':{'mode':mode}}})
            last=jid
        if cfg['run'].get('experiment')=='refinement' or label=='B5':
            jid='rules-'+identity
            jobs.append({'id':jid,'stage':'generate-code' if label=='B5' else 'refine','depends':[last],'configuration_from':last,'config':{}})
            last=jid
        terminals.append(last); branches.append((identity,last))
    # All selection must finish before any test protocol is opened.
    for identity,last in branches:
        if base['run'].get('experiment') == 'control':
            jobs.append({'id':'control-'+identity,'stage':'control','depends':list(terminals),'configuration_from':last,'config':{}})
            continue
        train='train-'+identity
        jobs.append({'id':train,'stage':'train','depends':list(terminals),'configuration_from':last,'config':{}})
        freeze='freeze-'+identity
        jobs.append({'id':freeze,'stage':'freeze','depends':[train,*terminals],'configuration_from':train,'comparison_from':terminals,'config':{}})
        jobs.append({'id':'evaluate-'+identity,'stage':'evaluate','depends':[freeze],'configuration_from':freeze,'checkpoint_from':train,'config':{}})
    return jobs


def merge(left,right):
    result=deepcopy(left)
    for key,value in right.items():
        result[key]=merge(result.get(key,{}),value) if isinstance(value,dict) else deepcopy(value)
    return result


def execute(job, config, results, directory):
    from .cli import main
    from .configuration import validate_config
    stage=job['stage']; cfg=deepcopy(config)
    if stage=='prepare':
        source=Path(job['source']); target=Path(cfg['data']); target.mkdir(parents=True,exist_ok=True)
        for path in source.iterdir():
            if path.is_file() and path.name != 'frozen.json':
                dest=target/path.name
                if dest.exists() and file_hash(dest) != file_hash(path):
                    if (target/'frozen.json').exists(): raise ValueError('Frozen dataset inputs changed; use a new campaign output')
                    dest.unlink()
                if not dest.exists():
                    if path.suffix=='.pt': os.link(path,dest)
                    else: dest.write_bytes(path.read_bytes())
        return {'configuration':cfg,'artifacts':[str(p) for p in target.iterdir() if p.name != 'frozen.json']}
    validate_config(cfg,stage)
    output=Path(cfg.get('output',directory)); output.mkdir(parents=True,exist_ok=True)
    if stage=='freeze' and cfg.get('evaluation',{}).get('fixed_block_set'):
        from .cli import load_components
        from .libraries import load_library
        from .tuning import freeze_union
        env,data,schema,library=load_components(cfg)
        try:
            from .libraries import save_library
            paths=list(cfg.get('evaluation',{}).get('compared_libraries',[]))
            for jid in job.get('comparison_from',[]):
                other=results[jid]['configuration']
                if other['run']['env'] != cfg['run']['env'] or other.get('environment',{}) != cfg.get('environment',{}): continue
                other_env, other_data, other_schema, other_lib=load_components(other)
                try:
                    if other_data.hash != data.hash or other_schema.to_dict() != schema.to_dict(): continue
                    destination=output/('comparison-'+other_lib.hash+'.json')
                    save_library(other_lib,destination); paths.append(str(destination))
                finally: other_env.close()
            cfg.setdefault('evaluation',{})['compared_libraries']=sorted(set(paths))
            libraries=[library]+[load_library(p,schema) for p in cfg.get('evaluation',{}).get('compared_libraries',[])]
            union=output/'fixed-union.json'
            if data.frozen is None: freeze_union(data,schema,libraries,union)
            cfg.setdefault('evaluation',{})['fixed_union_hash']=json.loads(union.read_text())['hash']
            cfg.setdefault('evaluation',{})['fixed_union']=str(union)
        finally: env.close()
    path=Path(directory)/(job['id']+'.yaml')
    import yaml
    path.write_text(yaml.safe_dump(cfg))
    argv=[stage,'--config',str(path)]
    if stage=='evaluate':
        argv+=['--split','test']
        if cfg['world_model'].get('backbone') not in {'pswm_only','llm_code'}:
            checkpoint=results[job['checkpoint_from']].get('checkpoint')
            if not checkpoint: raise ValueError('Evaluation requires a trained checkpoint')
            argv+=['--checkpoint',checkpoint]
    main(argv)
    artifacts=[]; result={}
    if stage=='collect': artifacts=list(Path(cfg['data']).glob('*'))
    elif stage=='tune':
        selection=output/('lambda-sweep.json' if cfg.get('tuning',{}).get('mode')=='lambda' else 'selection.json')
        cfg=json.loads(selection.read_text())['configuration']; artifacts=[selection]
    elif stage in {'refine','generate-code'}:
        selection=output/'llm'/'selection.json'; chosen=json.loads(selection.read_text())
        if chosen['library_hash']:
            cfg.setdefault('rules',{})['library']=str(output/'selected_library.json')
        elif chosen.get('fallback')=='purely_neural':
            cfg['world_model'].update(backbone='jopm_lstm',strategy='none')
        else: raise RuntimeError('Rule generation failed')
        artifacts=list((output/'llm').rglob('*'))+[output/'selected_library.json'] if chosen['library_hash'] else list((output/'llm').rglob('*'))
    elif stage=='freeze': artifacts=[Path(cfg['data'])/'frozen.json']+list(output.glob('fixed-union.json'))
    else:
        artifacts=[p for p in output.glob('*/*') if p.name.startswith(('evaluation-','update-','control.','collection-policy','mamba-'))]
        if stage=='train':
            checkpoints=list(output.glob('*/update-*.pt'))
            if checkpoints: result['checkpoint']=str(max(checkpoints,key=lambda p:int(p.stem.split('-')[-1])))
    return {**result,'configuration':cfg,'artifacts':[str(p) for p in artifacts if p.is_file()]}


def run_campaign(config, executor=execute):
    root=Path(config.get('output','outputs/campaign')).resolve(); root.mkdir(parents=True,exist_ok=True)
    explicit=config.get('campaign',{}).get('jobs')
    if not explicit:
        from .experiments import plan
        planned=plan(config,seconds_per_update=config.get('seconds_per_update'))
        (root/'plan.json').write_text(json.dumps(planned,indent=2))
    jobs=explicit or build_jobs(config)
    ids=[j['id'] for j in jobs]
    if len(ids)!=len(set(ids)): raise ValueError('Duplicate campaign job ids')
    known=set(); ordered=[]; pending=list(jobs)
    while pending:
        ready=[j for j in pending if set(j.get('depends',[])) <= known]
        if not ready: raise ValueError('Unknown dependency or cycle in campaign')
        for job in ready:
            if job.get('configuration_from') and job['configuration_from'] not in job.get('depends',[]):
                raise ValueError('configuration_from must be a dependency')
            ordered.append(job); known.add(job['id']); pending.remove(job)
    state_path=root/'campaign.json'
    old=json.loads(state_path.read_text()) if state_path.exists() else {}
    results={}; identity=code_identity(Path(__file__).resolve().parents[1])
    for job in ordered:
        deps=job.get('depends',[])
        cfg=merge(results[job['configuration_from']]['configuration'],job.get('config',{})) if job.get('configuration_from') and results[job['configuration_from']].get('configuration') else job.get('config',{})
        signature=digest([job,cfg,identity,input_hashes(cfg,job['stage']),{d:{'signature':results[d].get('signature'),'artifacts':results[d].get('artifacts')} for d in deps}])
        prior=old.get(job['id'],{})
        if any(results[d]['status']!='completed' for d in deps):
            result={'status':'blocked','reason':'Failed dependency','signature':signature}
        elif prior.get('status')=='completed' and prior.get('signature')==signature and prior.get('artifacts') and all(Path(p).is_file() and file_hash(p)==h for p,h in prior['artifacts'].items()) and config.get('campaign',{}).get('resume',True):
            result=prior
        else:
            try:
                result=executor(job,cfg,results,root)
                result['artifacts']={str(Path(p).resolve()):file_hash(p) for p in result.get('artifacts',[])}
                result.update(status='completed',signature=signature)
            except Exception as error:
                result={'status':'failed','error':str(error),'signature':signature}
        results[job['id']]=result
        temporary=state_path.with_suffix('.tmp'); temporary.write_text(json.dumps(results,indent=2)); temporary.replace(state_path)
    consolidate(root)
    return results


def consolidate(root):
    """Paper input includes test evaluations, control results and cost records only."""
    import csv
    from .artifacts import FIELDS
    root=Path(root); records={}
    eligible=set()
    for path in root.rglob('manifest.json'):
        manifest=json.loads(path.read_text())
        kind=manifest.get('configuration',{}).get('run',{}).get('experiment')
        if manifest.get('evaluation_split')=='test' or kind in {'control','llm_cost'}:
            eligible.add(manifest['run_id'])
    for path in root.rglob('raw_runs.csv'):
        if path.parent==root:continue
        with path.open() as handle:
            for row in csv.DictReader(handle):
                if row['run_id'] in eligible or row['metric'].startswith('training.'):
                    records[(row['run_id'],row['checkpoint'],row['metric'])]=row
    destination=root/'raw_runs.csv'
    with destination.open('w',newline='') as handle:
        writer=csv.DictWriter(handle,FIELDS);writer.writeheader();writer.writerows(records.values())
    return destination


def input_hashes(config, stage):
    paths=[]
    if stage not in {'collect','prepare'} and config.get('data'):
        root=Path(config['data'])
        paths.extend(root.glob('*.pt'))
        paths.extend(root/name for name in ('schema.json','splits.json'))
    rules=config.get('rules',{}); evaluation=config.get('evaluation',{})
    paths.extend(Path(p) for p in [rules.get('library'),rules.get('diagnostic_library'),
        evaluation.get('fixed_union'),config.get('collection',{}).get('policy_checkpoint')] if p)
    paths.extend(Path(p) for p in evaluation.get('compared_libraries',[]))
    paths.extend(Path(p) for p in rules.get('trace_versions',[]))
    return {str(p.resolve()):file_hash(p) if p.is_file() else None for p in paths}
