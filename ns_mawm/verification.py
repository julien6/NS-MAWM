"""Executable acceptance evidence; missing resources never count as passing."""
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from .artifacts import file_hash,code_identity

ROOT=Path(__file__).resolve().parents[1]


def benchmark_identity(distribution, expected_commit=None, expected_version=None):
    try:
        dist=importlib.metadata.distribution(distribution)
    except importlib.metadata.PackageNotFoundError:
        return {'status':'unverified','reason':f'{distribution} is not installed'}
    direct=json.loads(dist.read_text('direct_url.json') or '{}')
    commit=direct.get('vcs_info',{}).get('commit_id')
    result={'distribution':distribution,'version':dist.version,'commit':commit}
    if expected_version and dist.version!=expected_version:
        return {**result,'status':'failed','reason':'Installed version differs from pin'}
    if expected_commit and commit!=expected_commit:
        return {**result,'status':'unverified' if not commit else 'failed','reason':'Installed commit cannot be verified' if not commit else 'Installed commit differs from pin'}
    return {**result,'status':'passed'}


def preflight(config):
    import os
    identities={'mpe2':benchmark_identity('mpe2','147a8e88b669de9f0b06d1947c44d5c380c570bb','1.1.1'),
      'smacv2':benchmark_identity('smacv2','577ab5a2cff2391f8df582da5731ea9cd6adf3c6')}
    result=subprocess.run(['git','-C',str(ROOT/'Overcooked_AI'),'rev-parse','HEAD'],capture_output=True,text=True)
    identities['overcooked']={'status':'passed' if result.stdout.strip()=='3935fd4d4362502b45e38e1a5a54bfdad06e32ed' else 'unverified','commit':result.stdout.strip()}
    identities['starcraft_client']={'status':'available' if Path(os.environ.get('SC2PATH','/nonexistent')).is_dir() else 'unverified','reason':'Live gameplay is a separate check'}
    identities['llm']={'status':'unverified','endpoint':config.get('llm',{}).get('endpoint'),'reason':'No live request made'}
    return identities


def performance(config):
    from .environments import make_env,collect
    from .libraries import make_library
    from .rules import PSWM
    from .training import episode_contexts
    from .diagnostics import Diagnostics
    env=make_env('gridcraft',agents=2,max_steps=4,view_size=7)
    try:
        episode=collect(env,1)[0];engine=PSWM(make_library(env))
        context=episode_contexts(episode,env.schema,engine)[0][0]
        for _ in range(10):engine(context)
        count=config.get('verify',{}).get('performance_contexts',100)
        start=time.perf_counter()
        for _ in range(count):engine(context)
        latency=(time.perf_counter()-start)/count
        diagnostic=Diagnostics(engine);start=time.perf_counter()
        for _ in range(count):diagnostic.add(context,episode.observations[1],episode.observations[1])
        seconds=time.perf_counter()-start
        return {'joint_step_seconds':latency,'contexts':count,'diagnostic_seconds':seconds,'scenario':{'agents':2,'view_size':7,'max_steps':4},
          'NFR-PERF-1':{'status':'passed' if latency<=.001 else 'failed','limit_seconds':.001},
          'NFR-PERF-2':{'status':('passed' if seconds<=600 else 'failed') if count>=100000 else 'unverified',
                       'reason':None if count>=100000 else 'Short benchmark is not a 100000-context acceptance run'}}
    finally:env.close()


def regression(reference_file,records_file):
    """Compare supplied T-19/T-20 run-level records to original confidence intervals."""
    import csv
    from collections import defaultdict
    reference=json.loads(Path(reference_file).read_text())
    groups=defaultdict(list)
    with Path(records_file).open() as handle:
        for row in csv.DictReader(handle):
            if row['checkpoint']=='final' and row['value'] not in ('','None'):
                groups[(row['env'],row['config_hash'],row['metric'])].append((int(row['seed']),float(row['value'])))
    result=[]
    for criterion in reference['criteria']:
        values=groups[(criterion['env'],criterion['config_hash'],criterion['metric'])]
        if len(values) != len({s for s,v in values}): raise ValueError('Regression records duplicate a seed')
        enough=len({s for s,v in values})>=criterion.get('seeds',10)
        mean=sum(v for s,v in values)/len(values) if values else None
        bounds=criterion['ci95']
        result.append({**criterion,'mean':mean,'status':('passed' if bounds[0]<=mean<=bounds[1] else 'failed') if enough else 'unverified'})
    return result


def verify(config):
    output=Path(config.get('output','outputs/verification'));output.mkdir(parents=True,exist_ok=True)
    options=config.get('verify',{});checks=options.get('checks',['preflight','traceability'])
    checklist=json.loads((ROOT/'ns_mawm'/'requirements.json').read_text())
    results={'code':code_identity(ROOT),'scientifically_validated':False}
    if 'preflight' in checks: results['dependencies']=preflight(config)
    passed=set();failed=set();skipped=set()
    if 'tests' in checks:
        report=output/'tests.xml'
        report.unlink(missing_ok=True)
        process=subprocess.run([sys.executable,'-c','import torch, pytest, sys; sys.exit(pytest.main(sys.argv[1:]))','tests','-q','-p','no:cacheprovider',f'--junitxml={report.resolve()}',
           '--cov=ns_mawm.rules','--cov=ns_mawm.diagnostics','--cov-fail-under=90'],cwd=ROOT,capture_output=True,text=True,env={**__import__('os').environ, 'COVERAGE_FILE':str((output/'.coverage').resolve())})
        (output/'tests.log').write_text(process.stdout+process.stderr)
        results['tests']={'status':'passed' if process.returncode==0 else 'failed','log':str(output/'tests.log')}
        if report.exists():
            for case in ET.parse(report).iter('testcase'):
                module=case.attrib['classname'].replace('.','/')+'.py'
                node=module+'::'+case.attrib['name'].split('[')[0]
                target=failed if case.find('failure') is not None or case.find('error') is not None else skipped if case.find('skipped') is not None else passed
                target.update((module,node))
    if 'types' in checks:
        process=subprocess.run([sys.executable,'-m','mypy','--strict','--follow-imports=silent','ns_mawm/contracts.py','ns_mawm/api.py','--cache-dir',str((output/'mypy-cache').resolve())],cwd=ROOT,capture_output=True,text=True)
        results['types']={'status':'passed' if process.returncode==0 else 'failed','details':process.stdout+process.stderr}
    if 'performance' in checks:results['performance']=performance(config)
    if 'reproducibility' in checks:
        from .statistics import reproducibility
        reference=options.get('reference_results'); repeated=options.get('baseline_results')
        if not reference or not repeated:
            results['reproducibility']={'status':'unverified','reason':'Reference and repeated metric JSON files are required'}
        else:
            a=json.loads(Path(reference).read_text()); b=json.loads(Path(repeated).read_text())
            results['reproducibility']=reproducibility(a['metrics']['mse'],b['metrics']['mse'])
    if 'regression' in checks:
        if not options.get('reference_results') or not options.get('baseline_results'):
            results['regression']={'status':'unverified','reason':'Original confidence intervals and run records are required'}
        else: results['regression']=regression(options['reference_results'],options['baseline_results'])
    if options.get('live'):
        results['live']=live_checks(config)
    evidence=[]
    for requirement in checklist:
        paths=requirement['implementation'];nodes=requirement['checks']
        present=bool(paths) and all((ROOT/p).exists() for p in paths)
        smoke=bool(nodes) and all(n in passed and n not in failed and n not in skipped for n in nodes)
        artifact_statuses=[]
        for check in requirement.get('artifact_checks',[]):
            record=results
            for key in check.split('.'): record=record.get(key,{})
            artifact_statuses.append(record.get('status','unverified'))
        if artifact_statuses:
            smoke=all(status=='passed' for status in artifact_statuses)
        evidence.append({**requirement,'implementation_present':present,'smoke_tested':smoke,
            'status':'failed' if any(n in failed for n in nodes) or 'failed' in artifact_statuses else 'smoke-tested' if smoke else 'unverified',
            'scientifically_validated':False})
    results['requirements']=evidence
    (output/'acceptance.json').write_text(json.dumps(results,indent=2))
    return results


def live_checks(config):
    results={}
    from .environments import make_env,collect
    try:
        env=make_env('smacv2',max_steps=2)
        try: collect(env,1);results['smacv2']={'status':'passed','scope':'one short real episode'}
        finally:env.close()
    except (ImportError,FileNotFoundError) as error:results['smacv2']={'status':'unverified','reason':str(error)}
    except RuntimeError as error:
        missing=benchmark_identity('smacv2','577ab5a2cff2391f8df582da5731ea9cd6adf3c6')['status']=='unverified'
        results['smacv2']={'status':'unverified' if missing else 'failed','reason':str(error)}
    except Exception as error:results['smacv2']={'status':'failed','reason':str(error)}
    import urllib.request
    endpoint=config.get('llm',{}).get('endpoint')
    if not endpoint:results['llm']={'status':'unverified','reason':'No endpoint configured'}
    else:
        payload={'model':config['llm'].get('model','qwen3.6-35b-a3b-nvfp4'),'max_tokens':16,
                 'messages':[{'role':'user','content':'Return an empty JSON object.'}]}
        try:
            request=urllib.request.Request(endpoint.rstrip('/')+'/chat/completions',json.dumps(payload).encode(),{'Content-Type':'application/json'})
            with urllib.request.urlopen(request,timeout=10) as response: json.load(response)
            results['llm']={'status':'passed','scope':'bounded endpoint request; not scientific refinement validation'}
        except Exception as error:results['llm']={'status':'unverified','reason':str(error)}
    return results
