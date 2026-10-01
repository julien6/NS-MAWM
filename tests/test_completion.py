from copy import deepcopy
import json
from pathlib import Path
import pytest
import torch
from test_pipeline import tiny
from ns_mawm.configuration import validate_config
from ns_mawm.environments import make_env,collect
from ns_mawm.libraries import make_library
from ns_mawm.data import Dataset
from ns_mawm.training import train,evaluate


def bundle():
    env=make_env('gridcraft',agents=2,view_size=3,max_steps=6)
    data=Dataset.split(collect(env,8))
    return env,data,env.schema,make_library(env)


def test_config_rejects_disconnected_options():
    config=tiny(); config['data']='unused'
    validate_config(config,'train')
    for section,key in [('world_model','fake'),('evaluation','unknown'),('control','fake')]:
        c=deepcopy(config); c.setdefault(section,{})[key]=1
        with pytest.raises(ValueError,match='Unknown'): validate_config(c,'train')


def test_tuning_and_frozen_guard(tmp_path):
    from ns_mawm.tuning import tune
    env,data,schema,lib=bundle()
    try:
        cfg=tiny();cfg['tuning']={'learning_rates':[1e-4,3e-4],'widths':[8],'sequence_lengths':[2]}
        result=tune(data,schema,lib,env.action_size,cfg,tmp_path)
        assert result['configurations']==2
        data.freeze(lib.hash,cfg)
        with pytest.raises(PermissionError): tune(data,schema,lib,env.action_size,cfg,tmp_path)
        with pytest.raises(PermissionError): train(data,schema,lib,env.action_size,cfg)
    finally: env.close()


@pytest.mark.parametrize("strategy", ["none","projection","residual","regularization"])
def test_published_mamba_optimizer(strategy):
    from ns_mawm.mamba import MambaWM
    env,data,schema,lib=bundle()
    try:
        cfg=tiny();cfg['world_model'].update(backbone='mamba_wm',strategy=strategy)
        cfg['world_model']['sizes'].update(categories=2,classes=2)
        net,timing=train(data,schema,lib,env.action_size,cfg)
        assert isinstance(net,MambaWM) and timing['updates']==1
        assert evaluate(data,schema,lib,net,cfg)['metrics']['mse']>=0
    finally:env.close()


def test_union_provenance_and_imagined_diagnostics(tmp_path):
    from ns_mawm.tuning import freeze_union
    env,data,schema,lib=bundle()
    try:
        cfg=tiny(); path=tmp_path/'union.json'
        freeze_union(data,schema,[lib],path)
        cfg['evaluation'].update(fixed_union=str(path),diagnostics_mode='both')
        net,_=train(data,schema,lib,env.action_size,cfg)
        result=evaluate(data,schema,lib,net,cfg)
        assert result['imagined_diagnostics']
        record=json.loads(path.read_text());record['blocks']=[];path.write_text(json.dumps(record))
        with pytest.raises(ValueError,match='provenance'):evaluate(data,schema,lib,net,cfg)
    finally:env.close()


def test_campaign_resume_failure_and_artifact_change(tmp_path):
    from ns_mawm.campaign import run_campaign
    called=[]
    cfg={'output':str(tmp_path),'campaign':{'jobs':[
        {'id':'one','stage':'mock','depends':[],'config':{}},
        {'id':'two','stage':'mock','depends':['one'],'config':{}}]}}
    def execute(job,config,results,directory):
        called.append(job['id']);p=directory/(job['id']+'.txt');p.write_text(job['id'])
        return {'configuration':{},'artifacts':[str(p)]}
    run_campaign(cfg,execute);run_campaign(cfg,execute)
    assert called==['one','two']
    (tmp_path/'one.txt').write_text('corrupted')
    run_campaign(cfg,execute)
    assert called==['one','two','one']
    cfg['campaign']['jobs'][0]['config']={'change':1}
    def fail(*args):raise RuntimeError('failure')
    result=run_campaign(cfg,fail)
    assert result['one']['status']=='failed' and result['two']['status']=='blocked'


def test_external_mamba_short_control(tmp_path):
    from ns_mawm.mamba_control import run_control
    from ns_mawm.artifacts import Run
    env,data,schema,lib=bundle()
    try:
        cfg=tiny();cfg['environment'].update(max_steps=6,view_size=3)
        cfg['control']={'learner':'mamba','real_steps':12,'checkpoint_interval':6,'eval_episodes':1,
          'mamba':{'HIDDEN':8,'MODEL_HIDDEN':8,'EMBED':8,'DETERMINISTIC':8,'N_CATEGORICALS':2,'N_CLASSES':2,
          'ACTION_HIDDEN':8,'REWARD_HIDDEN':8,'PCONT_HIDDEN':8,'VALUE_HIDDEN':8,
          'CAPACITY':32,'MIN_BUFFER_SIZE':5,'SEQ_LENGTH':4,'MODEL_EPOCHS':1,'EPOCHS':1,'PPO_EPOCHS':1,'MODEL_BATCH_SIZE':1,'BATCH_SIZE':1,'HORIZON':3}}
        run=Run(tmp_path,cfg,lib,data,0,{'commit':'test'})
        result=run_control(env,data,schema,lib,cfg,run)
        assert result['real_steps']==12 and result['policy_updates']>0
    finally:env.close()


def test_seed_extension_and_reproduction():
    from ns_mawm.statistics import extension_decision,reproducibility
    arms={'a':dict(enumerate(range(10))),'b':dict(enumerate([0]*10))}
    assert extension_decision(arms,{'enabled':True,'ci_width_tolerance':.01})['additional_seeds']==list(range(10,20))
    assert reproducibility(1,1.001)['passed']
    assert not reproducibility(1,1.1)['passed']


def test_repair_contains_exact_rejected_json(tmp_path):
    from test_refinement_reporting import setup,patch
    from ns_mawm.refinement import refine
    from ns_mawm.rules import Library
    schema,data,workflow=setup(tmp_path)
    initial=Library([],schema);requests=[]
    def transport(payload):
        request=json.loads(payload['messages'][1]['content']);requests.append(request)
        if len(requests)==1:content='malformed candidate exactly as returned'
        else:content=json.dumps({'patches':[patch(initial)]})
        return {'choices':[{'message':{'content':content}}],'usage':{'prompt_tokens':1,'completion_tokens':1}}
    workflow.transport=transport;workflow.config.update(max_candidates=1,max_calls=3,min_support=1)
    assert refine(workflow,initial,tiny(),1,[]) is not None
    assert requests[1]['stage']=='P2'
    assert requests[1]['library']['rejected_candidate']=='malformed candidate exactly as returned'


def test_code_baseline_is_complete_or_fails(tmp_path):
    from test_refinement_reporting import setup,patch
    from ns_mawm.refinement import generate_code
    from ns_mawm.rules import Library
    schema,data,workflow=setup(tmp_path)
    base=Library([],schema)
    workflow.transport=lambda payload:{'choices':[{'message':{'content':json.dumps({'patches':[patch(base)]})}}],'usage':{'prompt_tokens':1,'completion_tokens':1}}
    workflow.config.update(max_candidates=1,max_calls=1)
    selected=generate_code(workflow,tiny(),1,[])
    assert selected.rules
    workflow.path=tmp_path/'failed';workflow.path.mkdir()
    workflow.calls=0
    workflow.transport=lambda payload:{'choices':[{'message':{'content':'{"patches": []}'}}],'usage':{'prompt_tokens':1,'completion_tokens':1}}
    with pytest.raises(RuntimeError,match='No complete'):generate_code(workflow,tiny(),1,[])


def test_backbone_only_contract_registration():
    from torch import nn
    from ns_mawm.extensions import register_backbone
    from ns_mawm.models import BACKBONES,WorldModel
    from ns_mawm.rules import PSWM,Context
    class External(nn.Module):
        def __init__(self,schema,action_size,**kwargs):
            super().__init__();self.weight=nn.Parameter(torch.zeros(schema.shape))
        def init_state(self,prefix=()):return None
        def step(self,state,obs,joint_action,extra=None):return obs+self.weight,state
    register_backbone('test_external',External)
    env,data,schema,lib=bundle()
    try:
        cfg=tiny();cfg['world_model']['backbone']='test_external'
        net,_=train(data,schema,lib,env.action_size,cfg)
        assert evaluate(data,schema,lib,net,cfg)['metrics']['mse']>=0
    finally:env.close();BACKBONES.pop('test_external')


def test_mamba_matches_pinned_upstream_computations():
    from ns_mawm.vendor.mamba.networks.dreamer.rnns import DiscreteLatentDist
    from ns_mawm.vendor.mamba.agent.optim.utils import compute_return
    record=json.loads((Path(__file__).parent/'mamba_reference.json').read_text())
    torch.manual_seed(record['seed'])
    model=DiscreteLatentDist(4,2,3,8)
    logits,latents=model(torch.arange(8,dtype=torch.float32).reshape(2,4)/10)
    assert torch.allclose(logits,torch.tensor(record['logits']),atol=1e-6)
    assert torch.equal(latents,torch.tensor(record['latents']))
    reward=torch.arange(6,dtype=torch.float32).reshape(3,2,1)/10
    result=compute_return(reward,reward+.2,torch.ones_like(reward)*.9,torch.ones(2,1)*.4,.95,.99)
    assert torch.allclose(result,torch.tensor(record['returns']),atol=1e-6)


def test_benchmarl_collection_checkpoint(tmp_path):
    pytest.importorskip('torchrl')
    from ns_mawm.benchmarl import run_control
    from ns_mawm.artifacts import Run
    from ns_mawm.policies import CheckpointPolicy
    env,data,schema,lib=bundle()
    try:
        cfg=tiny();cfg['environment'].update(max_steps=6,view_size=3)
        cfg['control']={'model_free':True,'real_steps':6,'frames_per_batch':3,'checkpoint_interval':3,'num_envs':1,
                       'eval_episodes':1,'hidden':8,'batch_size':2,'policy_updates':1,'real_capacity':32}
        run=Run(tmp_path,cfg,lib,data,0,{'commit':'test'})
        run_control(env,data,schema,lib,cfg,run)
        policy=CheckpointPolicy(env,run.path/'collection-policy.pt')
        try:
            policy.reset(1);obs=env.reset(1)
            for _ in range(2):
                actions=policy(obs);assert len(actions)==2
                obs,*_=env.step(actions)
        finally:policy.close()
    finally:env.close()


def test_two_condition_campaign_and_resume(tmp_path,capsys):
    import yaml
    from ns_mawm.campaign import run_campaign
    config=yaml.safe_load(Path('configs/smoke.yaml').read_text())
    config.update(output=str(tmp_path),seeds=1,campaign={'conditions':['B1','B4']},
                  tuning={'learning_rates':[.0003],'widths':[8],'sequence_lengths':[2]})
    config['evaluation']['fixed_block_set']='union_of_versions'
    config['world_model']['optim']['updates']=1
    first=run_campaign(config)
    assert all(r['status']=='completed' for r in first.values()),first
    second=run_campaign(config)
    assert all(r['status']=='completed' for r in second.values()),second
    assert first==second
    capsys.readouterr()


def test_http_llm_transport(tmp_path):
    import threading
    from http.server import BaseHTTPRequestHandler,HTTPServer
    from test_refinement_reporting import setup
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            request=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            assert request['messages'][0]['role']=='system'
            self.send_response(200);self.end_headers()
            self.wfile.write(json.dumps({'choices':[{'message':{'content':'{"patches": []}'}}],
                'usage':{'prompt_tokens':2,'completion_tokens':3}}).encode())
        def log_message(self,*args):pass
    server=HTTPServer(('127.0.0.1',0),Handler)
    worker=threading.Thread(target=server.serve_forever,daemon=True);worker.start()
    try:
        _,_,workflow=setup(tmp_path)
        workflow.config['endpoint']=f'http://127.0.0.1:{server.server_port}/v1'
        assert workflow.call('P1',[])=={'patches':[]}
        assert workflow.tokens==5
    finally:server.shutdown();server.server_close();worker.join()


def test_short_overcooked_control(tmp_path):
    pytest.importorskip('overcooked_ai_py');pytest.importorskip('torchrl')
    from ns_mawm.benchmarl import run_control
    from ns_mawm.artifacts import Run
    env=make_env('overcooked',agents=2,max_steps=3)
    try:
        data=Dataset.split(collect(env,4));schema=env.schema;lib=make_library(env)
        cfg=tiny('overcooked');cfg['control']={'real_steps':6,'frames_per_batch':3,'checkpoint_interval':3,
           'eval_episodes':1,'hidden':8,'batch_size':2,'world_model_updates':1,'rollout_cap':2,'real_capacity':32,'imagined_capacity':16}
        run=Run(tmp_path,cfg,lib,data,0,{'commit':'test'})
        assert run_control(env,data,schema,lib,cfg,run)['real_steps']==6
    finally:env.close()


def test_failed_refinement_produces_neural_checkpoint(tmp_path):
    from test_refinement_reporting import setup
    from ns_mawm.refinement import refine
    from ns_mawm.rules import Library
    schema,data,workflow=setup(tmp_path)
    workflow.config.update(max_calls=1,max_candidates=1)
    workflow.transport=lambda payload:{'choices':[{'message':{'content':'invalid'}}],'usage':{'prompt_tokens':1,'completion_tokens':1}}
    assert refine(workflow,Library([],schema),tiny(),1,[]) is None
    selection=json.loads((tmp_path/'selection.json').read_text())
    checkpoint=torch.load(selection['fallback_checkpoint'],weights_only=True)
    assert checkpoint['config']['world_model']['strategy']=='none'


def test_verification_map_covers_all_srs_ids():
    import re
    specification=Path('NS-MAWM — Software Requirements Specification.md').read_text()
    required=set(re.findall(r'\b(?:FR-[A-Z]+-\d+|NFR-[A-Z]+-\d+)\b',specification))
    records=json.loads(Path('ns_mawm/requirements.json').read_text())
    indexed={row['id']:row for row in records}
    assert required <= indexed.keys()
    assert {f'T-{i:02}' for i in range(1,21)} <= indexed.keys()
    for row in records:
        assert all(Path(p).is_file() for p in row['implementation'])
        for check in row['checks']:
            path,_,name=check.partition('::')
            assert Path(path).is_file()
            if name: assert 'def '+name+'(' in Path(path).read_text()


def test_smac_imagined_masks_use_prediction():
    from test_smac_contract import features
    from ns_mawm.smac import codec,encode,available_from_observation
    names=features();schema,mappings=codec(names,2)
    native=torch.zeros(2,len(names));obs=encode(schema,mappings,native)
    assert available_from_observation(schema,obs,7)==[[True,False,False,False,False,False,False]]*2
    native[:,names.index('own_health')]=1
    native[:,names.index('enemy_shootable_0')]=1
    masks=available_from_observation(schema,encode(schema,mappings,native),7)
    assert all(m[1] and m[6] and not m[0] for m in masks)


def test_registered_public_adapters():
    from ns_mawm.api import RegisteredEnvironment,RegisteredBackbone
    from torch import nn
    class Toy:
        agents=1;action_size=2;metadata={'environment':'toy'}
        def reset(self,seed):return torch.ones(1,1)
        def step(self,actions):return torch.ones(1,1),0.,False,True
        def close(self):pass
    wrapped=RegisteredEnvironment(Toy())
    assert wrapped.reset(0).shape==(1,1)
    with pytest.raises(ValueError):wrapped.step([3])
    class Predictor:
        def init_state(self,prefix=()):return None
        def step(self,state,obs,joint_action,extra=None):return obs,state
    model=RegisteredBackbone(Predictor())
    assert model.step(model.init_state(),torch.ones(1,1),torch.zeros(1))[0].shape==(1,1)


def test_exported_source_provenance_without_git(tmp_path,monkeypatch):
    import subprocess
    from ns_mawm import environments
    source=tmp_path/'source';source.mkdir()
    (source/'benchmark_commits.json').write_text(json.dumps({'ns_mawm':'a'*40,'overcooked':'3935fd4d4362502b45e38e1a5a54bfdad06e32ed'}))
    monkeypatch.setattr(environments,'ROOT',source)
    monkeypatch.setattr(environments.subprocess,'run',lambda *a,**k:subprocess.CompletedProcess(a,1,stdout='',stderr=''))
    assert environments.commit(source)=='a'*40
    assert environments.commit(source/'Overcooked_AI')=='3935fd4d4362502b45e38e1a5a54bfdad06e32ed'


def test_regression_reference_requires_real_seed_evidence(tmp_path):
    from ns_mawm.verification import regression
    reference=tmp_path/'reference.json';records=tmp_path/'records.csv'
    reference.write_text(json.dumps({'criteria':[{'env':'gridcraft','config_hash':'x','metric':'mse','ci95':[.09,.11],'seeds':10}]}))
    records.write_text('env,config_hash,metric,checkpoint,seed,value\n'+''.join(f'gridcraft,x,mse,final,{i},0.1\n' for i in range(10)))
    assert regression(reference,records)[0]['status']=='passed'
    records.write_text('env,config_hash,metric,checkpoint,seed,value\ngridcraft,x,mse,final,0,0.1\n')
    assert regression(reference,records)[0]['status']=='unverified'


def test_environment_plugin_library_roundtrip(tmp_path):
    from ns_mawm.extensions import register_environment,ENVIRONMENT_LIBRARIES,ENVIRONMENT_OPTIONS
    from ns_mawm.environments import ENVIRONMENTS
    from ns_mawm.schema import Schema,BlockSpec
    from ns_mawm.rules import Library,compile_rule
    from ns_mawm.libraries import save_library,load_library
    schema=Schema([BlockSpec('x',0,'scalar',None,(0,1))],(1,1))
    class Toy:
        agents=1;action_size=1
        def __init__(self):self.schema=schema;self.metadata={'environment':'toy_plugin'}
        def reset(self,seed):return torch.zeros(1,1)
        def step(self,actions):return torch.zeros(1,1),0.,False,True
        def close(self):pass
    def factory(env,schema):
        return Library([compile_rule({'id':'toy','version':0,'stage':0,'scope':'individual','reads':['x'],'writes':['x'],
          'source':'handcrafted','code':"def guard(c):\n return Tri.TRUE\ndef effect(c):\n return [Assignment('x', 0., 'toy', 0)]"})],schema)
    register_environment('toy_plugin',Toy,factory)
    try:
        env=make_env('toy_plugin');library=make_library(env)
        path=tmp_path/'library.json';save_library(library,path)
        assert load_library(path,schema).hash==library.hash
        validate_config({'run':{'env':'toy_plugin'},'environment':{}},'collect')
    finally:
        ENVIRONMENTS.pop('toy_plugin');ENVIRONMENT_LIBRARIES.pop('toy_plugin');ENVIRONMENT_OPTIONS.pop('toy_plugin')


def test_campaign_invalidates_changed_dependency_contents(tmp_path):
    from ns_mawm.campaign import run_campaign
    calls=[]
    config={'output':str(tmp_path),'campaign':{'jobs':[
        {'id':'upstream','stage':'mock','depends':[],'config':{}},
        {'id':'downstream','stage':'mock','depends':['upstream'],'config':{}}]}}
    def executor(job,config,results,directory):
        calls.append(job['id']);path=directory/(job['id']+'.txt');path.write_text(str(len(calls)))
        return {'configuration':{},'artifacts':[str(path)]}
    run_campaign(config,executor)
    (tmp_path/'upstream.txt').write_text('invalid')
    run_campaign(config,executor)
    assert calls==['upstream','downstream','upstream','downstream']
