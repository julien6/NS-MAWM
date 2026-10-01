import json
from dataclasses import replace
import pytest
import torch
from ns_mawm.schema import Schema,BlockSpec
from ns_mawm.data import Dataset,Episode
from ns_mawm.rules import Library,Context,PSWM,digest
from ns_mawm.refinement import RuleWorkflow,admissibility
from ns_mawm.provenance import RuleArchive
from ns_mawm.libraries import load_library,save_library
from ns_mawm.experiments import report,plan
from ns_mawm.artifacts import Run


def setup(tmp_path,transport=None):
    schema=Schema([BlockSpec('x',0,'scalar',None,(0,1))],(1,1))
    episodes=[Episode(f'ep{i}',torch.zeros(3,1,1),torch.zeros(2,1,dtype=torch.long),torch.zeros(2),
        torch.tensor([False,True]),torch.zeros(2,dtype=torch.bool),i) for i in range(10)]
    dataset=Dataset.split(episodes)
    return schema,dataset,RuleWorkflow(schema,dataset,tmp_path,{'max_candidates':4},transport)


def patch(library):
    tests=[{'context':{'observation':obs,'joint_action':[0]},'guard':guard,'assignments':assigned}
           for obs,guard,assigned in [({'x':0.},'TRUE',{'x':0.}),({'x':1.},'FALSE',{}),({},'UNKNOWN',{})]]
    return {'base_library_version':library.hash,'candidate_version':1,'rule_id':'constant','rule_version':0,
        'operation':'add','read_blocks':['x'],'write_blocks':['x'],'dependencies':[],
        'guard_code':"def guard(c):\n if 'x' not in c.observation:\n  return Tri.UNKNOWN\n return Tri.TRUE if c.observation['x'] == 0 else Tri.FALSE",
        'effect_code':"def effect(c):\n return [Assignment('x', 0., 'constant', 0)]",'assumptions':[],
        'evidence_ids':[],'change_summary':'exact zero persistence','test_cases':tests}


def test_candidate_versions_validation_archive(tmp_path):
    schema,data,workflow=setup(tmp_path)
    base=Library([],schema)
    response={'patches':[patch(base)]}
    lib=workflow.candidate(base,response)
    assert lib.validate(require_cases=True)['valid']
    save_library(lib,tmp_path/'lib.json')
    assert load_library(tmp_path/'lib.json',schema).hash==lib.hash
    stale=patch(lib);stale.update(operation='revise',rule_version=0)
    with pytest.raises(ValueError,match='new version'):
        workflow.candidate(lib,{'patches':[stale]})
    archive=RuleArchive(tmp_path/'archive')
    archive.add(lib.rules[0])
    with pytest.raises(ValueError,match='new rule version'):
        archive.add(replace(lib.rules[0],reads=frozenset()))
    updated=archive.human_edit(lib.rules[0],lib.rules[0].code.replace("'constant', 0)","'constant', 1)"))
    assert updated.source=='human_edited' and updated.version==1


def test_invalid_json_consumes_budget_and_archives(tmp_path):
    def transport(payload):return {'choices':[{'message':{'content':'not json'}}],'usage':{'prompt_tokens':3,'completion_tokens':2}}
    schema,data,workflow=setup(tmp_path,transport)
    with pytest.raises(ValueError):workflow.call('P1',[])
    assert workflow.calls==1 and workflow.tokens==5
    assert json.loads((tmp_path/'calls.jsonl').read_text())['error']


def test_admissibility_support():
    report={'rules':[{'rule_id':'x','support':0,'rdd':None}], 'conflict_fraction':0.,'accepted_coverage':1.}
    assert not admissibility(report,.5,{})['admissible']
    report['rules'][0].update(support=100,rdd=.01)
    assert admissibility(report,.5,{})['admissible']


def test_report_seed_grouping(tmp_path):
    schema,data,workflow=setup(tmp_path/'llm')
    library=Library([],schema)
    for seed in range(2):
        config={'run':{'env':'gridcraft','seed':seed},'world_model':{'strategy':'none'}}
        run=Run(tmp_path,config,library,data,seed,{'commit':'a'*40})
        run.record('mse',.1+seed*.01)
        run.finish()
    result=report(tmp_path/'raw_runs.csv',tmp_path/'report',resamples=100)
    assert result['summary'][0]['n']==2
    assert (tmp_path/'report'/'rule_diagnostics.svg').exists()


def test_grid_equal_budgets():
    cfg={'run':{'env':'gridcraft'},'world_model':{'backbone':'jopm_lstm','strategy':'regularization','optim':{'updates':1}},'seeds':2}
    grid=plan(cfg,seconds_per_update=1.)
    assert grid['runs']
    assert {c['run']['seed'] for c in grid['runs']}=={0,1}
    assert all(c['tuning']['configurations']==12 for c in grid['runs'])


def test_anonymous_binary_and_notebook(tmp_path):
    import zipfile
    from ns_mawm.artifacts import export_anonymous
    source=tmp_path/'source';source.mkdir()
    torch.save({'model':{'weight':torch.ones(2)},'path':'/home/private/models'},source/'weights.pt')
    (source/'example.ipynb').write_text(json.dumps({'metadata':{'author':'private'},'cells':[]}))
    destination=tmp_path/'anonymous.zip'
    export_anonymous(source,destination,deny=['private'],replacements={},benchmark_commits={'env':'a'*40})
    with zipfile.ZipFile(destination) as archive:
        assert 'weights.pt' in archive.namelist()
        assert 'private' not in archive.read('example.ipynb').decode()


def test_complete_mock_refinement(tmp_path):
    from ns_mawm.refinement import refine
    from test_pipeline import tiny
    schema,data,workflow=setup(tmp_path)
    initial=Library([],schema)
    def transport(payload):
        content=json.loads(payload['messages'][1]['content'])
        p=patch(initial)
        p['base_library_version']=content['library']['library_hash']
        return {'choices':[{'message':{'content':json.dumps({'patches':[p]})}}],
                'usage':{'prompt_tokens':20,'completion_tokens':20}}
    workflow.transport=transport
    workflow.config.update(min_support=1,max_candidates=1)
    selected=refine(workflow,initial,tiny(),1,[],rounds=1)
    assert selected is not None
    assert any(selected.enabled(r) for r in selected.rules)
    saved=json.loads((tmp_path/'selection.json').read_text())
    assert saved['library_hash']==selected.hash
