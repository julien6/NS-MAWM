import importlib.util
from pathlib import Path
import pytest
import torch


def test_benchmarl_short_control(tmp_path):
    pytest.importorskip('torchrl')
    pytest.importorskip('hydra')
    from ns_mawm.benchmarl import run_control
    from ns_mawm.environments import make_env, collect, ROOT
    from ns_mawm.data import Dataset
    from ns_mawm.libraries import make_library
    from ns_mawm.artifacts import Run, code_identity
    from test_pipeline import tiny
    torch.set_num_threads(1)
    cfg=tiny()
    cfg['environment'].update(view_size=3,max_steps=3)
    cfg['control']={'real_steps':6,'frames_per_batch':3,'checkpoint_interval':3,'num_envs':1,
        'eval_episodes':1,'hidden':8,'batch_size':2,'policy_updates':1,'world_model_updates':1,
        'rollout_cap':2,'imagined_capacity':16,'real_capacity':32}
    env=make_env('gridcraft',**cfg['environment'])
    try:
        dataset=Dataset.split(collect(env,4))
        schema=env.schema
        library=make_library(env)
        run=Run(tmp_path,cfg,library,dataset,0,code_identity(ROOT))
        result=run_control(env,dataset,schema,library,cfg,run)
        assert result['real_steps']==6
        assert result['learner']=='BenchMARL_MAMBPO'
        assert len(result['checkpoints'])>=2
    finally:
        env.close()
