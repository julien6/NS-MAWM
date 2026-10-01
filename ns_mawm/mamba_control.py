"""Serial environment bridge to the published MAMBA learner and PPO losses."""
from copy import deepcopy
from pathlib import Path
import time
import numpy as np
import torch
from torch.nn import functional as F
from .mamba import UPSTREAM_COMMIT
from .vendor.mamba.configs.dreamer.optimal.starcraft.LearnerConfig import DreamerLearnerConfig
from .vendor.mamba.agent.learners.DreamerLearner import DreamerLearner
from .vendor.mamba.environments import Env
from .training import seed_all, isolated_rng
from .statistics import control_metrics


@torch.no_grad()
def action(learner, observation, state, previous, available=None):
    model=learner.model
    device=next(model.parameters()).device
    # Each deployed agent sees only its own local history. Cross-agent attention
    # is used by the centralized imagined dynamics during learner updates.
    local=[]
    for index in range(observation.shape[1]):
        current=model(observation[:,index][None,None].to(device),
                      None if previous is None else previous[:,index:index+1],
                      None if state is None else state[index])
        local.append(current)
    features=torch.cat([s.get_features() for s in local],1)
    _,logits=learner.actor(features)
    if available is not None:
        logits=logits.masked_fill(~torch.as_tensor(available,device=device,dtype=torch.bool)[None],-1e9)
    sampled=torch.distributions.Categorical(logits=logits).sample()
    return sampled[0].cpu().tolist(),local,F.one_hot(sampled,logits.shape[-1]).float()


@isolated_rng
def run_control(env,dataset,schema,library,config,run):
    from .environments import make_env
    seed=config['run'].get('seed',0); seed_all(seed)
    options=config.get('control',{})
    cfg=DreamerLearnerConfig()
    cfg.ENV_TYPE=Env.STARCRAFT
    cfg.IN_DIM=schema.shape[0]; cfg.ACTION_SIZE=env.action_size
    cfg.DEVICE=config.get('device','cpu'); cfg.LOG_FOLDER=str(run.path/'mamba')
    # Explicit overrides are for smoke runs; every deviation is recorded.
    overrides=options.get('mamba',{})
    for key,value in overrides.items():
        if not hasattr(cfg,key): raise ValueError(f'Unknown MAMBA setting: {key}')
        setattr(cfg,key,value)
    cfg.STOCHASTIC=cfg.N_CATEGORICALS*cfg.N_CLASSES
    cfg.FEAT=cfg.STOCHASTIC+cfg.DETERMINISTIC; cfg.GLOBAL_FEAT=cfg.FEAT+cfg.EMBED
    if cfg.SEQ_LENGTH < 4 or cfg.MIN_BUFFER_SIZE <= cfg.SEQ_LENGTH:
        raise ValueError('MAMBA needs sequence_length >= 4 and minimum replay > sequence_length')
    learner=DreamerLearner(cfg)
    run.manifest['external_baseline']={'name':'MAMBA','commit':UPSTREAM_COMMIT,'preset':'optimal/starcraft',
       'settings':{k:v for k,v in vars(cfg).items() if isinstance(v,(int,float,str,bool))},
       'adaptations':['Schema observations/actions','Serial collection','Local-history deployment','Team rewards','Time limits marked separately'], 'overrides':overrides}
    maximum=options.get('real_steps',1000000); interval=options.get('checkpoint_interval',25000)
    steps,returns=[],[]; started=time.perf_counter(); state=previous=None
    obs=env.reset(seed+20000); episode=[]; episode_index=0
    for step in range(1,maximum+1):
        available=env.available_actions() if hasattr(env,'available_actions') else np.ones((env.agents,env.action_size))
        acts,state,previous=action(learner,obs,state,previous,available)
        nxt,reward,term,trunc=env.step(acts)
        episode.append({'observation':obs.T.numpy(),'action':previous[0].cpu().numpy(),
            'reward':np.full((env.agents,1),reward),'done':np.full((env.agents,1),term),
            'fake':np.zeros((env.agents,1)),'last':np.full((env.agents,1),term or trunc),
            'avail_action':np.asarray(available)})
        obs=nxt
        if term or trunc or step==maximum:
            # Terminal observation is an alignment row, never an extra interaction.
            episode.append({'observation':obs.T.numpy(),'action':np.zeros((env.agents,env.action_size)),
                'reward':np.zeros((env.agents,1)),'done':np.full((env.agents,1),term),
                'fake':np.ones((env.agents,1)),'last':np.ones((env.agents,1)),
                'avail_action':np.ones((env.agents,env.action_size))})
            learner.step({k:np.asarray([r[k] for r in episode],dtype=np.float32) for k in episode[0]})
            episode=[]; state=previous=None; episode_index+=1
            if step<maximum: obs=env.reset(seed+20000+episode_index)
        if step%interval==0 or step==maximum:
            evaluate_env=make_env(config['run']['env'],**config.get('environment',{}))
            values=[]
            try:
                for index in range(options.get('eval_episodes',5)):
                    eo=evaluate_env.reset(seed+100000+index); es=ep=None; total=0.
                    while True:
                        masks=evaluate_env.available_actions() if hasattr(evaluate_env,'available_actions') else None
                        ea,es,ep=action(learner,eo,es,ep,masks)
                        eo,r,d,t=evaluate_env.step(ea); total+=r
                        if d or t: break
                    values.append(total)
            finally: evaluate_env.close()
            value=float(np.mean(values)); steps.append(step); returns.append(value)
            run.record('return',value,step,unit='reward'); run.record('wall_seconds',time.perf_counter()-started,step,unit='seconds')
            torch.save({'parameters':learner.params(),'configuration':vars(cfg),'real_steps':step,'upstream_commit':UPSTREAM_COMMIT},run.path/f'mamba-{step}.pt')
    result={'learner':'MAMBA','real_steps':maximum,'checkpoints':steps,'returns':returns,'policy_updates':learner.cur_update-1}
    result.update(control_metrics(steps,returns,options.get('return_threshold',float('inf'))))
    for key in ('final_return','normalized_auc','interactions_to_threshold'): run.record(key,result[key])
    return result
