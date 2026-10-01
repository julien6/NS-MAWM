"""Small strict configuration validator; called before stage side effects."""
from copy import deepcopy

OPTIONS = {
 'run': 'experiment env env_commit seed split_manifest condition',
 'world_model': 'backbone strategy lambda sizes optim latent_kl_weight',
 'sizes': 'enc lstm g dec categories classes',
 'optim': 'name lr batch_seqs seq_len burn_in updates checkpoint_interval symbolic_cache_episodes',
 'collection': 'episodes seed split_seed policies policy_checkpoint',
 'rules': 'library filter corruption max_rdd minimum_support conflict_policy diagnostic_library trace_versions',
 'filter': 'scope disable_ids ids tags',
 'corruption': 'kind fraction',
 'evaluation': 'horizon stride validation_target error_split fixed_block_set compared_libraries fixed_union fixed_union_hash diagnostics_mode focal_agent',
 'llm': 'endpoint model model_version temperature seed seeds max_candidates max_calls rounds token_budget max_tokens timeout serving_gpus allowed_sources source_tree sections feedback min_support max_rdd max_conflict coverage_floor_fraction independent_author',
 'stats': 'resamples comparisons extension ci test correction',
 'extension': 'enabled ci_width_tolerance initial_seeds extended_seeds',
 'control': 'learner model_free real_steps checkpoint_interval rollout_cap real_ratio gamma alpha tau burn_in hidden lr frames_per_batch fit_interval num_envs eval_episodes batch_size policy_updates world_model_updates imagined_capacity real_capacity warmup model_episode_capacity imagined_branches return_threshold mamba',
 'tuning': 'mode configurations learning_rates widths sequence_lengths lambdas strategies',
 'campaign': 'jobs resume conditions stages',
 'verify': 'checks live reference_results performance_contexts baseline_results',
 'export': 'source destination deny replacements benchmark_commits',
}
TOP = set(OPTIONS) - {'sizes','optim','filter','corruption','extension'} | {'data','output','threads','device','seeds','seconds_per_update','protocol_hash'}


def _keys(value, allowed, path):
    if not isinstance(value, dict):
        raise ValueError(f'{path} must be a mapping')
    extra = set(value) - set(allowed)
    if extra:
        raise ValueError(f'Unknown {path} options: {sorted(extra)}')


def validate_config(config, stage=None):
    from .environments import ENVIRONMENTS
    from .models import BACKBONES, STRATEGIES
    from .extensions import STRATEGY_PLUGINS, ENVIRONMENT_OPTIONS
    from vGridcraft.vgridcraft.config import VGridcraftConfig
    from dataclasses import fields
    _keys(config, TOP | {'environment'}, 'configuration')
    for key, value in config.items():
        if key in OPTIONS:
            _keys(value, OPTIONS[key].split(), key)
    for parent, child in [('world_model','sizes'),('world_model','optim'),('rules','filter'),('rules','corruption'),('stats','extension')]:
        if child in config.get(parent, {}):
            _keys(config[parent][child], OPTIONS[child].split(), parent+'.'+child)
    name = config.get('run', {}).get('env')
    if stage not in ('report','export-anon','verify') and name not in ENVIRONMENTS:
        raise ValueError(f'Unknown environment: {name}')
    env_options = {
      'gridcraft': {f.name for f in fields(VGridcraftConfig)} | {'agents'},
      'overcooked': {'agents','max_steps','layout'},
      'predator_prey': {'agents','max_steps','landmarks'},
      'smacv2': {'agents','max_steps','map_name','benchmark_commit','capability_config','step_mul','fully_observable'},
      **ENVIRONMENT_OPTIONS}
    if name in env_options:
        _keys(config.get('environment', {}), env_options[name], 'environment')
    wm = config.get('world_model', {})
    if wm.get('backbone','jopm_lstm') not in set(BACKBONES) | {'pswm_only','llm_code'}:
        raise ValueError('Unknown backbone')
    if wm.get('strategy','none') not in STRATEGIES | STRATEGY_PLUGINS.keys():
        raise ValueError('Unknown strategy')
    if wm.get('optim', {}).get('name','adam') != 'adam':
        raise ValueError('Only the Adam optimizer is implemented')
    enums = [(config.get('evaluation',{}),'fixed_block_set',{'union_of_versions'}),
             (config.get('stats',{}),'ci',{'bootstrap_percentile'}),
             (config.get('stats',{}),'test',{'wilcoxon'}),
             (config.get('stats',{}),'correction',{'holm'}),
             (config.get('rules',{}),'conflict_policy',{'block_rejection','reject_library'})]
    for section,key,values in enums:
        if key in section and section[key] not in values:
            raise ValueError(f'Unsupported {key}: {section[key]}')
    if config.get('evaluation',{}).get('error_split',True) is not True:
        raise ValueError('The SRS requires covered/uncovered error reporting')
    if wm.get('lambda',1) < 0:
        raise ValueError('lambda must be nonnegative')
    if config.get('control', {}).get('learner','benchmarl') not in {'benchmarl','reference','mamba'}:
        raise ValueError('Unknown control learner')
    control = config.get('control', {})
    if stage == 'control' and control.get('learner','benchmarl') == 'benchmarl':
        frames = control.get('frames_per_batch',control.get('fit_interval',1000))
        if frames < 1 or frames % control.get('num_envs',1) or control.get('real_steps',1000000) % frames or control.get('checkpoint_interval',25000) % frames:
            raise ValueError('Control batches must divide the real budget and checkpoint interval exactly')
        if not control.get('model_free') and wm.get('backbone') in {'pswm_only','llm_code'}:
            raise ValueError('Model-based control requires reward/termination heads')
    extension = config.get('stats',{}).get('extension',{})
    if extension.get('enabled') and extension.get('ci_width_tolerance',0) <= 0:
        raise ValueError('Seed extension requires a positive predeclared CI-width tolerance')
    if config.get('evaluation',{}).get('diagnostics_mode','real') not in {'real','both'}:
        raise ValueError('diagnostics_mode must be real or both')
    if stage in {'train','evaluate','control','refine','generate-rules','tune','freeze','generate-code'}:
        if not config.get('data') or not wm:
            raise ValueError(f'{stage} requires data and world_model')
    return deepcopy(config)
