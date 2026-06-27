from html import escape


GENERAL = "General"
WORLD_MODEL_TRAINING = "World Model Training"
WORLD_MODEL_EVALUATION = "World Model Evaluation"
MARL_TRAINING = "MARL Training"
MARL_EVALUATION = "MARL Evaluation"

SECTIONS = (
  GENERAL,
  WORLD_MODEL_TRAINING,
  WORLD_MODEL_EVALUATION,
  MARL_TRAINING,
  MARL_EVALUATION,
)

NAMESPACE_TO_SECTION = {
  "wm_training": WORLD_MODEL_TRAINING,
  "world_model_training": WORLD_MODEL_TRAINING,
  "wm_evaluation": WORLD_MODEL_EVALUATION,
  "world_model_evaluation": WORLD_MODEL_EVALUATION,
  "marl_training": MARL_TRAINING,
  "policy_training": MARL_TRAINING,
  "marl_evaluation": MARL_EVALUATION,
  "policy_evaluation": MARL_EVALUATION,
}

WM_TRAINING_KEYS = {
  "training_wm_total_loss",
  "training_obs_loss",
  "training_mean_mse",
  "training_reward_loss",
  "training_done_loss",
  "training_symbolic_loss",
  "training_residual_loss",
}

WM_EVALUATION_KEYS = {
  "grid_mismatch",
  "terrain_mismatch",
  "block_mismatch",
  "entity_mismatch",
  "self_mse",
  "rvr",
  "determinable_mismatch",
  "undeterminable_mismatch",
  "determinable_count",
  "checkpoint_step",
}

MARL_TRAINING_KEYS = {
  "training_real_reward",
  "training_imagined_reward",
  "training_imagined_reward_mean_step",
  "training_sampled_imagined_reward",
  "policy_loss",
  "value_loss",
  "entropy",
  "policy_total_loss",
  "imagination_world_model_loss",
  "imagination_world_model_obs_loss",
  "imagination_world_model_reward_loss",
  "imagination_world_model_done_loss",
  "real_ratio",
  "imagined_ratio",
  "model_rollout_length",
  "model_buffer_size",
  "model_batch_size",
  "real_batch_size",
  "imagined_batch_size",
  "step_time_ms",
}

MARL_EVALUATION_KEYS = {
  "eval_real_reward",
  "eval_real_reward_std",
  "eval_real_episode_length",
  "eval_imagined_reward",
  "eval_imagined_reward_std",
  "eval_imagined_episode_length",
  "real_imagined_reward_gap",
  "planning_real_return",
  "planning_imagined_return",
  "planning_action_entropy",
  "planning_step_time_ms",
  "planning_batch_size",
  "planning_horizon",
  "planning_device",
}

INFO_PANELS = {
  GENERAL: {
    "name": "00 Information - experiment overview",
    "title": "Experiment configuration overview",
    "what": "This section documents the baseline identity, model architecture, downstream control algorithm, and Partial Symbolic Transition Rules (PSTRs) used by the run.",
    "interpretation": "Use these panels to verify that runs compare the intended baselines, architectures, symbolic coverage, and evaluation regimes before interpreting learning curves.",
    "expected": "Runs in the same comparison group should differ only in the controlled baseline factors such as NS-MAWM integration or symbolic coverage.",
    "failure": "Unexpected differences in architecture, dataset generation, action space, or evaluation budget can invalidate a baseline comparison.",
  },
  WORLD_MODEL_TRAINING: {
    "name": "00 Information - losses",
    "title": "World Model training losses",
    "what": "This section tracks the optimization signals of the VAE + MDN-RNN world model and, when enabled, the NS-MAWM auxiliary objectives.",
    "interpretation": "Lower values indicate that the model is fitting latent dynamics, rewards, terminal events, and symbolic constraints more accurately. Compare runs with the same dataset and evaluation budget.",
    "expected": "The total loss and its main components should decrease early and then stabilize. Symbolic and residual losses should be meaningful only for NS-MAWM variants.",
    "failure": "Flat or exploding losses usually indicate an incompatible checkpoint, insufficient data, unstable learning rate, or a mismatch between the baseline variant and the training inputs.",
  },
  WORLD_MODEL_EVALUATION: {
    "name": "00 Information - prediction fidelity",
    "title": "World Model evaluation and compounding error",
    "what": "This section compares imagined observations with real Gridcraft observations over one-step and multi-step horizons. The video panel shows the real full grid on the left, real observations, and imagined observations when available.",
    "interpretation": "Lower mismatch, self MSE, and RVR are better. Horizon-prefixed metrics show how prediction error compounds when the world model is rolled forward. In videos, stable imagined observations should remain visually close to the real observation panel.",
    "expected": "Good models have low one-step mismatch and slower degradation as the horizon increases. Projection variants should strongly reduce post-projection rule violations.",
    "failure": "High terrain/block/entity mismatch indicates poor observation fidelity. Increasing horizon errors indicate compounding model drift or inconsistent symbolic correction.",
  },
  MARL_TRAINING: {
    "name": "00 Information - policy optimization",
    "title": "MARL policy training",
    "what": "This section tracks policy optimization signals for model-free real training and imagined-only training.",
    "interpretation": "Higher training rewards are better. Policy and value losses describe optimizer behavior and should be interpreted together with rewards.",
    "expected": "Training reward should improve or stabilize. Imagined-only training can improve in the dream environment while still showing a transfer gap in real evaluation.",
    "failure": "Unstable losses or reward collapse can indicate poor exploration, an inaccurate dream environment, or hyperparameters that are too aggressive.",
  },
  MARL_EVALUATION: {
    "name": "00 Information - downstream performance",
    "title": "MARL downstream evaluation",
    "what": "This section evaluates policies or planners in real and imagined Gridcraft environments after or during training. The video panel shows the real rollout; model-based baselines also include imagined observations for comparison.",
    "interpretation": "Higher real reward is better. The real-imagined reward gap measures sim-to-real mismatch. MPC-CEM planning metrics report planned imagined return and realized real return. For non-model-based baselines, the video intentionally omits imagined observations.",
    "expected": "Useful world models reduce the real-imagined gap and improve real evaluation reward for imagined-only training or MPC-CEM planning.",
    "failure": "A large positive or negative gap indicates model bias. High imagined return with low real return means the policy or planner is exploiting model errors.",
  },
}

PSTR_RULES = [
  {
    "id": "PSTR_INDIV_STATIC_TERRAIN_SHIFT",
    "scope": "individual",
    "input": "Agent local grid and movement action.",
    "predicted": "Terrain cells that can be shifted in the egocentric observation.",
    "unknown": "New border cells and stochastic world changes.",
    "failure": "No terrain shift is imposed when movement cannot be locally determined.",
  },
  {
    "id": "PSTR_INDIV_STATIC_BLOCK_SHIFT",
    "scope": "individual",
    "input": "Agent local grid and movement action.",
    "predicted": "Static block cells shifted with the egocentric frame.",
    "unknown": "New border cells and stochastic block changes.",
    "failure": "Harvest/crafting side effects are handled by separate rules only when deterministic.",
  },
  {
    "id": "PSTR_INDIV_CENTER_AGENT",
    "scope": "individual",
    "input": "Any live agent observation.",
    "predicted": "Controlled agent entity at the center of its own local observation.",
    "unknown": "Other entity cells unless determined by joint alignment.",
    "failure": "Dead-agent observations are not explicitly modeled by this rule.",
  },
  {
    "id": "PSTR_INDIV_BLOCKED_WATER",
    "scope": "individual",
    "input": "Movement action targeting an observable water cell.",
    "predicted": "Movement is blocked; terrain/block frame does not shift.",
    "unknown": "Inventory, mobs, drops and unrelated cells.",
    "failure": "Only applies when the target cell is inside the local observation.",
  },
  {
    "id": "PSTR_INDIV_BLOCKED_TREE_STONE",
    "scope": "individual",
    "input": "Movement action targeting observable tree or stone.",
    "predicted": "Movement is blocked; terrain/block frame does not shift.",
    "unknown": "Changes caused by other agents or stochastic events.",
    "failure": "Does not decide whether a later harvest succeeds.",
  },
  {
    "id": "PSTR_INDIV_BLOCKED_ENTITY",
    "scope": "individual",
    "input": "Movement action targeting observable agent or mob.",
    "predicted": "Movement is blocked locally.",
    "unknown": "Future mob movement and combat outcomes.",
    "failure": "Entity identity beyond agent/mob class is not available in local observations.",
  },
  {
    "id": "PSTR_INDIV_HARVEST_TREE_WOOD",
    "scope": "individual",
    "input": "Harvest action with an adjacent observable tree.",
    "predicted": "Tree block becomes empty and wood inventory increases by one.",
    "unknown": "Apple drop remains unknown because it is stochastic.",
    "failure": "Only the first adjacent harvestable tree in environment order is predicted.",
  },
  {
    "id": "PSTR_INDIV_HARVEST_STONE_PICKAXE",
    "scope": "individual",
    "input": "Harvest action, adjacent observable stone, and pickaxe evidence in self vector.",
    "predicted": "Stone block becomes empty and stone inventory increases by one.",
    "unknown": "Hunger cost unless counters are available.",
    "failure": "Equipped tool is approximated from inventory unless explicit memory is added.",
  },
  {
    "id": "PSTR_INDIV_PICKUP_ITEM",
    "scope": "individual",
    "input": "Pickup action with item entity on the agent cell; item type from memory if known.",
    "predicted": "Center entity is cleared/agent remains; inventory item increases when item memory is known.",
    "unknown": "Inventory item type/count if not present in symbolic memory.",
    "failure": "Local observations encode item entity but not item type.",
  },
  {
    "id": "PSTR_INDIV_EAT_APPLE",
    "scope": "individual",
    "input": "Eat action, apple inventory > 0, hunger < max.",
    "predicted": "Apple decreases by one and hunger increases up to max.",
    "unknown": "Unrelated inventory and world cells.",
    "failure": "No effect is predicted when preconditions are not satisfied.",
  },
  {
    "id": "PSTR_INDIV_CRAFT_PLANK",
    "scope": "individual",
    "input": "Craft plank action and wood >= 1.",
    "predicted": "Wood decreases by one and planks increase by two.",
    "unknown": "World grid remains unknown except center agent.",
    "failure": "No effect is predicted when resources are insufficient.",
  },
  {
    "id": "PSTR_INDIV_CRAFT_STICK",
    "scope": "individual",
    "input": "Craft stick action and plank >= 2.",
    "predicted": "Planks decrease by two and sticks increase by four.",
    "unknown": "World grid remains unknown except center agent.",
    "failure": "No effect is predicted when resources are insufficient.",
  },
  {
    "id": "PSTR_INDIV_CRAFT_TOOLS",
    "scope": "individual",
    "input": "Tool craft action with required sticks and planks/stone.",
    "predicted": "Required resources decrease and target tool inventory increases.",
    "unknown": "Equipped slot is not represented in the tabular self vector.",
    "failure": "No effect is predicted when resources are insufficient.",
  },
  {
    "id": "PSTR_INDIV_ATTACK_MOB_LOCAL",
    "scope": "individual",
    "input": "Attack action with adjacent observable mob and optional mob hp in memory.",
    "predicted": "Mob hp memory is decremented when known.",
    "unknown": "Mob drop and future mob movement remain unknown.",
    "failure": "Without mob hp memory, no observation feature is masked.",
  },
  {
    "id": "PSTR_INDIV_HUNGER_COST_KNOWN_COUNTER",
    "scope": "individual",
    "input": "Move/harvest/attack plus symbolic hunger counters in memory.",
    "predicted": "Hunger decreases when the known counter reaches the configured interval.",
    "unknown": "Hunger remains unknown if counters are absent.",
    "failure": "Counters must be maintained by symbolic memory.",
  },
  {
    "id": "PSTR_JOINT_RELATIVE_AGENT_ALIGNMENT",
    "scope": "joint",
    "input": "Joint observations where agents see each other consistently.",
    "predicted": "Relative offsets between agents in symbolic memory.",
    "unknown": "Ambiguous identities when several agents are visible without disambiguation.",
    "failure": "No offset is inferred without a consistent reciprocal observation.",
  },
  {
    "id": "PSTR_JOINT_MAP_FUSION",
    "scope": "joint",
    "input": "Joint observations and known relative agent positions.",
    "predicted": "Shared reconstructed terrain/block/entity memory.",
    "unknown": "Cells never observed or not aligned into the shared frame.",
    "failure": "Conflicting observations currently use latest write semantics.",
  },
  {
    "id": "PSTR_JOINT_GLOBAL_STATIC_QUERY",
    "scope": "joint",
    "input": "Future local cells covered by reconstructed shared map.",
    "predicted": "Terrain and block values from shared map.",
    "unknown": "Cells outside reconstructed coverage.",
    "failure": "Depends on correct relative alignment.",
  },
  {
    "id": "PSTR_JOINT_MULTI_AGENT_COLLISION",
    "scope": "joint",
    "input": "Joint movement actions and known relative positions.",
    "predicted": "Colliding/swap movements are blocked when determinable.",
    "unknown": "Unaligned agents and hidden entities.",
    "failure": "Resolution is conservative when positions are unknown.",
  },
  {
    "id": "PSTR_JOINT_AGENT_ENTITY_PREDICTION",
    "scope": "joint",
    "input": "Known relative positions after joint actions.",
    "predicted": "Other agents inserted into each local entity plane when visible.",
    "unknown": "Unaligned or out-of-view agents.",
    "failure": "Does not distinguish agent identities in the local entity channel.",
  },
  {
    "id": "PSTR_JOINT_SHARED_WORLD_UPDATE",
    "scope": "joint",
    "input": "Deterministic harvest that changes a known shared-map block.",
    "predicted": "Shared reconstructed block map is updated for all agents.",
    "unknown": "Stochastic apple effects.",
    "failure": "Requires a known agent position in shared memory.",
  },
  {
    "id": "PSTR_JOINT_SHARED_ITEM_UPDATE",
    "scope": "joint",
    "input": "Deterministic pickup on a known shared-map item cell.",
    "predicted": "Shared reconstructed item/entity state is updated.",
    "unknown": "Item type/count if not known in memory.",
    "failure": "Local observations only encode item presence.",
  },
]


def general_panel_payload(wandb, config=None):
  config = config or {}
  return {
    f"{GENERAL}/01 PSTR rule catalog": pstr_rule_table(wandb),
    f"{GENERAL}/02 Baseline configuration": baseline_config_table(wandb, config),
    f"{GENERAL}/03 Architecture and algorithm configuration": architecture_config_table(wandb, config),
  }


def pstr_rule_table(wandb):
  return wandb.Table(
    columns=["PSTR id", "Scope", "Input requirement", "Predicted features", "Unknown features", "Failure/uncertainty cases"],
    data=[[rule["id"], rule["scope"], rule["input"], rule["predicted"], rule["unknown"], rule["failure"]] for rule in PSTR_RULES],
  )


def baseline_config_table(wandb, config):
  rows = []
  for key in (
    "baseline_id",
    "baseline_slug",
    "family",
    "wm_arch",
    "data_generator",
    "environment",
    "coverage",
    "integration",
    "eval_regime",
    "training_regime",
    "ns_variant",
    "seed",
    "phase",
    "policy_baseline",
  ):
    if key in config:
      rows.append([key, str(config[key])])
  if not rows:
    rows = [["configuration", "not provided"]]
  return wandb.Table(columns=["Field", "Value"], data=rows)


def architecture_config_table(wandb, config):
  rows = [
    ["Observation encoding", "Structured tabular Gridcraft observation: 7x7 local grid with terrain/block/entity categorical planes plus 11 self features."],
    ["Observation width", "550"],
    ["Action count", "15"],
    ["VAE architecture", "MLP encoder/decoder with two hidden layers of 512 units, latent size 64, categorical grid reconstruction losses, self-vector MSE, and KL tolerance."],
    ["MDN-RNN architecture", "LSTM hidden size 128, 5 Gaussian mixture components per latent dimension, reward regression, and done-logit prediction."],
    ["World model baseline", str(config.get("wm_arch", "vae_mdn_rnn"))],
    ["NS-MAWM integration", str(config.get("integration", "none"))],
    ["MARL/control algorithm", _algorithm_description(config)],
  ]
  return wandb.Table(columns=["Component", "Description"], data=rows)


def _algorithm_description(config):
  policy = config.get("policy_baseline")
  if policy == "real_mappo":
    return "Legacy model-free mono-agent actor-critic runner trained directly in real Gridcraft."
  if policy == "imagined_mappo":
    return "Legacy MAPPO-in-WM runner trained only in the Gridcraft dream environment and periodically evaluated in real Gridcraft."
  if config.get("downstream_policy_backend") == "dyna_actor_critic_world_model_only":
    return "Diagnostic dyna_actor_critic runner trained only from world-model rollouts."
  if policy == "mpc_cem":
    return "MPC-CEM planner acting in real Gridcraft by evaluating candidate action sequences inside the learned world model."
  if config.get("phase") == "world_model":
    return "No downstream policy is trained in this world-model-only phase."
  return "Not specified for this run."


def route_metrics(metrics, namespace=None):
  return {route_metric_key(key, namespace): value for key, value in metrics.items()}


def route_metric_key(key, namespace=None):
  key = str(key)
  if any(key.startswith(f"{section}/") for section in SECTIONS):
    return key
  if namespace in NAMESPACE_TO_SECTION:
    return f"{NAMESPACE_TO_SECTION[namespace]}/{canonical_metric_name(key)}"

  clean = key
  if clean.startswith("eval/"):
    clean = clean.split("/", 1)[1]
    return f"{WORLD_MODEL_EVALUATION}/{canonical_metric_name(clean)}"
  if clean.startswith("compounding_error/"):
    clean = "compounding_" + clean.split("/", 1)[1]
    return f"{WORLD_MODEL_EVALUATION}/{canonical_metric_name(clean)}"

  canonical = canonical_metric_name(clean)
  if clean in WM_TRAINING_KEYS:
    return f"{WORLD_MODEL_TRAINING}/{canonical}"
  if clean in WM_EVALUATION_KEYS or is_horizon_metric(clean):
    return f"{WORLD_MODEL_EVALUATION}/{canonical}"
  if clean in MARL_TRAINING_KEYS:
    return f"{MARL_TRAINING}/{canonical}"
  if clean in MARL_EVALUATION_KEYS:
    return f"{MARL_EVALUATION}/{canonical}"
  return key


def canonical_metric_name(key):
  if key.startswith("eval/"):
    key = key.split("/", 1)[1]
  if key.startswith("horizon_") and "@" in key:
    name, horizon = key.split("@", 1)
    return "compounding_" + name[len("horizon_"):] + "_h" + horizon
  if key.startswith("compounding_error/"):
    return "compounding_" + key.split("/", 1)[1]
  return key.replace("/", "_")


def is_horizon_metric(key):
  return key.startswith("horizon_") and "@" in key


def info_panel_payload(wandb, sections=None):
  payload = {}
  selected = sections or SECTIONS
  for section in selected:
    spec = INFO_PANELS.get(section)
    if not spec:
      continue
    panel_name = f"{section}/{spec['name']}"
    payload[panel_name] = wandb.Html(info_panel_html(section, spec), data_is_not_path=True)
  return payload


def info_panel_specs(sections=None):
  selected = sections or SECTIONS
  return {section: INFO_PANELS[section] for section in selected if section in INFO_PANELS}


def info_panel_html(section, spec):
  return (
    "<div style='font-family: -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif; "
    "font-size: 13px; line-height: 1.45;'>"
    f"<h3 style='margin: 0 0 8px 0;'>{escape(spec['title'])}</h3>"
    "<details open>"
    "<summary style='cursor: pointer; font-weight: 600;'>Information</summary>"
    "<div style='margin-top: 10px;'>"
    f"<p><strong>Section</strong><br>{escape(section)}</p>"
    f"<p><strong>What this panel shows</strong><br>{escape(spec['what'])}</p>"
    f"<p><strong>How to interpret it</strong><br>{escape(spec['interpretation'])}</p>"
    f"<p><strong>Expected trend</strong><br>{escape(spec['expected'])}</p>"
    f"<p><strong>Failure modes</strong><br>{escape(spec['failure'])}</p>"
    "</div>"
    "</details>"
    "</div>"
  )
