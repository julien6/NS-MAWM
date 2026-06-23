from html import escape


GENERAL = "General"
WORLD_MODEL_TRAINING = "World Model training"
WORLD_MODEL_EVALUATION = "World Model evaluation"
MARL_TRAINING = "MARL training"
MARL_EVALUATION = "MARL evaluation"

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
  "policy_loss",
  "value_loss",
  "entropy",
  "policy_total_loss",
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
    "id": "PSTR_STATIC_TERRAIN_PERSISTENCE",
    "description": "For observable cells whose next egocentric position is determined, the terrain category is predicted to persist from the corresponding previous local cell.",
  },
  {
    "id": "PSTR_STATIC_BLOCK_PERSISTENCE",
    "description": "For observable cells whose next egocentric position is determined, the block category is predicted to persist from the corresponding previous local cell.",
  },
  {
    "id": "PSTR_AGENT_CENTER_OCCUPANCY",
    "description": "The local observation is egocentric, so the controlled agent is expected to remain represented at the center entity cell after the transition.",
  },
  {
    "id": "PSTR_BLOCKED_BY_WATER",
    "description": "A movement action targeting an observable water cell is treated as blocked, so the static local terrain and block planes should not shift.",
  },
  {
    "id": "PSTR_BLOCKED_BY_OBSTACLE_BLOCK",
    "description": "A movement action targeting an observable blocking block such as a tree or stone is treated as blocked.",
  },
  {
    "id": "PSTR_BLOCKED_BY_ENTITY",
    "description": "A movement action targeting an observable blocking entity such as a mob or another agent is treated as blocked.",
  },
  {
    "id": "PSTR_EGOCENTRIC_SHIFT_AFTER_MOVE",
    "description": "When movement is locally determined to be successful, static terrain and block planes shift in the opposite direction in the next egocentric observation.",
  },
  {
    "id": "PSTR_CONSERVATIVE_COVERAGE_MASK",
    "description": "Only components considered locally observable and unambiguous are covered by symbolic targets; uncertain crafting, combat, drops, mobs, and inventory effects are left to the neural model.",
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
    columns=["PSTR identifier", "Description"],
    data=[[rule["id"], rule["description"]] for rule in PSTR_RULES],
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
    return "Model-free mono-agent actor-critic runner with MAPPO-compatible logging, trained directly in real Gridcraft."
  if policy == "imagined_mappo":
    return "Mono-agent actor-critic runner trained only in the Gridcraft dream environment and periodically evaluated in real Gridcraft."
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
