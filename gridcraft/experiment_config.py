from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class BaselineConfig:
  baseline_id: str
  family: str
  wm_arch: str
  data_generator: str
  environment: str
  coverage: float
  integration: str
  eval_regime: str
  training_regime: str
  ns_variant: str
  seed: int = 1

  @property
  def baseline_slug(self):
    parts = [
      self.baseline_id,
      _slug(self.family),
      _slug(self.wm_arch),
      _slug(self.data_generator),
      _slug(self.environment),
      _slug(self.integration),
      _slug(f"{self.eval_regime}-{self.training_regime}"),
      _slug(self.ns_variant),
      f"k{self.coverage}",
    ]
    return "_".join(parts)

  def to_dict(self):
    data = asdict(self)
    data["baseline_slug"] = self.baseline_slug
    return data


def _slug(value):
  return str(value).strip().lower().replace("_", "-").replace(" ", "-")


GRIDCRAFT_BASELINES = [
  BaselineConfig("B00", "model_free_control", "none", "real", "Gridcraft", 0.0, "none", "SV", "SV", "neural"),
  BaselineConfig("B10", "neural_wm", "vae_mdn_rnn", "random", "Gridcraft", 0.0, "none", "SV", "SV", "neural"),
  BaselineConfig("B19", "ns_mawm_coverage", "vae_mdn_rnn", "random", "Gridcraft", 0.1, "regularization", "UV", "MV", "regularization"),
  BaselineConfig("B20", "ns_mawm_coverage", "vae_mdn_rnn", "random", "Gridcraft", 0.3, "regularization", "UV", "MV", "regularization"),
  BaselineConfig("B21", "ns_mawm_coverage", "vae_mdn_rnn", "random", "Gridcraft", 0.5, "regularization", "UV", "MV", "regularization"),
  BaselineConfig("B22", "ns_mawm_coverage", "vae_mdn_rnn", "random", "Gridcraft", 0.7, "regularization", "UV", "MV", "regularization"),
  BaselineConfig("B23", "ns_mawm_coverage", "vae_mdn_rnn", "random", "Gridcraft", 1.0, "regularization", "UV", "MV", "regularization"),
  BaselineConfig("B24", "ns_mawm_strategy", "vae_mdn_rnn", "random", "Gridcraft", 0.3, "regularization", "UV", "MV", "regularization"),
  BaselineConfig("B25", "ns_mawm_strategy", "vae_mdn_rnn", "random", "Gridcraft", 0.3, "projection", "UV", "MV", "projection"),
  BaselineConfig("B26", "ns_mawm_strategy", "vae_mdn_rnn", "random", "Gridcraft", 0.3, "residual", "UV", "MV", "residual"),
  BaselineConfig("B27", "ns_mawm_strategy", "vae_mdn_rnn", "random", "Gridcraft", 0.6, "regularization", "UV", "MV", "regularization"),
  BaselineConfig("B28", "ns_mawm_strategy", "vae_mdn_rnn", "random", "Gridcraft", 0.6, "projection", "UV", "MV", "projection"),
  BaselineConfig("B29", "ns_mawm_strategy", "vae_mdn_rnn", "random", "Gridcraft", 0.6, "residual", "UV", "MV", "residual"),
  BaselineConfig("B36", "data_generator_ablation", "vae_mdn_rnn", "random", "Gridcraft", 0.0, "none", "UV", "MV", "neural"),
  BaselineConfig("B41", "ns_mawm_data_generator", "vae_mdn_rnn", "random", "Gridcraft", 0.3, "regularization", "UV", "MV", "regularization"),
]


def get_baseline(baseline_id):
  for baseline in GRIDCRAFT_BASELINES:
    if baseline.baseline_id == baseline_id:
      return baseline
  valid = ", ".join(b.baseline_id for b in GRIDCRAFT_BASELINES)
  raise ValueError(f"unknown baseline_id {baseline_id}; valid ids: {valid}")


def list_baselines():
  return GRIDCRAFT_BASELINES
