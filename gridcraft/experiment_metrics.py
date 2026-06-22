import numpy as np


def auc(values):
  values = np.asarray(values, dtype=np.float32)
  if values.size == 0:
    return 0.0
  return float(np.trapz(values))


def time_to_threshold(values, threshold):
  for index, value in enumerate(values):
    if value >= threshold:
      return int(index)
  return -1


def summarize_learning_curve(values, prefix, reward_threshold=None):
  values = np.asarray(values, dtype=np.float32)
  if values.size == 0:
    return {
      f"{prefix}_mean": 0.0,
      f"{prefix}_standard_deviation": 0.0,
      f"AUC_{prefix}": 0.0,
    }
  summary = {
    f"{prefix}_mean": float(np.mean(values)),
    f"{prefix}_standard_deviation": float(np.std(values)),
    f"{prefix}_final": float(values[-1]),
    f"AUC_{prefix}": auc(values),
  }
  if reward_threshold is not None:
    summary[f"time_to_{prefix}_threshold"] = time_to_threshold(values, reward_threshold)
  return summary


def summarize_compounding_error(horizon_metrics):
  result = {}
  for key, value in horizon_metrics.items():
    if key.startswith("horizon_grid_mismatch@"):
      horizon = key.split("@", 1)[1]
      result[f"compounding_error/grid_mismatch_h{horizon}"] = value
    elif key.startswith("horizon_rvr@"):
      horizon = key.split("@", 1)[1]
      result[f"compounding_error/rvr_h{horizon}"] = value
    elif key.startswith("horizon_self_mse@"):
      horizon = key.split("@", 1)[1]
      result[f"compounding_error/self_mse_h{horizon}"] = value
  return result


def flatten_variant_summary(summary):
  flat = {}
  for variant, metrics in summary.items():
    for key, value in metrics.items():
      flat[f"{variant}/{key}"] = value
    flat.update({f"{variant}/{key}": value for key, value in summarize_compounding_error(metrics).items()})
  return flat
