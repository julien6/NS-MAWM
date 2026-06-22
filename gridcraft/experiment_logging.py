import json
import os
import time


class ExperimentLogger:
  def __init__(self, enabled=False, project="ns-mawm-gridcraft", entity=None, group=None, name=None, tags=None, config=None, mode=None, out_dir="trainlog"):
    self.enabled = enabled
    self.out_dir = out_dir
    self.run = None
    self._wandb = None
    os.makedirs(out_dir, exist_ok=True)
    if not enabled:
      return
    try:
      import wandb
    except ImportError:
      print("wandb is not installed; continuing without W&B logging")
      return
    kwargs = {
      "project": project,
      "entity": entity,
      "group": group,
      "name": name,
      "tags": tags or [],
      "config": config or {},
    }
    if mode:
      kwargs["mode"] = mode
    kwargs = {key: value for key, value in kwargs.items() if value is not None}
    self._wandb = wandb
    self.run = wandb.init(**kwargs)

  def log(self, metrics, step=None):
    clean = flatten_metrics(metrics)
    if self.run is not None:
      self._wandb.log(clean, step=step)

  def log_summary(self, summary):
    if self.run is not None:
      for key, value in flatten_metrics(summary).items():
        self.run.summary[key] = value

  def save_json(self, path, payload):
    out_dir = os.path.dirname(path)
    if out_dir:
      os.makedirs(out_dir, exist_ok=True)
    with open(path, "w") as f:
      json.dump(payload, f, indent=2)
    if self.run is not None:
      self._wandb.save(path, base_path=os.path.dirname(path) or ".")

  def finish(self):
    if self.run is not None:
      self._wandb.finish()


def add_wandb_args(parser):
  parser.add_argument("--wandb", action="store_true")
  parser.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "ns-mawm-gridcraft"))
  parser.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY"))
  parser.add_argument("--wandb-group", default=None)
  parser.add_argument("--wandb-name", default=None)
  parser.add_argument("--wandb-mode", default=os.environ.get("WANDB_MODE"))
  parser.add_argument("--wandb-tags", nargs="*", default=[])


def logger_from_args(args, config=None, default_group=None, default_name=None, tags=None):
  merged_tags = list(tags or []) + list(getattr(args, "wandb_tags", []) or [])
  return ExperimentLogger(
    enabled=bool(getattr(args, "wandb", False)),
    project=getattr(args, "wandb_project", "ns-mawm-gridcraft"),
    entity=getattr(args, "wandb_entity", None),
    group=getattr(args, "wandb_group", None) or default_group,
    name=getattr(args, "wandb_name", None) or default_name,
    tags=merged_tags,
    config=config,
    mode=getattr(args, "wandb_mode", None),
  )


def flatten_metrics(metrics, prefix=""):
  flat = {}
  for key, value in metrics.items():
    name = f"{prefix}/{key}" if prefix else str(key)
    if isinstance(value, dict):
      flat.update(flatten_metrics(value, name))
    elif isinstance(value, (int, float, str, bool)) or value is None:
      flat[name] = value
    else:
      try:
        flat[name] = float(value)
      except (TypeError, ValueError):
        flat[name] = str(value)
  return flat


def now_ms():
  return int(time.time() * 1000)
