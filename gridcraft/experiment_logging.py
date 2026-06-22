import json
import os
import time

import numpy as np

from wandb_schema import GENERAL, general_panel_payload, info_panel_payload, info_panel_specs, route_metrics


class ExperimentLogger:
  def __init__(self, enabled=False, project="ns-mawm-gridcraft", entity=None, group=None, name=None, tags=None, config=None, mode=None, out_dir="trainlog", info_panels=True, info_sections=None):
    self.enabled = enabled
    self.out_dir = out_dir
    self.run = None
    self._wandb = None
    self.config = config or {}
    os.makedirs(out_dir, exist_ok=True)
    self.info_sections = info_sections
    self.save_json(os.path.join(out_dir, "wandb_panels.json"), info_panel_specs(info_sections) if info_panels else {})
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
      "tags": normalize_wandb_tags(tags or []),
      "config": config or {},
    }
    if mode:
      kwargs["mode"] = mode
    kwargs = {key: value for key, value in kwargs.items() if value is not None}
    self._wandb = wandb
    if wandb.run is not None:
      self.run = wandb.run
      if config:
        try:
          self.run.config.update(config, allow_val_change=True)
        except TypeError:
          self.run.config.update(config)
    else:
      self.run = wandb.init(**kwargs)
    if info_panels:
      self.log_info_panels()

  def log(self, metrics, step=None, namespace=None):
    clean = flatten_metrics(metrics)
    if self.run is not None:
      self._wandb.log(route_metrics(clean, namespace=namespace), step=step)

  def log_summary(self, summary, namespace=None):
    if self.run is not None:
      for key, value in route_metrics(flatten_metrics(summary), namespace=namespace).items():
        self.run.summary[key] = value

  def log_video(self, name, frames, fps=10, step=None, namespace=None):
    if self.run is None or self._wandb is None:
      return
    frames = np.asarray(frames, dtype=np.uint8)
    if frames.ndim != 4 or frames.shape[-1] != 3:
      raise ValueError("video frames must be shaped (T, H, W, 3)")
    video = np.transpose(frames, (0, 3, 1, 2))
    routed = route_metrics({name: self._wandb.Video(video, fps=fps, format="mp4")}, namespace=namespace)
    self._wandb.log(routed, step=step)

  def log_info_panels(self):
    if self.run is not None:
      payload = {}
      if self.info_sections is None or GENERAL in self.info_sections:
        payload.update(general_panel_payload(self._wandb, self.config))
      payload.update(info_panel_payload(self._wandb, self.info_sections))
      self._wandb.log(payload)

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
  panel_group = parser.add_mutually_exclusive_group()
  panel_group.add_argument("--wandb-info-panels", dest="wandb_info_panels", action="store_true", default=True)
  panel_group.add_argument("--no-wandb-info-panels", dest="wandb_info_panels", action="store_false")
  video_group = parser.add_mutually_exclusive_group()
  video_group.add_argument("--wandb-videos", dest="wandb_videos", action="store_true", default=True)
  video_group.add_argument("--no-wandb-videos", dest="wandb_videos", action="store_false")
  parser.add_argument("--video-episodes", type=int, default=1)
  parser.add_argument("--video-max-steps", type=int, default=100)
  parser.add_argument("--video-fps", type=int, default=10)


def should_log_wandb_videos(args):
  return bool(getattr(args, "wandb", False) and getattr(args, "wandb_videos", True))


def logger_from_args(args, config=None, default_group=None, default_name=None, tags=None, info_sections=None, out_dir=None):
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
    out_dir=out_dir or "trainlog",
    info_panels=bool(getattr(args, "wandb_info_panels", True)),
    info_sections=info_sections,
  )


def normalize_wandb_tags(tags, max_len=64):
  normalized = []
  seen = set()
  for tag in tags or []:
    text = str(tag).strip()
    if not text:
      continue
    if len(text) > max_len:
      text = text[:max_len]
    if text not in seen:
      normalized.append(text)
      seen.add(text)
  return normalized


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
