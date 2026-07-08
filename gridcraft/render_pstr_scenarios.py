from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from pstr_scenarios import ACTION_NAMES, expected_rules, run_scenario, scenario_ids, scenario_specs

SCENARIO_GRID_WIDTH = 11
SCENARIO_TILE_SIZE = 48


def main() -> None:
  parser = argparse.ArgumentParser(description="Render visual explanations for Gridcraft PSTR rules.")
  parser.add_argument("--list", action="store_true", help="List available PSTR scenarios and exit.")
  parser.add_argument("--rule", default="all", help="PSTR id to render, or 'all'.")
  parser.add_argument("--out-dir", default="pstr_viz", help="Output directory for PNG files and catalog.json.")
  parser.add_argument("--format", choices=("gif", "png", "both", "diagram"), default="gif", help="Output format. PNG writes the final frame; diagram writes the real-observation => action => symbolic-observation figure.")
  parser.add_argument("--steps", type=int, default=6, help="Number of frames per scenario, including the initial real-observation frame.")
  parser.add_argument("--fps", type=float, default=0.75, help="GIF playback speed.")
  parser.add_argument("--save-frames", action="store_true", help="Save every frame as a numbered PNG.")
  parser.add_argument("--mode", choices=("rgb_array", "human"), default="rgb_array")
  parser.add_argument("--hold", type=float, default=4.0, help="Seconds to keep the human preview open after one animation loop.")
  args = parser.parse_args()

  specs = scenario_specs()
  if args.list:
    for rule_id in scenario_ids():
      spec = specs[rule_id]
      actions = ", ".join(f"{agent}={ACTION_NAMES.get(int(action), str(action))}" for agent, action in spec.action.items())
      print(f"{rule_id}: {actions}")
    return

  selected = scenario_ids() if args.rule == "all" else [args.rule]
  out_dir = Path(args.out_dir)
  out_dir.mkdir(parents=True, exist_ok=True)
  catalog = []
  for rule_id in selected:
    result = run_scenario(rule_id, render_mode="rgb_array", steps=args.steps)
    image_path = None
    gif_path = None
    diagram_path = None
    frame_paths = []
    if args.format in ("gif", "both"):
      gif_path = out_dir / f"{rule_id}.gif"
      _save_gif(result.frames, gif_path, fps=args.fps)
      print(f"saved {gif_path}")
    if args.format in ("png", "both"):
      image_path = out_dir / f"{rule_id}.png"
      Image.fromarray(result.frames[-1]).save(image_path)
      print(f"saved {image_path}")
    if args.format == "diagram":
      diagram_path = out_dir / f"{rule_id}.png"
      _save_diagram(result, diagram_path)
      print(f"saved {diagram_path}")
    if args.save_frames:
      frame_dir = out_dir / f"{rule_id}_frames"
      frame_dir.mkdir(parents=True, exist_ok=True)
      for idx, frame in enumerate(result.frames):
        frame_path = frame_dir / f"step_{idx:03d}.png"
        Image.fromarray(frame).save(frame_path)
        frame_paths.append(str(frame_path))
    if args.mode == "human":
      _show_frames(result.frames, fps=args.fps, hold_seconds=args.hold, title=rule_id)
    catalog.append({
      "rule_id": result.rule_id,
      "description": result.description,
      "actions": result.action_names,
      "expected_rules": list(expected_rules(rule_id)),
      "gif": str(gif_path) if gif_path is not None else None,
      "image": str(image_path) if image_path is not None else None,
      "diagram": str(diagram_path) if diagram_path is not None else None,
      "frames": frame_paths,
      "actions_by_step": result.actions_by_step,
      "rewards_by_step": result.rewards_by_step,
      "done_by_step": result.done_by_step,
      "input_ascii": result.input_ascii,
      "output_ascii": result.output_ascii,
      "output_ascii_by_step": result.output_ascii_by_step,
      "triggered_rules": _triggered_rule_ids(result.report),
      "triggered_rules_by_step": [_triggered_rule_ids(report) for report in result.reports_by_step],
      "report": _jsonable(result.report),
    })

  catalog_path = out_dir / "catalog.json"
  catalog_path.write_text(json.dumps(catalog, indent=2, sort_keys=True), encoding="utf-8")
  print(f"saved {catalog_path}")


def _triggered_rule_ids(report: dict[str, Any]) -> list[str]:
  rules = report.get("rules", [])
  if isinstance(rules, dict):
    return sorted(str(rule_id) for rule_id in rules)
  return sorted({str(row.get("rule_id")) for row in rules if isinstance(row, dict) and row.get("rule_id")})


def _jsonable(value: Any) -> Any:
  if isinstance(value, dict):
    return {str(key): _jsonable(item) for key, item in value.items()}
  if isinstance(value, (list, tuple)):
    return [_jsonable(item) for item in value]
  if isinstance(value, np.ndarray):
    return value.tolist()
  if isinstance(value, np.generic):
    return value.item()
  return value


def _save_gif(frames: list[np.ndarray], path: Path, fps: float) -> None:
  duration = int(1000 / max(0.1, fps))
  images = [Image.fromarray(frame).convert("P", palette=Image.ADAPTIVE) for frame in frames]
  images[0].save(path, save_all=True, append_images=images[1:], duration=duration, loop=0, optimize=False)


def _save_diagram(result, path: Path) -> None:
  diagram = build_diagram_image(result)
  diagram.save(path)


def build_diagram_image(result) -> Image.Image:
  step = 1 if len(result.plain_frames) > 1 else 0
  real_panel, symbolic_panel = diagram_panels(result)
  action = result.actions_by_step[0] if result.actions_by_step else {"all": "initial"}
  action_text = ", ".join(f"{agent}={name}" for agent, name in sorted(action.items()))

  real_image = _label_panel(real_panel, "Real observation")
  symbolic_image = _label_panel(symbolic_panel, "Partial symbolic prediction")
  action_image = _action_panel(action_text, height=real_image.height)

  margin = 24
  width = real_image.width + action_image.width + symbolic_image.width + margin * 4
  height = max(real_image.height, action_image.height, symbolic_image.height) + margin * 2
  canvas = Image.new("RGB", (width, height), (245, 245, 245))
  x = margin
  y = margin
  canvas.paste(real_image, (x, y))
  x += real_image.width + margin
  canvas.paste(action_image, (x, y))
  x += action_image.width + margin
  canvas.paste(symbolic_image, (x, y))
  return canvas


def diagram_panels(result) -> tuple[np.ndarray, np.ndarray]:
  step = 1 if len(result.plain_frames) > 1 else 0
  input_frame = result.plain_frames[0]
  output_frame = result.plain_frames[step]
  world_width = _scenario_world_width()
  panel_width = max(1, (output_frame.shape[1] - world_width) // 2)
  panel_height = _diagram_panel_height(result, output_frame)
  real_panel = input_frame[:panel_height, world_width:world_width + panel_width]
  symbolic_panel = output_frame[:panel_height, world_width + panel_width:world_width + 2 * panel_width]
  return real_panel, symbolic_panel


def _scenario_world_width() -> int:
  return SCENARIO_GRID_WIDTH * SCENARIO_TILE_SIZE


def _diagram_panel_height(result, frame: np.ndarray) -> int:
  num_agents = max(1, len(result.input_observations))
  tile_size = 48
  padding = max(2, tile_size // 4)
  observation_tile_size = max(2, (3 * tile_size) // 7)
  observation_size = 7 * observation_tile_size
  inventory_height = padding + tile_size + padding + 3 * tile_size
  section_height = padding + max(inventory_height, observation_size) + padding
  return min(frame.shape[0], max(1, num_agents * section_height))


def _label_panel(panel: np.ndarray, title: str) -> Image.Image:
  image = Image.fromarray(panel).convert("RGB")
  font = ImageFont.load_default()
  title_height = 34
  labeled = Image.new("RGB", (image.width, image.height + title_height), (20, 20, 20))
  draw = ImageDraw.Draw(labeled)
  bbox = draw.textbbox((0, 0), title, font=font)
  draw.text(((image.width - (bbox[2] - bbox[0])) // 2, 10), title, fill=(255, 255, 255), font=font)
  labeled.paste(image, (0, title_height))
  return labeled


def _action_panel(action_text: str, height: int) -> Image.Image:
  font_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
  font = ImageFont.truetype(str(font_path), 22) if font_path.exists() else ImageFont.load_default()
  arrow_font = ImageFont.truetype(str(font_path), 42) if font_path.exists() else ImageFont.load_default()
  lines = [action_text, "\u27f9"]
  probe = Image.new("RGB", (1, 1), (245, 245, 245))
  draw = ImageDraw.Draw(probe)
  width = 220
  for text, active_font in ((lines[0], font), (lines[1], arrow_font)):
    bbox = draw.textbbox((0, 0), text, font=active_font)
    width = max(width, bbox[2] - bbox[0] + 48)
  image = Image.new("RGB", (width, height), (245, 245, 245))
  draw = ImageDraw.Draw(image)
  total_height = 0
  bboxes = []
  for text, active_font in ((lines[0], font), (lines[1], arrow_font)):
    bbox = draw.textbbox((0, 0), text, font=active_font)
    bboxes.append((bbox, active_font, text))
    total_height += bbox[3] - bbox[1] + 12
  y = (height - total_height) // 2
  for bbox, active_font, text in bboxes:
    text_width = bbox[2] - bbox[0]
    draw.text(((width - text_width) // 2, y), text, fill=(20, 20, 20), font=active_font)
    y += bbox[3] - bbox[1] + 12
  return image


def _show_frames(frames: list[np.ndarray], fps: float, hold_seconds: float, title: str) -> None:
  import pygame

  pygame.init()
  pygame.display.set_caption(title)
  screen = pygame.display.set_mode((frames[0].shape[1], frames[0].shape[0]))
  deadline = pygame.time.get_ticks() + int(max(0.1, hold_seconds) * 1000)
  clock = pygame.time.Clock()
  running = True
  frame_idx = 0
  while running and pygame.time.get_ticks() < deadline:
    for event in pygame.event.get():
      if event.type == pygame.QUIT:
        running = False
    frame = frames[frame_idx % len(frames)]
    surface = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
    screen.blit(surface, (0, 0))
    pygame.display.flip()
    frame_idx += 1
    clock.tick(max(1, fps))
  pygame.display.quit()


if __name__ == "__main__":
  main()
