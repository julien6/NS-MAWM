#!/usr/bin/env python3
"""Render MPE2 Simple World Comm frames as frameX.png files."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the MPE2 Simple World Comm parallel environment with random "
            "actions and save RGB frames as frame0.png, frame1.png, ..."
        )
    )
    parser.add_argument("--out", default="frames/simple_world_comm")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-cycles", type=int, default=25)
    parser.add_argument(
        "--num-frames",
        type=int,
        default=None,
        help=(
            "Exact number of frames to write, including the initial reset frame. "
            "Defaults to max_cycles + 1 unless the episode ends earlier."
        ),
    )
    parser.add_argument("--num-good", type=int, default=2)
    parser.add_argument("--num-adversaries", type=int, default=4)
    parser.add_argument("--num-obstacles", type=int, default=1)
    parser.add_argument("--num-food", type=int, default=2)
    parser.add_argument("--num-forests", type=int, default=2)
    parser.add_argument("--continuous-actions", action="store_true")
    parser.add_argument("--dynamic-rescaling", action="store_true")
    return parser.parse_args()


def save_frame(frame, out_dir: Path, frame_idx: int) -> None:
    Image.fromarray(frame).save(out_dir / f"frame{frame_idx}.png")


def sample_actions(env) -> dict:
    return {agent: env.action_space(agent).sample() for agent in env.agents}


def main() -> None:
    args = parse_args()

    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "hide")
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

    from mpe2 import simple_world_comm_v3

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    target_frames = args.num_frames or args.max_cycles + 1
    if target_frames < 1:
        raise ValueError("--num-frames must be >= 1")

    env = simple_world_comm_v3.parallel_env(
        num_good=args.num_good,
        num_adversaries=args.num_adversaries,
        num_obstacles=args.num_obstacles,
        num_food=args.num_food,
        max_cycles=max(args.max_cycles, target_frames - 1),
        num_forests=args.num_forests,
        continuous_actions=args.continuous_actions,
        render_mode="rgb_array",
        dynamic_rescaling=args.dynamic_rescaling,
    )

    frame_idx = 0
    try:
        env.reset(seed=args.seed)
        print(
            "Simple World Comm: "
            f"agents={len(env.agents)} max_cycles={args.max_cycles} "
            f"target_frames={target_frames}"
        )

        frame = env.render()
        save_frame(frame, out_dir, frame_idx)
        frame_idx += 1

        while env.agents and frame_idx < target_frames:
            _obs, _rewards, terminations, truncations, _infos = env.step(
                sample_actions(env)
            )
            frame = env.render()
            save_frame(frame, out_dir, frame_idx)
            frame_idx += 1

            if all(terminations.values()) or all(truncations.values()):
                break
    finally:
        env.close()

    print(f"Wrote {frame_idx} PNG frames to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
