from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vGridcraft"))
sys.path.insert(0, str(ROOT / "Gridcraft"))

from vgridcraft import VGridcraftConfig, VectorizedGridcraftEnv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--num-agents", type=int, default=1)
    parser.add_argument("--env-index", type=int, default=0)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--mode", choices=("human", "rgb_array"), default="human")
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--reset-on-done", action="store_true", help="Reset only the displayed environment when it terminates and keep rendering.")
    parser.add_argument("--hold-final-frames", type=int, default=12, help="Number of extra final frames to render before exiting when not resetting.")
    parser.add_argument("--save-video", default=None, help="Optional output path, e.g. /tmp/vgridcraft.mp4")
    parser.add_argument("--save-frame", default=None, help="Optional output image path, e.g. /tmp/vgridcraft.png")
    args = parser.parse_args()

    config = VGridcraftConfig(
        num_agents=args.num_agents,
        max_steps=args.steps,
        seed=args.seed,
        fps=args.fps,
    )
    env = VectorizedGridcraftEnv(
        num_envs=args.num_envs,
        num_agents=args.num_agents,
        device=args.device,
        seed=args.seed,
        config=config,
    )
    generator = torch.Generator(device=env.device)
    generator.manual_seed(args.seed + 1234)
    frames = []
    try:
        env.reset(seed=args.seed)
        for step in range(args.steps):
            frame = env.render(env_index=args.env_index, mode=args.mode)
            if args.mode == "rgb_array" or args.save_video or args.save_frame:
                frame = frame if frame is not None else env.render(env_index=args.env_index, mode="rgb_array")
                frames.append(frame)
            actions = torch.randint(
                0,
                config.action_size,
                (args.num_envs, args.num_agents),
                generator=generator,
                device=env.device,
            )
            _, _, done, truncated, _ = env.step(actions)
            displayed_done = bool((done | truncated)[args.env_index])
            if displayed_done:
                final_frame = env.render(env_index=args.env_index, mode=args.mode)
                if args.mode == "rgb_array" or args.save_video or args.save_frame:
                    final_frame = final_frame if final_frame is not None else env.render(env_index=args.env_index, mode="rgb_array")
                    frames.append(final_frame)
                if args.reset_on_done:
                    env.reset(env_ids=torch.tensor([args.env_index], device=env.device))
                else:
                    print(f"displayed env_index={args.env_index} terminated at step {step + 1}")
                    for _ in range(max(0, args.hold_final_frames)):
                        hold_frame = env.render(env_index=args.env_index, mode=args.mode)
                        if args.mode == "rgb_array" or args.save_video or args.save_frame:
                            hold_frame = hold_frame if hold_frame is not None else env.render(env_index=args.env_index, mode="rgb_array")
                            frames.append(hold_frame)
                    break
        if args.save_frame and frames:
            save_frame(args.save_frame, frames[-1])
        if args.save_video and frames:
            save_video(args.save_video, frames, fps=args.fps)
    finally:
        env.close()


def save_frame(path: str, frame) -> None:
    from PIL import Image

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    Image.fromarray(frame).save(path)
    print(f"saved frame {path}")


def save_video(path: str, frames, fps: int) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    try:
        import imageio.v2 as imageio
    except ImportError as exc:
        raise SystemExit("Install imageio to save videos: ../.venv/bin/python -m pip install imageio imageio-ffmpeg") from exc
    imageio.mimsave(path, frames, fps=fps)
    print(f"saved video {path}")


if __name__ == "__main__":
    main()
