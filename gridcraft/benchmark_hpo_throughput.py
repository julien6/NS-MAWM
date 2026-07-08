from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "gridcraft"))
sys.path.insert(0, str(ROOT / "vGridcraft"))

from resource_profile import build_profile, detect_hardware  # noqa: E402


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _time(device: torch.device, fn) -> tuple[float, Any]:
    _sync(device)
    start = time.time()
    result = fn()
    _sync(device)
    return time.time() - start, result


def _gpu_util() -> float | None:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=2,
        )
        values = [float(line.strip()) for line in out.splitlines() if line.strip()]
        return max(values) if values else None
    except Exception:
        return None


def _cpu_snapshot() -> dict[str, float]:
    try:
        import psutil  # type: ignore

        return {
            "cpu_percent": float(psutil.cpu_percent(interval=0.2)),
            "ram_available_gib": float(psutil.virtual_memory().available / (1024**3)),
        }
    except Exception:
        return {"cpu_percent": 0.0, "ram_available_gib": 0.0}


def benchmark_env(device: torch.device, values: dict[str, Any], args) -> dict[str, Any]:
    from vgridcraft import VGridcraftConfig, VectorizedGridcraftEnv

    num_envs = int(args.num_envs or values.get("WM_NUM_ENVS", 256))
    num_agents = int(args.num_agents)
    steps = int(args.env_steps)
    cfg = VGridcraftConfig(num_agents=num_agents, max_steps=500, seed=args.seed)
    env = VectorizedGridcraftEnv(num_envs=num_envs, num_agents=num_agents, device=device, seed=args.seed, config=cfg)
    gen = torch.Generator(device=device)
    gen.manual_seed(args.seed + 31)
    try:
        env.reset()

        def run_steps():
            for _ in range(steps):
                actions = torch.randint(0, cfg.action_size, (num_envs, num_agents), generator=gen, device=device)
                env.step(actions)

        seconds, _ = _time(device, run_steps)
        return {
            "num_envs": num_envs,
            "num_agents": num_agents,
            "steps": steps,
            "env_steps_per_second": num_envs * steps / max(seconds, 1e-9),
            "agent_steps_per_second": num_envs * num_agents * steps / max(seconds, 1e-9),
            "seconds": seconds,
        }
    finally:
        env.close()


def benchmark_torch(device: torch.device, values: dict[str, Any], args) -> dict[str, Any]:
    batch = int(args.batch_size or values.get("WM_BATCH_SIZE", 1024))
    steps = int(args.torch_steps)
    obs_dim = 550
    z_dim = 64
    hidden = 512
    model = torch.nn.Sequential(
        torch.nn.Linear(obs_dim, hidden),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden, hidden),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden, z_dim),
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    x = torch.randn(batch, obs_dim, device=device)
    y = torch.randn(batch, z_dim, device=device)

    def train():
        for _ in range(steps):
            opt.zero_grad(set_to_none=True)
            loss = torch.nn.functional.mse_loss(model(x), y)
            loss.backward()
            opt.step()

    seconds, _ = _time(device, train)
    return {
        "batch_size": batch,
        "steps": steps,
        "vae_like_samples_per_second": batch * steps / max(seconds, 1e-9),
        "seconds": seconds,
    }


def build_warnings(report: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    gpu_util = report.get("gpu_util_after")
    cpu = report.get("cpu", {}).get("cpu_percent", 0.0)
    if isinstance(gpu_util, (int, float)) and report.get("device") == "cuda" and gpu_util < 60.0:
        warnings.append(
            "GPU utilization is below 60%; consider increasing WM_NUM_ENVS/WM_BATCH_SIZE/MARL_NUM_ENVS/MARL_FRAMES_PER_BATCH, disabling videos, or stopping other GPU jobs such as VLLM."
        )
    if isinstance(cpu, (int, float)) and cpu < 25.0:
        warnings.append(
            "CPU utilization is low; if runtime is slow, the bottleneck is likely Python synchronization or a single-core section rather than available CPU capacity."
        )
    env_sps = report.get("env", {}).get("env_steps_per_second", 0.0)
    if isinstance(env_sps, (int, float)) and env_sps <= 0:
        warnings.append("vGridcraft benchmark did not produce env throughput; check CUDA/device configuration.")
    return warnings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("smoke", "quick", "standard", "spark_max"), default="spark_max")
    parser.add_argument("--target", choices=("wm", "marl", "all"), default="all")
    parser.add_argument("--device", default=None)
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--num-agents", type=int, default=int(os.environ.get("NUM_AGENTS", "3")))
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--env-steps", type=int, default=50)
    parser.add_argument("--torch-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    info = detect_hardware()
    values = build_profile(info, args.profile, "all")
    device_name = args.device or values.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")

    report: dict[str, Any] = {
        "profile": args.profile,
        "target": args.target,
        "device": str(device),
        "hardware": {
            "cuda_available": info.cuda_available,
            "gpu_name": info.gpu_name,
            "gpu_count": info.gpu_count,
            "gpu_free_gib": info.gpu_free_gib,
            "gpu_total_gib": info.gpu_total_gib,
            "cpu_count": info.cpu_count,
            "ram_total_gib": info.ram_total_gib,
            "ram_available_gib": info.ram_available_gib,
        },
        "selected_profile": values,
        "gpu_util_before": _gpu_util(),
        "cpu": _cpu_snapshot(),
    }
    if args.target in {"wm", "marl", "all"}:
        report["env"] = benchmark_env(device, values, args)
    if args.target in {"wm", "marl", "all"}:
        report["torch_train"] = benchmark_torch(device, values, args)
    report["gpu_util_after"] = _gpu_util()
    report["warnings"] = build_warnings(report)

    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n")


if __name__ == "__main__":
    main()
