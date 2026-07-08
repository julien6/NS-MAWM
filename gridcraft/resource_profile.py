from __future__ import annotations

import argparse
import json
import os
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class HardwareInfo:
    cuda_available: bool
    gpu_name: str
    gpu_count: int
    gpu_total_gib: float
    gpu_free_gib: float
    cpu_count: int
    ram_total_gib: float
    ram_available_gib: float


def _ram_info() -> tuple[float, float]:
    try:
        import psutil  # type: ignore

        memory = psutil.virtual_memory()
        return memory.total / (1024**3), memory.available / (1024**3)
    except Exception:
        meminfo = {}
        try:
            with Path("/proc/meminfo").open() as handle:
                for line in handle:
                    key, value = line.split(":", 1)
                    meminfo[key] = float(value.strip().split()[0]) * 1024.0
            total = meminfo.get("MemTotal", 0.0)
            available = meminfo.get("MemAvailable", meminfo.get("MemFree", 0.0))
            return total / (1024**3), available / (1024**3)
        except Exception:
            return 0.0, 0.0


def detect_hardware() -> HardwareInfo:
    cpu_count = os.cpu_count() or 1
    ram_total_gib, ram_available_gib = _ram_info()
    cuda_available = False
    gpu_name = "none"
    gpu_count = 0
    gpu_total_gib = 0.0
    gpu_free_gib = 0.0
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        if cuda_available:
            gpu_count = int(torch.cuda.device_count())
            index = int(os.environ.get("CUDA_VISIBLE_DEVICE_INDEX", "0"))
            if index >= gpu_count:
                index = 0
            props = torch.cuda.get_device_properties(index)
            gpu_name = str(props.name)
            try:
                free_bytes, total_bytes = torch.cuda.mem_get_info(index)
            except TypeError:
                free_bytes, total_bytes = torch.cuda.mem_get_info()
            gpu_total_gib = total_bytes / (1024**3)
            gpu_free_gib = free_bytes / (1024**3)
    except Exception:
        pass
    return HardwareInfo(
        cuda_available=cuda_available,
        gpu_name=gpu_name,
        gpu_count=gpu_count,
        gpu_total_gib=gpu_total_gib,
        gpu_free_gib=gpu_free_gib,
        cpu_count=cpu_count,
        ram_total_gib=ram_total_gib,
        ram_available_gib=ram_available_gib,
    )


def _worker_count(info: HardwareInfo, cap: int) -> int:
    if info.cpu_count <= 4:
        return 0
    return max(2, min(cap, info.cpu_count // 2))


def _base_profile(info: HardwareInfo, profile: str) -> dict[str, Any]:
    cuda = info.cuda_available
    free = info.gpu_free_gib or info.gpu_total_gib
    workers = _worker_count(info, 16)
    if profile == "smoke":
        return {
            "DEVICE": "cuda" if cuda else "cpu",
            "WM_NUM_ENVS": 8 if cuda else 4,
            "WM_BATCH_SIZE": 64,
            "WM_NUM_WORKERS": 0,
            "RESIDUAL_WM_BATCH_SIZE": 32,
            "MARL_NUM_ENVS": 8 if cuda else 4,
            "MARL_FRAMES_PER_BATCH": 128,
            "MARL_TRAIN_BATCH_SIZE": 64,
            "MARL_OPTIMIZER_STEPS": 1,
            "MARL_MEMORY_SIZE": 50_000,
        }
    if profile == "quick":
        return {
            "DEVICE": "cuda" if cuda else "cpu",
            "WM_NUM_ENVS": 256 if cuda else 32,
            "WM_BATCH_SIZE": 1024 if cuda else 256,
            "WM_NUM_WORKERS": _worker_count(info, 8),
            "RESIDUAL_WM_BATCH_SIZE": 512 if cuda else 128,
            "MARL_NUM_ENVS": 128 if cuda else 16,
            "MARL_FRAMES_PER_BATCH": 2048 if cuda else 512,
            "MARL_TRAIN_BATCH_SIZE": 512 if cuda else 128,
            "MARL_OPTIMIZER_STEPS": 2,
            "MARL_MEMORY_SIZE": 250_000,
        }
    if profile == "standard":
        return {
            "DEVICE": "cuda" if cuda else "cpu",
            "WM_NUM_ENVS": 512 if cuda else 64,
            "WM_BATCH_SIZE": 2048 if cuda else 512,
            "WM_NUM_WORKERS": _worker_count(info, 12),
            "RESIDUAL_WM_BATCH_SIZE": 1024 if cuda else 256,
            "MARL_NUM_ENVS": 256 if cuda else 32,
            "MARL_FRAMES_PER_BATCH": 8192 if cuda else 1024,
            "MARL_TRAIN_BATCH_SIZE": 1024 if cuda else 256,
            "MARL_OPTIMIZER_STEPS": 4,
            "MARL_MEMORY_SIZE": 1_000_000 if cuda else 200_000,
        }
    if profile != "spark_max":
        raise ValueError(f"Unsupported resource profile: {profile}")

    if not cuda:
        return {
            "DEVICE": "cpu",
            "WM_NUM_ENVS": 64,
            "WM_BATCH_SIZE": 512,
            "WM_NUM_WORKERS": _worker_count(info, 16),
            "RESIDUAL_WM_BATCH_SIZE": 256,
            "MARL_NUM_ENVS": 32,
            "MARL_FRAMES_PER_BATCH": 1024,
            "MARL_TRAIN_BATCH_SIZE": 256,
            "MARL_OPTIMIZER_STEPS": 2,
            "MARL_MEMORY_SIZE": 200_000,
        }
    if free >= 64:
        wm_envs, wm_batch, marl_envs, frames, train_batch, memory = 4096, 8192, 1536, 24576, 2048, 3_000_000
    elif free >= 40:
        wm_envs, wm_batch, marl_envs, frames, train_batch, memory = 2048, 8192, 1024, 16384, 2048, 2_000_000
    elif free >= 24:
        wm_envs, wm_batch, marl_envs, frames, train_batch, memory = 1024, 4096, 512, 8192, 1024, 1_000_000
    elif free >= 12:
        wm_envs, wm_batch, marl_envs, frames, train_batch, memory = 512, 2048, 256, 4096, 512, 500_000
    else:
        wm_envs, wm_batch, marl_envs, frames, train_batch, memory = 256, 1024, 128, 2048, 256, 250_000
    return {
        "DEVICE": "cuda",
        "WM_NUM_ENVS": wm_envs,
        "WM_BATCH_SIZE": wm_batch,
        "WM_NUM_WORKERS": workers,
        "RESIDUAL_WM_BATCH_SIZE": max(512, wm_batch // 2),
        "MARL_NUM_ENVS": marl_envs,
        "MARL_FRAMES_PER_BATCH": frames,
        "MARL_TRAIN_BATCH_SIZE": train_batch,
        "MARL_OPTIMIZER_STEPS": 4,
        "MARL_MEMORY_SIZE": memory,
    }


def build_profile(info: HardwareInfo, profile: str, target: str) -> dict[str, Any]:
    values = _base_profile(info, profile)
    values["RESOURCE_PROFILE"] = profile
    values["RESOURCE_PROFILE_TARGET"] = target
    values["RESOURCE_PROFILE_DEVICE"] = values["DEVICE"]
    values["RESOURCE_PROFILE_GPU_NAME"] = info.gpu_name
    values["RESOURCE_PROFILE_GPU_FREE_GIB"] = round(info.gpu_free_gib, 2)
    values["RESOURCE_PROFILE_GPU_TOTAL_GIB"] = round(info.gpu_total_gib, 2)
    values["RESOURCE_PROFILE_CPU_COUNT"] = info.cpu_count
    values["RESOURCE_PROFILE_RAM_TOTAL_GIB"] = round(info.ram_total_gib, 2)

    if target in {"campaign", "all"}:
        values.setdefault("MB_WORLD_MODEL_BATCH_SIZE", min(1024, max(256, int(values["WM_BATCH_SIZE"]) // 4)))
    if target in {"wm_hpo", "all"}:
        values.update(
            {
                "HPO_DEVICE": values["DEVICE"],
                "HPO_NUM_ENVS": values["WM_NUM_ENVS"],
                "HPO_WM_NUM_WORKERS": values["WM_NUM_WORKERS"],
            }
        )
    if target in {"marl_hpo", "all"}:
        values.update(
            {
                "MARL_HPO_DEVICE": values["DEVICE"],
                "MARL_HPO_NUM_ENVS": values["MARL_NUM_ENVS"],
                "MARL_HPO_FRAMES_PER_BATCH": values["MARL_FRAMES_PER_BATCH"],
                "MARL_HPO_TRAIN_BATCH_SIZE": values["MARL_TRAIN_BATCH_SIZE"],
                "MARL_HPO_OPTIMIZER_STEPS": values["MARL_OPTIMIZER_STEPS"],
                "MARL_HPO_MEMORY_SIZE": values["MARL_MEMORY_SIZE"],
            }
        )
    return values


def _shell_value(value: Any) -> str:
    return shlex.quote(str(value))


def emit_shell(values: dict[str, Any]) -> str:
    lines = []
    for key in sorted(values):
        lines.append(f"export {key}={_shell_value(values[key])}")
    return "\n".join(lines)


def emit_summary(info: HardwareInfo, values: dict[str, Any]) -> str:
    interesting = [
        "DEVICE",
        "WM_NUM_ENVS",
        "WM_BATCH_SIZE",
        "WM_NUM_WORKERS",
        "RESIDUAL_WM_BATCH_SIZE",
        "MARL_NUM_ENVS",
        "MARL_FRAMES_PER_BATCH",
        "MARL_TRAIN_BATCH_SIZE",
        "MARL_OPTIMIZER_STEPS",
        "MARL_MEMORY_SIZE",
        "HPO_NUM_ENVS",
        "MARL_HPO_NUM_ENVS",
    ]
    lines = [
        "[resource-profile] hardware: "
        f"cuda={info.cuda_available}, gpu={info.gpu_name}, "
        f"vram_free={info.gpu_free_gib:.1f}GiB/{info.gpu_total_gib:.1f}GiB, "
        f"cpus={info.cpu_count}, ram={info.ram_available_gib:.1f}GiB/{info.ram_total_gib:.1f}GiB",
        f"[resource-profile] profile={values.get('RESOURCE_PROFILE')} target={values.get('RESOURCE_PROFILE_TARGET')}",
    ]
    chosen = ", ".join(f"{key}={values[key]}" for key in interesting if key in values)
    lines.append(f"[resource-profile] selected: {chosen}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Recommend Gridcraft resource settings for the current machine.")
    parser.add_argument("--profile", choices=("smoke", "quick", "standard", "spark_max"), default="spark_max")
    parser.add_argument("--target", choices=("campaign", "wm_hpo", "marl_hpo", "all"), default="all")
    parser.add_argument("--format", choices=("shell", "json", "summary"), default="summary")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="When emitting shell exports, overwrite variables already present in the environment.",
    )
    args = parser.parse_args()

    info = detect_hardware()
    values = build_profile(info, args.profile, args.target)
    if args.format == "shell":
        shell_values = values if args.overwrite else {key: value for key, value in values.items() if key not in os.environ}
        print(emit_shell(shell_values))
    elif args.format == "json":
        print(json.dumps({"hardware": info.__dict__, "settings": values}, indent=2, sort_keys=True))
    else:
        print(emit_summary(info, values))


if __name__ == "__main__":
    main()
