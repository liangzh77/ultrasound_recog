"""Conservative resource guards for local research training runs."""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class ResourcePolicy:
    start_disk_free_gb: float = 20.0
    start_memory_available_gb: float = 8.0
    stop_memory_available_gb: float = 4.0
    stop_cpu_average_percent: float = 80.0
    warn_gpu_temperature_c: float = 83.0
    stop_gpu_temperature_c: float = 88.0
    soft_time_limit_hours: float = 11.5
    hard_time_limit_hours: float = 23.5
    gpu_memory_budget_gb: float = 9.0
    dataloader_workers: int = 2
    torch_intraop_threads: int = 4
    torch_interop_threads: int = 1


@dataclass(frozen=True)
class ResourceSnapshot:
    disk_free_gb: float
    memory_available_gb: float
    cpu_average_percent: float
    gpu_memory_used_gb: float
    gpu_temperature_c: float
    elapsed_hours: float = 0.0


@dataclass(frozen=True)
class GuardDecision:
    allowed: bool
    status: Literal[
        "CONTINUE",
        "START_REJECTED",
        "TIME_BUDGET_REACHED",
        "HARD_TIME_LIMIT_REACHED",
        "RESOURCE_GUARD_STOPPED",
    ]
    reasons: tuple[str, ...] = ()


@dataclass
class RuntimeGuard:
    """Track sustained CPU/GPU pressure across epoch-boundary snapshots."""

    policy: ResourcePolicy
    sustained_pressure_seconds: float = 300.0
    cpu_high_since_seconds: float | None = None
    gpu_hot_since_seconds: float | None = None

    def __post_init__(self) -> None:
        if self.sustained_pressure_seconds <= 0:
            raise ValueError("sustained_pressure_seconds must be positive")

    def evaluate(self, current: ResourceSnapshot) -> GuardDecision:
        immediate = evaluate_runtime(current, self.policy)
        if not immediate.allowed:
            return immediate

        observed_seconds = current.elapsed_hours * 3600.0
        reasons = []
        if current.cpu_average_percent > self.policy.stop_cpu_average_percent:
            if self.cpu_high_since_seconds is None:
                self.cpu_high_since_seconds = observed_seconds
            elif (
                observed_seconds - self.cpu_high_since_seconds
                >= self.sustained_pressure_seconds
            ):
                reasons.append("cpu_average_above_80pct_for_5min")
        else:
            self.cpu_high_since_seconds = None

        if current.gpu_temperature_c > self.policy.stop_gpu_temperature_c:
            if self.gpu_hot_since_seconds is None:
                self.gpu_hot_since_seconds = observed_seconds
            elif (
                observed_seconds - self.gpu_hot_since_seconds
                >= self.sustained_pressure_seconds
            ):
                reasons.append("gpu_temperature_above_88c_for_5min")
        else:
            self.gpu_hot_since_seconds = None

        return GuardDecision(
            allowed=not reasons,
            status="CONTINUE" if not reasons else "RESOURCE_GUARD_STOPPED",
            reasons=tuple(reasons),
        )


def evaluate_training_start(
    current: ResourceSnapshot,
    policy: ResourcePolicy,
) -> GuardDecision:
    reasons = []
    if current.disk_free_gb < policy.start_disk_free_gb:
        reasons.append("disk_free_below_20gb")
    if current.memory_available_gb < policy.start_memory_available_gb:
        reasons.append("memory_available_below_8gb")
    if current.gpu_memory_used_gb > policy.gpu_memory_budget_gb:
        reasons.append("gpu_memory_already_above_9gb")
    return GuardDecision(
        allowed=not reasons,
        status="CONTINUE" if not reasons else "START_REJECTED",
        reasons=tuple(reasons),
    )


def evaluate_runtime(
    current: ResourceSnapshot,
    policy: ResourcePolicy,
) -> GuardDecision:
    if current.elapsed_hours >= policy.hard_time_limit_hours:
        return GuardDecision(
            allowed=False,
            status="HARD_TIME_LIMIT_REACHED",
            reasons=("elapsed_above_23_5h",),
        )
    if current.elapsed_hours >= policy.soft_time_limit_hours:
        return GuardDecision(
            allowed=False,
            status="TIME_BUDGET_REACHED",
            reasons=("elapsed_above_11_5h",),
        )

    reasons = []
    if current.memory_available_gb < policy.stop_memory_available_gb:
        reasons.append("memory_available_below_4gb")
    return GuardDecision(
        allowed=not reasons,
        status="CONTINUE" if not reasons else "RESOURCE_GUARD_STOPPED",
        reasons=tuple(reasons),
    )


def configure_conservative_threads(policy: ResourcePolicy | None = None) -> None:
    policy = policy or ResourcePolicy()
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = "2"
    os.environ["TORCH_NUM_THREADS"] = str(policy.torch_intraop_threads)
    os.environ["TORCH_NUM_INTEROP_THREADS"] = str(policy.torch_interop_threads)


def set_below_normal_priority() -> bool:
    if os.name != "nt":
        return False
    import psutil

    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    return True


def parse_nvidia_smi_row(row: str) -> dict[str, float | str]:
    fields = [part.strip() for part in row.strip().split(",")]
    if len(fields) != 5:
        raise ValueError(f"Unexpected nvidia-smi output: {row!r}")
    name, total_mib, used_mib, temperature_c, utilization = fields
    return {
        "gpu_name": name,
        "gpu_memory_total_gb": round(float(total_mib) / 1024, 2),
        "gpu_memory_used_gb": round(float(used_mib) / 1024, 2),
        "gpu_temperature_c": float(temperature_c),
        "gpu_utilization_percent": float(utilization),
    }


def collect_resource_snapshot(
    workspace_root: Path,
) -> tuple[ResourceSnapshot, dict[str, float | str]]:
    import psutil

    command = [
        "nvidia-smi",
        "--query-gpu=name,memory.total,memory.used,temperature.gpu,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    output = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout.splitlines()[0]
    gpu = parse_nvidia_smi_row(output)
    memory = psutil.virtual_memory()
    disk = shutil.disk_usage(workspace_root)
    current = ResourceSnapshot(
        disk_free_gb=round(disk.free / (1024**3), 2),
        memory_available_gb=round(memory.available / (1024**3), 2),
        cpu_average_percent=float(psutil.cpu_percent(interval=1.0)),
        gpu_memory_used_gb=float(gpu["gpu_memory_used_gb"]),
        gpu_temperature_c=float(gpu["gpu_temperature_c"]),
    )
    return current, gpu


def policy_as_dict(policy: ResourcePolicy) -> dict[str, object]:
    return asdict(policy)
