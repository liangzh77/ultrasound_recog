"""Independent parent-process hard timeout for research training commands."""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import psutil


@dataclass(frozen=True)
class WatchdogResult:
    returncode: int
    timed_out: bool


def _stop_process_tree(process: subprocess.Popen) -> None:
    try:
        parent = psutil.Process(process.pid)
        processes = parent.children(recursive=True) + [parent]
    except psutil.Error:
        processes = []
    for item in processes:
        try:
            item.terminate()
        except psutil.Error:
            pass
    _, alive = psutil.wait_procs(processes, timeout=10)
    for item in alive:
        try:
            item.kill()
        except psutil.Error:
            pass
    if process.poll() is None:
        process.kill()
    process.wait()


def run_with_hard_timeout(
    command: Sequence[str],
    cwd: Path,
    timeout_seconds: float,
) -> WatchdogResult:
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    process = subprocess.Popen(
        list(command),
        cwd=cwd,
        creationflags=creationflags,
    )
    try:
        return WatchdogResult(
            returncode=process.wait(timeout=timeout_seconds),
            timed_out=False,
        )
    except subprocess.TimeoutExpired:
        _stop_process_tree(process)
        return WatchdogResult(returncode=124, timed_out=True)
    except KeyboardInterrupt:
        _stop_process_tree(process)
        raise
