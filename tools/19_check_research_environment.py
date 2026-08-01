"""Create the research experiment skeleton and verify local resources."""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import platform
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.common_paths import (  # noqa: E402
    PATIENT_MULTIMODAL_ARTIFACTS_DIR,
    PATIENT_MULTIMODAL_CONFIGS_DIR,
    PATIENT_MULTIMODAL_EXPERIMENT_DIR,
    PATIENT_MULTIMODAL_LOGS_DIR,
    PATIENT_MULTIMODAL_REPORTS_DIR,
)
from src.research_runtime import (  # noqa: E402
    ResourcePolicy,
    collect_resource_snapshot,
    configure_conservative_threads,
    evaluate_training_start,
    policy_as_dict,
    set_below_normal_priority,
)


PACKAGES = (
    "torch",
    "torchvision",
    "timm",
    "scikit-learn",
    "mlflow-skinny",
    "SQLAlchemy",
    "alembic",
    "psutil",
    "openpyxl",
)


def package_versions() -> dict[str, str | None]:
    result = {}
    for package in PACKAGES:
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = None
    return result


def git_state() -> dict[str, str]:
    def run(*args: str) -> str:
        completed = subprocess.run(
            ["git", *args],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        return completed.stdout.strip()

    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "status": run("status", "--short"),
    }


def pip_check() -> dict[str, object]:
    completed = subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return {
        "passed": completed.returncode == 0,
        "output": (completed.stdout or completed.stderr).strip(),
    }


def gpu_amp_smoke_test() -> dict[str, object]:
    configure_conservative_threads()
    import torch

    if not torch.cuda.is_available():
        return {"passed": False, "reason": "cuda_unavailable"}
    device = torch.device("cuda")
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        a = torch.randn((64, 64), device=device)
        result = a @ a
    torch.cuda.synchronize()
    return {
        "passed": bool(torch.isfinite(result).all().item()),
        "device": torch.cuda.get_device_name(0),
        "dtype": str(result.dtype),
    }


def mlflow_smoke_test() -> dict[str, object]:
    import mlflow

    tracking = PATIENT_MULTIMODAL_EXPERIMENT_DIR / "tracking"
    tracking.mkdir(parents=True, exist_ok=True)
    database = (tracking / "mlflow.db").resolve()
    mlflow.set_tracking_uri(f"sqlite:///{database.as_posix()}")
    mlflow.set_experiment("environment-smoke")
    with mlflow.start_run(run_name="parent-smoke") as parent:
        mlflow.log_param("purpose", "p0_smoke")
        with mlflow.start_run(run_name="child-smoke", nested=True) as child:
            mlflow.log_metric("passed", 1.0)
    return {
        "passed": True,
        "version": mlflow.__version__,
        "parent_run_created": bool(parent.info.run_id),
        "child_run_created": bool(child.info.run_id),
        "sqlite_database_created": database.exists(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-gpu", action="store_true")
    parser.add_argument("--smoke-mlflow", action="store_true")
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Always return success after writing the report.",
    )
    args = parser.parse_args()

    for directory in (
        PATIENT_MULTIMODAL_CONFIGS_DIR,
        PATIENT_MULTIMODAL_ARTIFACTS_DIR,
        PATIENT_MULTIMODAL_LOGS_DIR,
        PATIENT_MULTIMODAL_REPORTS_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    policy = ResourcePolicy()
    current, gpu = collect_resource_snapshot(ROOT)
    decision = evaluate_training_start(current, policy)
    report = {
        "created_at": datetime.now().astimezone().isoformat(),
        "experiment_id": PATIENT_MULTIMODAL_EXPERIMENT_DIR.name,
        "platform": platform.platform(),
        "python": sys.version,
        "packages": package_versions(),
        "mlflow_available": importlib.util.find_spec("mlflow") is not None,
        "resource_policy": policy_as_dict(policy),
        "resource_snapshot": asdict(current),
        "gpu": gpu,
        "training_start_decision": asdict(decision),
        "git": git_state(),
        "pip_check": pip_check(),
        "below_normal_priority_set": set_below_normal_priority(),
    }
    if args.smoke_gpu:
        report["gpu_amp_smoke_test"] = gpu_amp_smoke_test()
    if args.smoke_mlflow:
        report["mlflow_smoke_test"] = mlflow_smoke_test()

    destination = PATIENT_MULTIMODAL_REPORTS_DIR / "environment.json"
    destination.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(destination)
    print(json.dumps(report["training_start_decision"], ensure_ascii=False))
    return 0 if args.report_only or decision.allowed else 2


if __name__ == "__main__":
    raise SystemExit(main())
