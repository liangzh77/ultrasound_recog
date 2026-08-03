"""Run the frozen G0 patient-level normal/abnormal image-gate experiment."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SHARED_ENTRYPOINT = ROOT / "tools/26_train_patient_image_mean.py"


def main(argv: list[str] | None = None) -> int:
    spec = importlib.util.spec_from_file_location(
        "research_patient_image_training_entrypoint",
        SHARED_ENTRYPOINT,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load the shared patient-image training entrypoint")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return int(module.main(sys.argv[1:] if argv is None else argv))


if __name__ == "__main__":
    raise SystemExit(main())
