"""Entry point for the E2 gated-attention patient MIL experiment."""

from pathlib import Path
import runpy


if __name__ == "__main__":
    runpy.run_path(
        str(Path(__file__).with_name("26_train_patient_image_mean.py")),
        run_name="__main__",
    )
