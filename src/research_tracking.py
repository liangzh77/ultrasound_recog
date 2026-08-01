"""Local MLflow contract that excludes patient identity and raw paths."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import mlflow
from mlflow import MlflowClient


FORBIDDEN_TRACKING_FIELDS = (
    "raw_path",
    "image_path",
    "annotation_path",
    "filename",
    "patient_folder",
    "patient_name",
    "person_name",
    "identity",
    "姓名",
    "编号",
)


def validate_tracking_metadata(metadata: Mapping[str, Any]) -> None:
    for key in metadata:
        lowered = str(key).casefold()
        if any(fragment.casefold() in lowered for fragment in FORBIDDEN_TRACKING_FIELDS):
            raise ValueError(f"Forbidden tracking field: {key}")


class LocalResearchTracker:
    def __init__(self, tracking_root: Path, experiment_name: str) -> None:
        self.tracking_root = tracking_root.resolve()
        self.tracking_root.mkdir(parents=True, exist_ok=True)
        database = (self.tracking_root / "mlflow.db").as_posix()
        mlflow.set_tracking_uri(f"sqlite:///{database}")
        self.client = MlflowClient()
        experiment = self.client.get_experiment_by_name(experiment_name)
        if experiment is None:
            artifact_root = self.tracking_root / "artifacts"
            artifact_root.mkdir(exist_ok=True)
            experiment_id = self.client.create_experiment(
                experiment_name,
                artifact_location=artifact_root.as_uri(),
            )
        else:
            experiment_id = experiment.experiment_id
        self.experiment_id = experiment_id

    @contextmanager
    def parent_run(
        self,
        run_name: str,
        metadata: Mapping[str, Any],
    ) -> Iterator[Any]:
        validate_tracking_metadata(metadata)
        with mlflow.start_run(
            experiment_id=self.experiment_id,
            run_name=run_name,
        ) as run:
            mlflow.log_params(dict(metadata))
            yield run

    @contextmanager
    def fold_run(
        self,
        run_name: str,
        metadata: Mapping[str, Any],
    ) -> Iterator[Any]:
        validate_tracking_metadata(metadata)
        if mlflow.active_run() is None:
            raise RuntimeError("fold_run requires an active parent run")
        with mlflow.start_run(run_name=run_name, nested=True) as run:
            mlflow.log_params(dict(metadata))
            yield run

    @staticmethod
    def log_metrics(metrics: Mapping[str, float], step: int | None = None) -> None:
        mlflow.log_metrics({key: float(value) for key, value in metrics.items()}, step=step)
