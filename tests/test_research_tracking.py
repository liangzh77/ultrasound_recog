import pytest

from src.research_tracking import LocalResearchTracker, validate_tracking_metadata


def test_tracking_metadata_rejects_identity_and_path_fields():
    validate_tracking_metadata(
        {
            "experiment_code": "E1",
            "backbone": "efficientnet_b2.ra_in1k",
            "outer_fold": 0,
        }
    )

    with pytest.raises(ValueError, match="raw_path"):
        validate_tracking_metadata({"raw_path": "data/patient.jpg"})
    with pytest.raises(ValueError, match="patient_name"):
        validate_tracking_metadata({"patient_name": "someone"})


def test_local_tracker_writes_nested_parent_and_fold_runs(tmp_path):
    tracker = LocalResearchTracker(tmp_path, experiment_name="tracking-test")

    with tracker.parent_run(
        "E1",
        {"experiment_code": "E1", "prediction_level": "patient"},
    ) as parent:
        with tracker.fold_run(
            "E1-fold0",
            {"outer_fold": 0, "input_mode": "roi"},
        ) as child:
            tracker.log_metrics({"macro_f1": 0.5}, step=1)

    assert parent.info.run_id
    assert child.info.run_id
    child_data = tracker.client.get_run(child.info.run_id)
    assert child_data.data.params["outer_fold"] == "0"
    assert child_data.data.metrics["macro_f1"] == 0.5
    assert child_data.data.tags["mlflow.parentRunId"] == parent.info.run_id
