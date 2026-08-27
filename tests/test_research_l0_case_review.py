from src.research_l0_case_review import build_case_review_queue


def _row(person: str, reference: str, reference_id: int, top1: str, p0: float):
    return {
        "person_key": person,
        "outer_fold": "0",
        "reference_class": reference,
        "reference_id": str(reference_id),
        "image_count": "2",
        "top1": top1,
        "prob_normal": str(p0),
        "prob_ra": str(1 - p0),
        "prob_ga": "0",
        "prob_spa": "0",
        "prob_oa": "0",
        "prob_injury": "0",
    }


def test_case_review_queue_covers_changed_drift_and_mask_controls():
    l0 = [
        _row("P1", "正常", 0, "正常", 0.95),
        _row("P2", "正常", 0, "类风湿性关节炎", 0.2),
        _row("P3", "正常", 0, "正常", 0.8),
        _row("P4", "正常", 0, "正常", 0.7),
    ]
    e1 = [
        _row("P1", "正常", 0, "类风湿性关节炎", 0.1),
        _row("P2", "正常", 0, "正常", 0.75),
        _row("P3", "正常", 0, "正常", 0.6),
        _row("P4", "正常", 0, "正常", 0.6),
    ]
    queue = build_case_review_queue(
        l0,
        e1,
        {"P1": 1, "P2": 1, "P3": 1, "P4": 0},
        per_class_changed=1,
        per_class_drift=1,
        per_class_stable_mask_group=1,
    )
    by_person = {row["person_key"]: row for row in queue}
    assert set(by_person) == {"P1", "P2", "P3", "P4"}
    assert "l0_corrected_e1_error" in by_person["P1"]["review_reasons"]
    assert "largest_probability_drift" in by_person["P1"]["review_reasons"]
    assert "l0_added_error" in by_person["P2"]["review_reasons"]
    assert by_person["P3"]["mask_group"] == "any_positive"
    assert by_person["P4"]["mask_group"] == "all_zero"


def test_case_review_rejects_incomplete_mask_coverage():
    row = _row("P1", "正常", 0, "正常", 0.9)
    try:
        build_case_review_queue([row], [row], {})
    except ValueError as error:
        assert "Mask presence" in str(error)
    else:
        raise AssertionError("expected incomplete mask coverage to fail")
