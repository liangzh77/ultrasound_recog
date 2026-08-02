from src.research_annotation_review import (
    FORBIDDEN_PUBLIC_FIELDS,
    PUBLIC_REVIEW_FIELDS,
    build_blinded_review_queue,
)


def _synthetic_rows():
    rows = []
    categories = {}
    for fold in range(5):
        for diagnosis_id in range(2):
            for index in range(4):
                image_key = f"i-{fold}-{diagnosis_id}-{index}"
                rows.append(
                    {
                        "image_key": image_key,
                        "person_key": f"p-{fold}-{diagnosis_id}-{index}",
                        "diagnosis": f"d{diagnosis_id}",
                        "outer_fold": fold,
                        "include": 1,
                    }
                )
                categories[image_key] = {"积液"} if index < 2 else set()
    return rows, categories


def test_review_queue_is_blinded_unique_and_fold_balanced():
    rows, categories = _synthetic_rows()
    public, audit = build_blinded_review_queue(
        rows,
        categories,
        targets=["积液"],
        seed=7,
        per_fold_per_bucket=2,
        required_independent_reviews=2,
    )

    assert len(public) == 20
    assert len({row["image_key"] for row in public}) == 20
    assert set(public[0]) == set(PUBLIC_REVIEW_FIELDS)
    assert not (set(public[0]) & FORBIDDEN_PUBLIC_FIELDS)
    assert all(row["required_independent_reviews"] == 2 for row in public)
    assert audit["contract"]["legacy_unlabeled_is_negative"] is False
    assert {row["count"] for row in audit["target_fold_counts"]} == {4}


def test_review_queue_rejects_insufficient_unique_patients():
    rows = [
        {
            "image_key": f"i{index}",
            "person_key": "same-patient",
            "diagnosis": "d0",
            "outer_fold": 0,
            "include": 1,
        }
        for index in range(3)
    ]
    categories = {row["image_key"]: {"积液"} for row in rows}

    try:
        build_blinded_review_queue(
            rows,
            categories,
            targets=["积液"],
            seed=7,
            per_fold_per_bucket=2,
            required_independent_reviews=2,
        )
    except ValueError as error:
        assert "Insufficient unique" in str(error)
    else:
        raise AssertionError("Repeated patients must not satisfy a fold quota")
