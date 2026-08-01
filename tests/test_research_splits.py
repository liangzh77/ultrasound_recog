from src.research_splits import build_inner_rows, build_outer_folds


def patients():
    return [
        {"person_key": f"PERSON_{diagnosis}_{index:02d}", "diagnosis": diagnosis}
        for diagnosis in ("正常", "损伤", "类风湿性关节炎")
        for index in range(15)
    ]


def test_outer_folds_are_patient_unique_stratified_and_deterministic():
    first = build_outer_folds(patients(), n_splits=5, seed=20260724)
    second = build_outer_folds(patients(), n_splits=5, seed=20260724)

    assert first == second
    assert len(first) == 45
    assert set(first.values()) == {0, 1, 2, 3, 4}
    for diagnosis in ("正常", "损伤", "类风湿性关节炎"):
        keys = [
            row["person_key"] for row in patients() if row["diagnosis"] == diagnosis
        ]
        assert {first[key] for key in keys} == {0, 1, 2, 3, 4}


def test_inner_rows_never_put_outer_test_patient_in_train_or_validation():
    outer = build_outer_folds(patients(), n_splits=5, seed=20260724)
    rows = build_inner_rows(patients(), outer, seed=20260724)

    assert len(rows) == 45 * 5
    for outer_fold in range(5):
        fold_rows = [row for row in rows if row["outer_fold"] == outer_fold]
        assert len(fold_rows) == 45
        for row in fold_rows:
            expected = "test" if outer[row["person_key"]] == outer_fold else None
            if expected:
                assert row["split"] == expected
            else:
                assert row["split"] in {"train", "validation"}
