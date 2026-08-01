"""Safely merge newly delivered raw annotations into the working raw dataset.

The source and target use different patient-folder names.  Matching is based on
the complete set of image identities (file name, size, and SHA-256), never on
patient names.  Source annotations are authoritative, while target-only
``ultrasound_*`` crop metadata is preserved.

The command is a dry-run unless ``--apply`` is supplied.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.common_paths import RAW_LABEL_DIR, REGISTRY_DIR, ROOT


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
EXCLUDED_DISEASES = {"膝关节2026未标注"}
CROP_KEYS = {
    "ultrasound_rect",
    "ultrasound_candidates",
    "ultrasound_rect_reviewed",
}


@dataclass(frozen=True)
class PatientAction:
    disease: str
    source_dir: Path | None
    target_dir: Path
    matched_images: int = 0


@dataclass(frozen=True)
class FileCopyAction:
    disease: str
    source_file: Path
    target_file: Path


@dataclass
class SyncPlan:
    source_base: Path
    target_base: Path
    matched_patient_actions: list[PatientAction] = field(default_factory=list)
    rename_actions: list[PatientAction] = field(default_factory=list)
    new_patient_actions: list[PatientAction] = field(default_factory=list)
    unmatched_target_actions: list[PatientAction] = field(default_factory=list)
    excel_actions: list[FileCopyAction] = field(default_factory=list)
    metadata_actions: list[FileCopyAction] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)

    @property
    def patient_actions(self) -> list[PatientAction]:
        return self.matched_patient_actions + self.new_patient_actions


@dataclass
class SyncReport:
    folders_renamed: int = 0
    new_patients_copied: int = 0
    files_copied: int = 0
    files_replaced: int = 0
    json_merged: int = 0
    files_unchanged: int = 0
    excel_copied: int = 0
    metadata_copied: int = 0
    unmatched_targets_preserved: int = 0
    conflicts: int = 0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _image_files(patient_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in patient_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    )


def _image_identities(patient_dir: Path) -> frozenset[tuple[str, int, str]]:
    return frozenset(
        (path.name.casefold(), path.stat().st_size, _sha256(path))
        for path in _image_files(patient_dir)
    )


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def merge_annotation_data(
    source_data: dict[str, Any],
    target_data: dict[str, Any] | None,
) -> dict[str, Any]:
    """Use the source annotation while preserving target crop metadata."""
    merged = json.loads(json.dumps(source_data, ensure_ascii=False))
    if target_data:
        for key in CROP_KEYS:
            if key in target_data:
                merged[key] = target_data[key]
    return merged


def _best_source_match(
    target_identity: frozenset[tuple[str, int, str]],
    source_identities: dict[Path, frozenset[tuple[str, int, str]]],
) -> tuple[Path | None, int, bool]:
    scored: list[tuple[int, float, Path]] = []
    for source_dir, source_identity in source_identities.items():
        intersection = len(target_identity & source_identity)
        if intersection == 0:
            continue
        union = len(target_identity | source_identity)
        scored.append((intersection, intersection / union, source_dir))
    if not scored:
        return None, 0, False
    scored.sort(key=lambda item: (item[0], item[1], item[2].name), reverse=True)
    top = scored[0]
    tied = [
        item
        for item in scored
        if item[0] == top[0] and abs(item[1] - top[1]) < 1e-12
    ]
    return top[2], top[0], len(tied) > 1


def build_sync_plan(source_base: Path, target_base: Path) -> SyncPlan:
    source_base = source_base.resolve()
    target_base = target_base.resolve()
    if not source_base.is_dir():
        raise FileNotFoundError(f"Source dataset does not exist: {source_base}")
    if not target_base.is_dir():
        raise FileNotFoundError(f"Target dataset does not exist: {target_base}")
    if source_base == target_base:
        raise ValueError("Source and target datasets must be different directories")

    plan = SyncPlan(source_base=source_base, target_base=target_base)
    source_diseases = sorted(
        path
        for path in source_base.iterdir()
        if path.is_dir() and path.name not in EXCLUDED_DISEASES
    )
    source_dirs = [
        patient_dir
        for disease_dir in source_diseases
        for patient_dir in sorted(path for path in disease_dir.iterdir() if path.is_dir())
    ]
    target_dirs = [
        patient_dir
        for disease_dir in sorted(
            path
            for path in target_base.iterdir()
            if path.is_dir() and path.name not in EXCLUDED_DISEASES
        )
        for patient_dir in sorted(path for path in disease_dir.iterdir() if path.is_dir())
    ]
    source_identities = {
        path: _image_identities(path)
        for path in source_dirs
    }
    target_identities = {
        path: _image_identities(path)
        for path in target_dirs
    }
    empty_sources = [path for path, identity in source_identities.items() if not identity]
    for path in empty_sources:
        plan.conflicts.append(
            f"{path.parent.name}: source patient has no images: {path}"
        )

    chosen_sources: dict[Path, Path] = {}
    for target_dir, target_identity in target_identities.items():
        target_disease = target_dir.parent.name
        if not target_identity:
            plan.unmatched_target_actions.append(
                PatientAction(target_disease, None, target_dir)
            )
            continue
        source_dir, overlap, tied = _best_source_match(
            target_identity,
            source_identities,
        )
        if tied:
            plan.conflicts.append(
                f"{target_disease}: ambiguous source match for target "
                f"{target_dir.name}"
            )
            continue
        if source_dir is None:
            plan.unmatched_target_actions.append(
                PatientAction(target_disease, None, target_dir)
            )
            continue
        source_disease = source_dir.parent.name
        if source_dir in chosen_sources:
            plan.conflicts.append(
                f"{source_disease}: source {source_dir.name} matches both "
                f"{chosen_sources[source_dir]} and {target_dir}"
            )
            continue
        chosen_sources[source_dir] = target_dir
        action = PatientAction(source_disease, source_dir, target_dir, overlap)
        plan.matched_patient_actions.append(action)
        destination = target_base / source_disease / source_dir.name
        if target_dir.resolve() != destination.resolve():
            plan.rename_actions.append(action)

    assigned_sources = set(chosen_sources)
    for source_dir in source_dirs:
        if source_dir in assigned_sources or source_dir in empty_sources:
            continue
        disease = source_dir.parent.name
        destination = target_base / disease / source_dir.name
        plan.new_patient_actions.append(
            PatientAction(disease, source_dir, destination)
        )

    moving_targets = {action.target_dir.resolve() for action in plan.rename_actions}
    for action in plan.rename_actions:
        destination = (target_base / action.disease / action.source_dir.name).resolve()
        if destination.exists() and destination not in moving_targets:
            plan.conflicts.append(
                f"{action.disease}: rename destination already exists: {destination}"
            )
    for action in plan.new_patient_actions:
        if action.target_dir.exists():
            plan.conflicts.append(
                f"{action.disease}: new patient destination already exists: "
                f"{action.target_dir}"
            )

    for source_disease in source_diseases:
        disease = source_disease.name
        target_disease = target_base / disease
        for source_file in sorted(source_disease.iterdir()):
            if source_file.is_file() and source_file.suffix.lower() in {".xlsx", ".xls"}:
                plan.excel_actions.append(
                    FileCopyAction(
                        disease,
                        source_file,
                        target_disease / source_file.name,
                    )
                )
    for source_file in sorted(path for path in source_base.iterdir() if path.is_file()):
        plan.metadata_actions.append(
            FileCopyAction("", source_file, target_base.parent / source_file.name)
        )

    return plan


def _backup_file(path: Path, target_base: Path, backup_root: Path) -> None:
    if not path.exists():
        return
    relative = path.resolve().relative_to(target_base.resolve())
    backup_path = backup_root / relative
    if backup_path.exists():
        return
    backup_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, backup_path)


def _write_json(path: Path, data: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.sync-{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(data, ensure_ascii=False, indent=4),
        encoding="utf-8",
    )
    temporary.replace(path)


def _sync_patient_files(
    source_dir: Path,
    target_dir: Path,
    *,
    target_base: Path,
    backup_root: Path,
    report: SyncReport,
) -> None:
    for source_file in sorted(path for path in source_dir.rglob("*") if path.is_file()):
        relative = source_file.relative_to(source_dir)
        target_file = target_dir / relative
        if not _is_within(target_file, target_base):
            raise ValueError(f"Refusing to write outside target dataset: {target_file}")
        target_file.parent.mkdir(parents=True, exist_ok=True)

        if source_file.suffix.lower() == ".json" and target_file.exists():
            source_data = json.loads(source_file.read_text(encoding="utf-8"))
            target_data = json.loads(target_file.read_text(encoding="utf-8"))
            merged = merge_annotation_data(source_data, target_data)
            if merged == target_data:
                report.files_unchanged += 1
                continue
            _backup_file(target_file, target_base, backup_root)
            _write_json(target_file, merged)
            report.json_merged += 1
            continue

        if target_file.exists():
            if (
                source_file.stat().st_size == target_file.stat().st_size
                and _sha256(source_file) == _sha256(target_file)
            ):
                report.files_unchanged += 1
                continue
            if (
                source_file.parent == source_dir
                and source_file.suffix.lower() in IMAGE_EXTS
            ):
                raise ValueError(
                    f"Image conflict for matched patient: {source_file} -> {target_file}"
                )
            _backup_file(target_file, target_base, backup_root)
            shutil.copy2(source_file, target_file)
            report.files_replaced += 1
        else:
            shutil.copy2(source_file, target_file)
            report.files_copied += 1


def execute_sync_plan(plan: SyncPlan, *, backup_root: Path) -> SyncReport:
    """Execute a conflict-free plan and return operation counts."""
    if plan.conflicts:
        raise ValueError(
            "Sync plan has conflicts and cannot be applied:\n- "
            + "\n- ".join(plan.conflicts)
        )
    backup_root = backup_root.resolve()
    target_base = plan.target_base.resolve()
    if _is_within(backup_root, target_base):
        raise ValueError("Backup directory must be outside the target raw dataset")
    backup_root.mkdir(parents=True, exist_ok=True)

    report = SyncReport(
        unmatched_targets_preserved=len(plan.unmatched_target_actions),
        conflicts=len(plan.conflicts),
    )
    temporary_moves: list[tuple[PatientAction, Path]] = []
    for action in plan.rename_actions:
        if not _is_within(action.target_dir, target_base):
            raise ValueError(f"Unsafe rename source: {action.target_dir}")
        temporary = action.target_dir.with_name(
            f".sync-{uuid.uuid4().hex}-{action.target_dir.name}"
        )
        action.target_dir.rename(temporary)
        temporary_moves.append((action, temporary))

    for action, temporary in temporary_moves:
        destination = target_base / action.disease / action.source_dir.name
        if destination.exists():
            raise FileExistsError(f"Rename destination appeared during sync: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary.rename(destination)
        report.folders_renamed += 1

    for action in plan.matched_patient_actions:
        target_dir = target_base / action.disease / action.source_dir.name
        _sync_patient_files(
            action.source_dir,
            target_dir,
            target_base=target_base,
            backup_root=backup_root,
            report=report,
        )

    for action in plan.new_patient_actions:
        if not _is_within(action.target_dir, target_base):
            raise ValueError(f"Unsafe new patient destination: {action.target_dir}")
        action.target_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(action.source_dir, action.target_dir, copy_function=shutil.copy2)
        report.new_patients_copied += 1
        report.files_copied += sum(
            1 for path in action.source_dir.rglob("*") if path.is_file()
        )

    for action in plan.excel_actions:
        action.target_file.parent.mkdir(parents=True, exist_ok=True)
        if action.target_file.exists():
            if (
                action.source_file.stat().st_size == action.target_file.stat().st_size
                and _sha256(action.source_file) == _sha256(action.target_file)
            ):
                report.files_unchanged += 1
                continue
            _backup_file(action.target_file, target_base, backup_root)
        shutil.copy2(action.source_file, action.target_file)
        report.excel_copied += 1
        report.files_copied += 1

    for action in plan.metadata_actions:
        raw_root = target_base.parent
        if not _is_within(action.target_file, raw_root):
            raise ValueError(
                f"Refusing to write metadata outside target raw root: "
                f"{action.target_file}"
            )
        if action.target_file.exists():
            if (
                action.source_file.stat().st_size == action.target_file.stat().st_size
                and _sha256(action.source_file) == _sha256(action.target_file)
            ):
                report.files_unchanged += 1
                continue
            _backup_file(action.target_file, raw_root, backup_root)
        shutil.copy2(action.source_file, action.target_file)
        report.metadata_copied += 1
        report.files_copied += 1

    return report


def _plan_as_dict(plan: SyncPlan) -> dict[str, Any]:
    def patient(action: PatientAction) -> dict[str, Any]:
        return {
            "disease": action.disease,
            "source_dir": str(action.source_dir) if action.source_dir else None,
            "target_dir": str(action.target_dir),
            "matched_images": action.matched_images,
        }

    return {
        "source_base": str(plan.source_base),
        "target_base": str(plan.target_base),
        "matched_patients": [patient(item) for item in plan.matched_patient_actions],
        "renames": [patient(item) for item in plan.rename_actions],
        "new_patients": [patient(item) for item in plan.new_patient_actions],
        "unmatched_targets": [
            patient(item) for item in plan.unmatched_target_actions
        ],
        "excel_files": [
            {
                "disease": item.disease,
                "source_file": str(item.source_file),
                "target_file": str(item.target_file),
            }
            for item in plan.excel_actions
        ],
        "metadata_files": [
            {
                "source_file": str(item.source_file),
                "target_file": str(item.target_file),
            }
            for item in plan.metadata_actions
        ],
        "conflicts": plan.conflicts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=ROOT / "data" / "膝关节已标注",
    )
    parser.add_argument("--target", type=Path, default=RAW_LABEL_DIR)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument(
        "--report",
        type=Path,
        default=REGISTRY_DIR / "raw_data_sync_plan.json",
    )
    parser.add_argument(
        "--backup-root",
        type=Path,
        default=REGISTRY_DIR / "raw_data_sync_backup",
    )
    args = parser.parse_args()

    plan = build_sync_plan(args.source, args.target)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    report_data = {"plan": _plan_as_dict(plan), "applied": False}
    args.report.write_text(
        json.dumps(report_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(
        f"matched={len(plan.matched_patient_actions)} "
        f"renames={len(plan.rename_actions)} "
        f"new={len(plan.new_patient_actions)} "
        f"unmatched_target={len(plan.unmatched_target_actions)} "
        f"excel={len(plan.excel_actions)} "
        f"metadata={len(plan.metadata_actions)} "
        f"conflicts={len(plan.conflicts)}"
    )
    print(f"report={args.report}")
    if plan.conflicts:
        for conflict in plan.conflicts:
            print(f"CONFLICT: {conflict}")
        raise SystemExit(2)
    if not args.apply:
        print("dry-run only; pass --apply to execute")
        return

    result = execute_sync_plan(plan, backup_root=args.backup_root)
    report_data["applied"] = True
    report_data["result"] = asdict(result)
    report_data["backup_root"] = str(args.backup_root.resolve())
    args.report.write_text(
        json.dumps(report_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
