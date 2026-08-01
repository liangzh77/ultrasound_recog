"""Read-only source fingerprinting for research data freezes."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass(frozen=True)
class SourceFingerprint:
    cohort: str
    relative_path: str
    kind: str
    size_bytes: int
    mtime_ns: int
    sha256: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def sha256_file(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def source_kind(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in IMAGE_EXTENSIONS and any(
        parent.name.casefold() == "mask" for parent in path.parents
    ):
        return "annotation_mask"
    if suffix in IMAGE_EXTENSIONS:
        return "image"
    if suffix == ".json":
        return "annotation"
    if suffix in {".xlsx", ".xls"}:
        return "clinical_workbook"
    if suffix in {".yaml", ".yml"}:
        return "annotation_config"
    return "other"


def fingerprint_tree(
    root: Path,
    cohort: str,
    relative_to: Path,
) -> list[SourceFingerprint]:
    records = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path.name.startswith("~$"):
            continue
        stat = path.stat()
        records.append(
            SourceFingerprint(
                cohort=cohort,
                relative_path=path.relative_to(relative_to).as_posix(),
                kind=source_kind(path),
                size_bytes=stat.st_size,
                mtime_ns=stat.st_mtime_ns,
                sha256=sha256_file(path),
            )
        )
    return records


def dataset_version(records: list[SourceFingerprint]) -> str:
    digest = hashlib.sha256()
    for record in sorted(records, key=lambda item: item.relative_path):
        digest.update(record.relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(record.sha256.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()
