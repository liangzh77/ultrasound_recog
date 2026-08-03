"""Verify the continuous research ledger, hashes, links, and privacy boundary."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.research_ledger import validate_research_ledger  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ledger",
        type=Path,
        default=ROOT / "docs" / "research" / "experiment_ledger.yaml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            ROOT
            / "workspace"
            / "experiments"
            / "active"
            / "exp_2026-07_patient_multimodal_v1"
            / "reports"
            / "documentation_audit.json"
        ),
    )
    return parser.parse_args()


def _markdown_files() -> list[Path]:
    roots = [
        ROOT / "README.md",
        ROOT / "docs" / "research",
        ROOT / "docs" / "project",
        ROOT / "docs" / "decisions",
    ]
    files: list[Path] = []
    for root in roots:
        if root.is_file():
            files.append(root)
        elif root.is_dir():
            files.extend(root.rglob("*.md"))
    return sorted(set(files))


def _local_markdown_target(target: str) -> str | None:
    cleaned = target.strip().strip("<>")
    if not cleaned or cleaned.startswith("#"):
        return None
    if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", cleaned):
        return None
    if cleaned.startswith("mailto:"):
        return None
    return unquote(cleaned.split("#", 1)[0])


def _validate_markdown_links(path: Path) -> int:
    text = path.read_text(encoding="utf-8")
    count = 0
    for target in re.findall(r"\[[^]]+\]\(([^)]+)\)", text):
        local_target = _local_markdown_target(target)
        if local_target is None:
            continue
        resolved = (path.parent / local_target).resolve()
        if not resolved.exists():
            raise ValueError(f"Broken Markdown link in {path}: {target}")
        count += 1
    return count


def main() -> int:
    args = parse_args()
    result = validate_research_ledger(args.ledger.resolve(), ROOT, verify_artifacts=True)
    markdown_files = _markdown_files()
    link_count = sum(_validate_markdown_links(path) for path in markdown_files)
    report = {
        "status": "PASS",
        **result,
        "markdown_files": len(markdown_files),
        "validated_relative_links": link_count,
        "ledger": args.ledger.resolve().relative_to(ROOT).as_posix(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
