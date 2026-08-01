"""Versioned, disease-independent annotation normalization."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from src.label_mapping import fix_label


def normalize_annotation(
    source: dict[str, Any],
    disease: str,
) -> tuple[dict[str, Any], dict[str, str]]:
    normalized = deepcopy(source)
    changes = {}
    for item in normalized.get("objects", []):
        old = str(item.get("category", ""))
        new = fix_label(old, disease)
        item["category"] = new
        if old != new:
            changes[old] = new
    return normalized, changes


def annotation_geometry_signature(source: dict[str, Any]) -> str:
    """Serialize every annotation field except category names."""
    geometry = {
        "info": {
            key: source.get("info", {}).get(key)
            for key in ("width", "height", "depth")
        },
        "objects": [
            {key: value for key, value in item.items() if key != "category"}
            for item in source.get("objects", [])
        ],
        "ultrasound_rect": source.get("ultrasound_rect"),
        "ultrasound_candidates": source.get("ultrasound_candidates"),
        "ultrasound_rect_reviewed": source.get("ultrasound_rect_reviewed"),
    }
    return json.dumps(
        geometry,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def load_and_normalize(
    source_path: Path,
    disease: str,
) -> tuple[dict[str, Any], dict[str, str], int]:
    source = json.loads(source_path.read_text(encoding="utf-8"))
    normalized, changes = normalize_annotation(source, disease)
    if annotation_geometry_signature(source) != annotation_geometry_signature(
        normalized
    ):
        raise ValueError(f"Geometry changed during normalization: {source_path}")
    return normalized, changes, len(source.get("objects", []))
