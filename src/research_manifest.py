"""Patient identity pseudonymization and manifest validation helpers."""

from __future__ import annotations

import hashlib
import hmac
import math
from pathlib import PurePosixPath
from typing import Any


def extract_identity_token(annotation: dict[str, Any]) -> str:
    folder = str(annotation.get("info", {}).get("folder", "")).strip()
    normalized = folder.replace("\\", "/").rstrip("/")
    if not normalized:
        raise ValueError("Annotation does not contain info.folder")
    token = PurePosixPath(normalized).name.strip()
    if not token:
        raise ValueError("Annotation identity token is empty")
    return token


def pseudonymous_key(identity: str, secret: bytes, prefix: str) -> str:
    digest = hmac.new(
        secret,
        identity.strip().casefold().encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    return f"{prefix}_{digest[:16].upper()}"


def resolve_identity_tokens(
    identities: set[str],
    patient_folder: str,
) -> str:
    if not identities:
        raise ValueError("No identity token was found")
    folder_alias = patient_folder.strip().casefold()
    original_tokens = {
        token for token in identities if token.strip().casefold() != folder_alias
    }
    candidates = original_tokens or identities
    if len(candidates) != 1:
        raise ValueError("Multiple non-alias identity tokens were found")
    return next(iter(candidates))


def validate_roi(
    rect: dict[str, Any] | None,
    width: int | float,
    height: int | float,
) -> bool:
    if not rect:
        return False
    try:
        x1, y1, x2, y2 = (
            float(rect[name]) for name in ("x1", "y1", "x2", "y2")
        )
        width, height = float(width), float(height)
    except (KeyError, TypeError, ValueError):
        return False
    values = (x1, y1, x2, y2, width, height)
    if not all(math.isfinite(value) for value in values):
        return False
    return 0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height
