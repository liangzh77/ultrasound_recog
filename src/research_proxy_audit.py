"""Patient-level audit for nonmedical acquisition and export proxies."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
from PIL import Image


GEOMETRY_FEATURES = (
    "width",
    "height",
    "log_pixel_count",
    "image_aspect_ratio",
    "roi_x1_fraction",
    "roi_y1_fraction",
    "roi_x2_fraction",
    "roi_y2_fraction",
    "roi_width_fraction",
    "roi_height_fraction",
    "roi_area_fraction",
    "roi_aspect_ratio",
    "log_bytes_per_pixel",
    "is_jpeg",
    "is_png",
)

PIXEL_STAT_SUFFIXES = (
    "red_mean",
    "green_mean",
    "blue_mean",
    "red_std",
    "green_std",
    "blue_std",
    "luminance_mean",
    "luminance_std",
    "black_fraction",
    "white_fraction",
    "gray_fraction",
    "colour_fraction",
)

FULL_BORDER_FEATURES = tuple(f"full_border_{name}" for name in PIXEL_STAT_SUFFIXES)
OUTSIDE_ROI_FEATURES = tuple(f"outside_roi_{name}" for name in PIXEL_STAT_SUFFIXES)
ROI_BORDER_FEATURES = tuple(f"roi_border_{name}" for name in PIXEL_STAT_SUFFIXES)

DIMENSION_EXPORT_FEATURES = (
    "width",
    "height",
    "log_pixel_count",
    "image_aspect_ratio",
    "log_bytes_per_pixel",
    "is_jpeg",
    "is_png",
)
ROI_GEOMETRY_FEATURES = (
    "roi_x1_fraction",
    "roi_y1_fraction",
    "roi_x2_fraction",
    "roi_y2_fraction",
    "roi_width_fraction",
    "roi_height_fraction",
    "roi_area_fraction",
    "roi_aspect_ratio",
)
ROI_RESOLUTION_FEATURES = (
    "roi_width_pixels",
    "roi_height_pixels",
    "log_roi_pixel_count",
    "roi_aspect_ratio",
)

FEATURE_GROUPS = {
    "image_count_only": (),
    "dimensions_export": DIMENSION_EXPORT_FEATURES,
    "roi_geometry": ROI_GEOMETRY_FEATURES,
    "outer_pixels": FULL_BORDER_FEATURES + OUTSIDE_ROI_FEATURES,
    "full_visible_nonmedical": (
        "image_aspect_ratio",
        *FULL_BORDER_FEATURES,
        *OUTSIDE_ROI_FEATURES,
    ),
    "roi_aspect_visible": ("roi_aspect_ratio",),
    "roi_resolution_upper_bound": ROI_RESOLUTION_FEATURES,
    "roi_edge_visible_control": (
        "roi_aspect_ratio",
        *ROI_BORDER_FEATURES,
    ),
    "geometry": GEOMETRY_FEATURES,
    "outer_nonmedical": GEOMETRY_FEATURES
    + FULL_BORDER_FEATURES
    + OUTSIDE_ROI_FEATURES,
    "roi_edge_control": GEOMETRY_FEATURES + ROI_BORDER_FEATURES,
}

HIGH_PROXY_MACRO_F1 = 0.20
HIGH_PROXY_MACRO_AUC = 0.65
MODERATE_PROXY_MACRO_F1 = 0.17
MODERATE_PROXY_MACRO_AUC = 0.58
PIXEL_SAMPLE_LIMIT = 4_096


@dataclass(frozen=True)
class ProxyTable:
    person_keys: tuple[str, ...]
    targets: np.ndarray
    outer_folds: np.ndarray
    features: np.ndarray
    feature_names: tuple[str, ...]


def _clamped_roi(
    roi: Mapping[str, float], width: int, height: int
) -> tuple[int, int, int, int]:
    x1 = max(0, min(width - 1, int(round(float(roi["x1"])))))
    y1 = max(0, min(height - 1, int(round(float(roi["y1"])))))
    x2 = max(x1 + 1, min(width, int(round(float(roi["x2"])))))
    y2 = max(y1 + 1, min(height, int(round(float(roi["y2"])))))
    return x1, y1, x2, y2


def _sample_strips(
    strips: Iterable[np.ndarray],
    limit: int = PIXEL_SAMPLE_LIMIT,
) -> np.ndarray:
    flattened = [item.reshape(-1, 3) for item in strips if item.size]
    if not flattened:
        return np.zeros((1, 3), dtype=np.uint8)
    # Sample each view before concatenating. Some outside-ROI strips contain
    # almost the whole frame; materializing all of them made this audit spend
    # most of its time copying pixels that were immediately discarded.
    per_strip_limit = max(1, limit // len(flattened))
    sampled = []
    for pixels in flattened:
        if len(pixels) > per_strip_limit:
            indices = np.linspace(
                0,
                len(pixels) - 1,
                per_strip_limit,
                dtype=np.int64,
            )
            pixels = pixels[indices]
        sampled.append(pixels)
    pixels = np.concatenate(sampled, axis=0)
    if len(pixels) > limit:
        indices = np.linspace(0, len(pixels) - 1, limit, dtype=np.int64)
        pixels = pixels[indices]
    return pixels


def _pixel_statistics(pixels: np.ndarray, prefix: str) -> dict[str, float]:
    values = pixels.astype(np.float32)
    channel_mean = values.mean(axis=0) / 255.0
    channel_std = values.std(axis=0) / 255.0
    luminance = (
        0.2126 * values[:, 0] + 0.7152 * values[:, 1] + 0.0722 * values[:, 2]
    ) / 255.0
    channel_range = values.max(axis=1) - values.min(axis=1)
    maximum = values.max(axis=1)
    minimum = values.min(axis=1)
    raw = (
        *channel_mean,
        *channel_std,
        luminance.mean(),
        luminance.std(),
        np.mean(maximum < 16),
        np.mean(minimum > 240),
        np.mean(channel_range < 3),
        np.mean((channel_range > 64) & (maximum > 128)),
    )
    return {
        f"{prefix}_{name}": float(value)
        for name, value in zip(PIXEL_STAT_SUFFIXES, raw, strict=True)
    }


def extract_image_proxy_features(
    image_path: Path,
    roi: Mapping[str, float],
) -> dict[str, float]:
    """Extract only acquisition/export summaries; never derive names or labels."""
    with Image.open(image_path) as source:
        image = np.asarray(source.convert("RGB"), dtype=np.uint8)
        image_format = (source.format or image_path.suffix.lstrip(".")).casefold()
    height, width = image.shape[:2]
    x1, y1, x2, y2 = _clamped_roi(roi, width, height)
    roi_width = x2 - x1
    roi_height = y2 - y1
    pixel_count = width * height
    suffix = image_path.suffix.casefold()
    features = {
        "width": float(width),
        "height": float(height),
        "log_pixel_count": float(np.log1p(pixel_count)),
        "image_aspect_ratio": float(width / height),
        "roi_x1_fraction": float(x1 / width),
        "roi_y1_fraction": float(y1 / height),
        "roi_x2_fraction": float(x2 / width),
        "roi_y2_fraction": float(y2 / height),
        "roi_width_fraction": float(roi_width / width),
        "roi_height_fraction": float(roi_height / height),
        "roi_area_fraction": float((roi_width * roi_height) / pixel_count),
        "roi_aspect_ratio": float(roi_width / roi_height),
        "roi_width_pixels": float(roi_width),
        "roi_height_pixels": float(roi_height),
        "log_roi_pixel_count": float(np.log1p(roi_width * roi_height)),
        "log_bytes_per_pixel": float(
            np.log1p(image_path.stat().st_size / pixel_count)
        ),
        "is_jpeg": float(image_format in {"jpeg", "jpg"} or suffix in {".jpg", ".jpeg"}),
        "is_png": float(image_format == "png" or suffix == ".png"),
    }

    full_band = max(1, round(min(width, height) * 0.03))
    full_border = _sample_strips(
        (
            image[:full_band],
            image[-full_band:],
            image[full_band:-full_band, :full_band],
            image[full_band:-full_band, -full_band:],
        )
    )
    outside_roi = _sample_strips(
        (
            image[:y1],
            image[y2:],
            image[y1:y2, :x1],
            image[y1:y2, x2:],
        )
    )
    roi_pixels = image[y1:y2, x1:x2]
    roi_band = max(1, round(min(roi_width, roi_height) * 0.03))
    roi_border = _sample_strips(
        (
            roi_pixels[:roi_band],
            roi_pixels[-roi_band:],
            roi_pixels[roi_band:-roi_band, :roi_band],
            roi_pixels[roi_band:-roi_band, -roi_band:],
        )
    )
    features.update(_pixel_statistics(full_border, "full_border"))
    features.update(_pixel_statistics(outside_roi, "outside_roi"))
    features.update(_pixel_statistics(roi_border, "roi_border"))
    return features


def aggregate_patient_proxy_features(
    image_rows: Sequence[Mapping[str, float | int | str]],
    image_feature_names: Sequence[str],
) -> ProxyTable:
    groups: dict[str, list[Mapping[str, float | int | str]]] = defaultdict(list)
    for row in image_rows:
        groups[str(row["person_key"])].append(row)
    if not groups:
        raise ValueError("No proxy feature rows were supplied")

    aggregations = ("mean", "std", "min", "max")
    feature_names = ("image_count",) + tuple(
        f"{name}_{aggregation}"
        for name in image_feature_names
        for aggregation in aggregations
    )
    person_keys = tuple(sorted(groups))
    targets: list[int] = []
    outer_folds: list[int] = []
    matrix: list[list[float]] = []
    for person_key in person_keys:
        rows = groups[person_key]
        patient_targets = {int(row["diagnosis_id"]) for row in rows}
        patient_folds = {int(row["outer_fold"]) for row in rows}
        if len(patient_targets) != 1 or len(patient_folds) != 1:
            raise ValueError(f"Mixed target or fold for patient {person_key}")
        targets.append(patient_targets.pop())
        outer_folds.append(patient_folds.pop())
        patient_features = [float(len(rows))]
        for name in image_feature_names:
            values = np.asarray([float(row[name]) for row in rows], dtype=np.float64)
            patient_features.extend(
                (values.mean(), values.std(ddof=0), values.min(), values.max())
            )
        matrix.append(patient_features)
    return ProxyTable(
        person_keys=person_keys,
        targets=np.asarray(targets, dtype=np.int64),
        outer_folds=np.asarray(outer_folds, dtype=np.int64),
        features=np.asarray(matrix, dtype=np.float64),
        feature_names=feature_names,
    )


def run_proxy_oof(
    table: ProxyTable,
    class_count: int,
    seed: int,
    targets: np.ndarray | None = None,
) -> np.ndarray:
    """Fit preprocessing and classifier independently inside each outer fold."""
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    labels = table.targets if targets is None else np.asarray(targets, dtype=np.int64)
    if labels.shape != table.targets.shape:
        raise ValueError("Proxy targets must contain one label per patient")
    expected_classes = np.arange(class_count)
    probabilities = np.full((len(labels), class_count), np.nan, dtype=np.float64)
    folds = np.unique(table.outer_folds)
    if len(folds) < 2:
        raise ValueError("Proxy audit requires at least two outer folds")
    for fold in folds:
        test_mask = table.outer_folds == fold
        train_mask = ~test_mask
        train_classes = np.unique(labels[train_mask])
        if not np.array_equal(train_classes, expected_classes):
            raise ValueError(f"Training fold {fold} does not contain every class")
        model = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        C=1.0,
                        class_weight="balanced",
                        max_iter=2_000,
                        random_state=seed + int(fold),
                    ),
                ),
            ]
        )
        model.fit(table.features[train_mask], labels[train_mask])
        fold_probabilities = model.predict_proba(table.features[test_mask])
        classes = model.named_steps["classifier"].classes_.astype(int)
        probabilities[np.ix_(test_mask, classes)] = fold_probabilities
    if not np.isfinite(probabilities).all():
        raise RuntimeError("Proxy OOF did not produce complete finite probabilities")
    return probabilities


def permute_targets_within_folds(
    targets: np.ndarray,
    outer_folds: np.ndarray,
    seed: int,
) -> np.ndarray:
    """Shuffle labels without changing the class count of any test fold."""
    targets = np.asarray(targets, dtype=np.int64)
    outer_folds = np.asarray(outer_folds, dtype=np.int64)
    if targets.shape != outer_folds.shape:
        raise ValueError("Targets and folds must have identical shapes")
    result = targets.copy()
    generator = np.random.default_rng(seed)
    for fold in np.unique(outer_folds):
        indices = np.flatnonzero(outer_folds == fold)
        result[indices] = generator.permutation(targets[indices])
    return result


def assess_proxy_risk(
    macro_f1: float,
    macro_auc: float,
    permutation_p_value: float,
) -> str:
    """Apply the fixed, deliberately conservative A1 shortcut thresholds."""
    statistically_detectable = permutation_p_value <= 0.05
    if statistically_detectable and (
        macro_f1 >= HIGH_PROXY_MACRO_F1 or macro_auc >= HIGH_PROXY_MACRO_AUC
    ):
        return "high"
    if statistically_detectable or (
        macro_f1 >= MODERATE_PROXY_MACRO_F1
        or macro_auc >= MODERATE_PROXY_MACRO_AUC
    ):
        return "moderate"
    return "low"


def proxy_permutation_test(
    table: ProxyTable,
    class_count: int,
    observed_macro_f1: float,
    count: int,
    seed: int,
) -> dict[str, float | list[float]]:
    """Refit the complete fold-local pipeline under fold-stratified label nulls."""
    from sklearn.metrics import f1_score

    if count < 1:
        raise ValueError("Permutation count must be positive")
    scores = []
    labels = np.arange(class_count)
    for index in range(count):
        permuted = permute_targets_within_folds(
            table.targets,
            table.outer_folds,
            seed=seed + index,
        )
        probabilities = run_proxy_oof(
            table,
            class_count=class_count,
            seed=seed + index,
            targets=permuted,
        )
        scores.append(
            float(
                f1_score(
                    permuted,
                    probabilities.argmax(axis=1),
                    labels=labels,
                    average="macro",
                    zero_division=0,
                )
            )
        )
    null_scores = np.asarray(scores, dtype=np.float64)
    exceedances = int(np.count_nonzero(null_scores >= observed_macro_f1))
    return {
        "count": count,
        "seed": seed,
        "p_value": float((exceedances + 1) / (count + 1)),
        "null_mean": float(null_scores.mean()),
        "null_95th_percentile": float(np.quantile(null_scores, 0.95)),
        "null_macro_f1": scores,
    }
