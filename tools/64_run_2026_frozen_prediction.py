"""Generate the one-time diagnosis-hidden R2 patient prediction package."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.common_paths import (  # noqa: E402
    PATIENT_MULTIMODAL_REGISTRY_DIR,
    PATIENT_MULTIMODAL_REPORTS_DIR,
)
from src.research_auto_roi import ROIBox, require_sha256, sha256_file  # noqa: E402
from src.research_config import load_research_config  # noqa: E402
from src.research_dataset import ResearchImageRecord  # noqa: E402
from src.research_frozen_inference import (  # noqa: E402
    confidence_flag,
    hierarchical_probabilities,
    median_roi_box,
    run_unlabeled_patient_inference,
)
from src.research_schema import DIAGNOSIS_CLASSES  # noqa: E402


CONFIG_PATH = ROOT / "configs/research/r2_2026_frozen_inference.yaml"
PUBLIC_IMAGE_PATH = PATIENT_MULTIMODAL_REPORTS_DIR / "r2_2026/2026_images_public.csv"
PRIVATE_DIR = PATIENT_MULTIMODAL_REGISTRY_DIR / "private/r2_2026"
SOURCE_PATH = PRIVATE_DIR / "2026_image_sources.csv"
IDENTITY_PATH = PRIVATE_DIR / "2026_person_identity_map.csv"
C3_PATH = PRIVATE_DIR / "2026_c3_conditional_probabilities.csv"
C3_REPORT_PATH = PATIENT_MULTIMODAL_REPORTS_DIR / "r2_2026/2026_c3_component_manifest.json"
MANIFEST_PATH = PATIENT_MULTIMODAL_REPORTS_DIR / "r2_2026/2026_formal_prediction_manifest.json"
PRIVATE_PREDICTION_PATH = PRIVATE_DIR / "2026_formal_predictions_private.csv"
HOSPITAL_DELIVERY_PATH = PRIVATE_DIR / "2026_hospital_prediction_delivery.csv"
ROI_DELIVERY_PATH = PRIVATE_DIR / "2026_hospital_roi_delivery.csv"
ROI_PATH = PRIVATE_DIR / "2026_r0_median_rois.csv"
E1_PATH = PRIVATE_DIR / "2026_e1_ensemble_probabilities.csv"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def git_revision_clean() -> str:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    if status:
        raise RuntimeError("Formal 2026 prediction requires a clean Git worktree")
    return revision


def definition(mapping: dict, fold: int) -> dict[str, str]:
    return mapping.get(fold, mapping.get(str(fold)))


def validate_registry(
    public_rows: list[dict[str, str]], source_rows: list[dict[str, str]], expected_images: int
) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, str]]]:
    public = {row["image_key"]: row for row in public_rows}
    sources = {row["image_key"]: row for row in source_rows}
    if len(public) != expected_images or len(sources) != expected_images or set(public) != set(sources):
        raise ValueError("2026 public/private image registries do not match exactly")
    for key, row in sources.items():
        if row["person_key"] != public[key]["person_key"]:
            raise ValueError("2026 public/private image patient mappings differ")
        path = (ROOT / row["raw_image_path"]).resolve()
        require_sha256(path, row["content_sha256"], "2026 source image")
    return public, sources


def infer_rois(
    config: dict,
    public: dict[str, dict[str, str]],
    sources: dict[str, dict[str, str]],
) -> tuple[dict[str, ROIBox], list[dict[str, Any]], list[dict[str, Any]]]:
    from ultralytics import YOLO

    contract = config["auto_roi_ensemble"]
    base = contract["preprocessing_config"]
    require_sha256(ROOT / base["path"], base["sha256"], "R0 preprocessing config")
    r0_config = yaml.safe_load((ROOT / base["path"]).read_text(encoding="utf-8"))
    ordered_keys = sorted(public)
    source_files = [(ROOT / sources[key]["raw_image_path"]).resolve() for key in ordered_keys]
    path_to_key = {str(path).casefold(): key for path, key in zip(source_files, ordered_keys, strict=True)}
    boxes_by_image: dict[str, list[ROIBox]] = defaultdict(list)
    confidences_by_image: dict[str, list[float]] = defaultdict(list)
    model_manifest = []
    for fold in range(5):
        member = definition(contract["checkpoints"], fold)
        checkpoint = ROOT / member["path"]
        digest = require_sha256(checkpoint, member["sha256"], f"R0 fold {fold}")
        model = YOLO(str(checkpoint))
        seen = set()
        results = model.predict(
            source=[str(path) for path in source_files],
            imgsz=int(r0_config["model"]["image_size"]),
            conf=float(contract["inference_confidence"]),
            device=r0_config["runtime"]["device"],
            stream=True,
            verbose=False,
        )
        for result in results:
            key = path_to_key.get(str(Path(result.path).resolve()).casefold())
            if key is None or key in seen:
                raise ValueError("R0 returned an unknown or duplicate 2026 image")
            seen.add(key)
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue
            best = int(boxes.conf.argmax().item())
            predicted = ROIBox(*(float(value) for value in boxes.xyxy[best].tolist()))
            predicted.validate(int(public[key]["width"]), int(public[key]["height"]))
            boxes_by_image[key].append(predicted)
            confidences_by_image[key].append(float(boxes.conf[best].item()))
        if seen != set(ordered_keys):
            raise ValueError("R0 did not return every 2026 image exactly once")
        model_manifest.append({"fold": fold, "sha256": digest})
        del model

    roi_by_image = {}
    roi_rows = []
    for key in ordered_keys:
        width, height = int(public[key]["width"]), int(public[key]["height"])
        roi = median_roi_box(boxes_by_image[key], width, height)
        fallback = roi is None
        if fallback:
            roi = ROIBox(0.0, 0.0, float(width), float(height))
        roi_by_image[key] = roi
        roi_rows.append(
            {
                "image_key": key,
                "person_key": public[key]["person_key"],
                "image_index": int(public[key]["image_index"]),
                "roi_x1": roi.x1,
                "roi_y1": roi.y1,
                "roi_x2": roi.x2,
                "roi_y2": roi.y2,
                "available_r0_members": len(boxes_by_image[key]),
                "mean_detection_confidence": (
                    float(np.mean(confidences_by_image[key])) if confidences_by_image[key] else ""
                ),
                "full_image_fallback": int(fallback),
            }
        )
    return roi_by_image, roi_rows, model_manifest


def infer_e1(
    config: dict,
    public: dict[str, dict[str, str]],
    sources: dict[str, dict[str, str]],
    roi_by_image: dict[str, ROIBox],
) -> tuple[tuple[str, ...], np.ndarray, np.ndarray, list[dict[str, Any]]]:
    import torch
    from torch.utils.data import DataLoader

    from src.research_dataset import (
        ResearchImageDataset,
        UnlabeledPatientBagDataset,
        collate_unlabeled_patient_bags,
    )
    from src.research_models import MaskedMeanClassifier, create_timm_encoder
    from src.research_runtime import configure_conservative_threads, set_below_normal_priority

    configure_conservative_threads()
    set_below_normal_priority()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for frozen 2026 image inference")
    contract = config["image_classifier_ensemble"]
    base = contract["preprocessing_config"]
    require_sha256(ROOT / base["path"], base["sha256"], "E1 preprocessing config")
    e1_config = load_research_config(ROOT / base["path"])
    records = []
    for key in sorted(public):
        roi = roi_by_image[key]
        records.append(
            ResearchImageRecord(
                image_key=key,
                person_key=public[key]["person_key"],
                diagnosis="",
                diagnosis_id=-1,
                image_path=(ROOT / sources[key]["raw_image_path"]).resolve(),
                roi={name: float(getattr(roi, name)) for name in ("x1", "y1", "x2", "y2")},
            )
        )
    member_probabilities = []
    expected_keys: tuple[str, ...] | None = None
    image_counts: np.ndarray | None = None
    model_manifest = []
    for fold in range(5):
        preprocessing = definition(contract["preprocessing_summaries"], fold)
        summary_path = ROOT / preprocessing["path"]
        require_sha256(summary_path, preprocessing["sha256"], f"E1 preprocessing fold {fold}")
        fill = tuple(int(value) for value in preprocessing["letterbox_fill_rgb"])
        image_dataset = ResearchImageDataset(
            records,
            input_mode="roi",
            resize_mode=e1_config["data"]["resize_mode"],
            output_size=int(e1_config["data"]["output_size"]),
            image_transform=None,
            letterbox_fill=fill,
        )
        loader = DataLoader(
            UnlabeledPatientBagDataset(image_dataset),
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            collate_fn=collate_unlabeled_patient_bags,
        )
        member = definition(contract["checkpoints"], fold)
        checkpoint = ROOT / member["path"]
        digest = require_sha256(checkpoint, member["sha256"], f"E1 fold {fold}")
        encoder, feature_dim = create_timm_encoder(e1_config["model"]["name"], pretrained=False)
        model = MaskedMeanClassifier(
            encoder,
            feature_dim,
            num_classes=int(e1_config["model"]["num_classes"]),
            dropout=float(e1_config["model"]["dropout"]),
        ).cuda()
        state = torch.load(checkpoint, map_location="cuda", weights_only=False)
        model.load_state_dict(state["model"])
        result = run_unlabeled_patient_inference(
            model,
            loader,
            device=torch.device("cuda"),
            expected_columns=6,
            amp=bool(e1_config["training"]["amp"]),
            instance_chunk_size=int(e1_config["data"]["max_instances_train"]),
        )
        keys = tuple(result["person_keys"])
        counts = np.asarray(result["image_counts"], dtype=np.int64)
        if expected_keys is None:
            expected_keys, image_counts = keys, counts
        elif keys != expected_keys or not np.array_equal(counts, image_counts):
            raise ValueError("E1 member patient order or image counts differ")
        member_probabilities.append(result["probabilities"])
        model_manifest.append({"fold": fold, "sha256": digest, "letterbox_fill_rgb": list(fill)})
        del model, encoder, loader, image_dataset
        torch.cuda.empty_cache()
    assert expected_keys is not None and image_counts is not None
    ensemble = np.mean(np.stack(member_probabilities, axis=0), axis=0)
    if not np.isfinite(ensemble).all() or not np.allclose(ensemble.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("E1 ensemble probabilities are invalid")
    return expected_keys, image_counts, ensemble, model_manifest


def final_rows(
    person_keys: tuple[str, ...],
    image_counts: np.ndarray,
    image_probabilities: np.ndarray,
    c3_rows: dict[str, dict[str, str]],
    roi_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    roi_by_person: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in roi_rows:
        roi_by_person[str(row["person_key"])].append(row)
    c3_columns = ("prob_ra", "prob_ga", "prob_spa", "prob_oa", "prob_injury")
    probability_slugs = ("normal", "ra", "ga", "spa", "oa", "injury")
    rows = []
    for index, key in enumerate(person_keys):
        clinical = [float(c3_rows[key][column]) for column in c3_columns]
        hierarchical = np.asarray(
            hierarchical_probabilities(image_probabilities[index], clinical), dtype=np.float64
        )
        image_ranking = np.argsort(image_probabilities[index])[::-1]
        hierarchical_ranking = np.argsort(hierarchical)[::-1]
        image_flag, image_reasons = confidence_flag(image_probabilities[index])
        hierarchical_flag, hierarchical_reasons = confidence_flag(hierarchical)
        patient_rois = roi_by_person[key]
        row: dict[str, Any] = {
            "person_key": key,
            "image_count": int(image_counts[index]),
            "roi_full_image_fallback_images": sum(int(item["full_image_fallback"]) for item in patient_rois),
            "roi_min_available_r0_members": min(int(item["available_r0_members"]) for item in patient_rois),
            "image_top1": DIAGNOSIS_CLASSES[int(image_ranking[0])],
            "image_top2": DIAGNOSIS_CLASSES[int(image_ranking[1])],
            "image_low_confidence": int(image_flag),
            "image_low_confidence_reasons": "|".join(image_reasons),
            "hierarchical_top1": DIAGNOSIS_CLASSES[int(hierarchical_ranking[0])],
            "hierarchical_top2": DIAGNOSIS_CLASSES[int(hierarchical_ranking[1])],
            "hierarchical_low_confidence": int(hierarchical_flag),
            "hierarchical_low_confidence_reasons": "|".join(hierarchical_reasons),
        }
        for slug, value in zip(probability_slugs, image_probabilities[index], strict=True):
            row[f"image_prob_{slug}"] = float(value)
        for slug, value in zip(probability_slugs, hierarchical, strict=True):
            row[f"hierarchical_prob_{slug}"] = float(value)
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    revision = git_revision_clean()
    final_paths = (MANIFEST_PATH, PRIVATE_PREDICTION_PATH, HOSPITAL_DELIVERY_PATH, ROI_DELIVERY_PATH, ROI_PATH, E1_PATH)
    if any(path.exists() for path in final_paths):
        raise FileExistsError("One-time 2026 formal prediction outputs already exist")
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    if config.get("status") != "development_route_frozen_ready_for_one_formal_prediction":
        raise ValueError("R2 route is not ready for formal prediction")
    if config["route"]["selected"] != "full_auto" or config["route"].get("manual_lesion_contours_required") is not False:
        raise ValueError("R2 final route must be full-auto without manual lesion contours")
    public_rows = read_csv(PUBLIC_IMAGE_PATH)
    source_rows = read_csv(SOURCE_PATH)
    public, sources = validate_registry(public_rows, source_rows, int(config["cohort"]["images"]))
    if len({row["person_key"] for row in public_rows}) != int(config["cohort"]["patients"]):
        raise ValueError("2026 image registry patient count differs from R2")

    roi_by_image, roi_rows, r0_manifest = infer_rois(config, public, sources)
    person_keys, image_counts, image_probabilities, e1_manifest = infer_e1(
        config, public, sources, roi_by_image
    )
    c3_report = json.loads(C3_REPORT_PATH.read_text(encoding="utf-8"))
    require_sha256(C3_PATH, c3_report["private_prediction_sha256"], "2026 C3 component")
    if c3_report.get("reference_diagnoses_read") is not False:
        raise ValueError("C3 component does not preserve diagnosis blinding")
    c3_rows = {row["person_key"]: row for row in read_csv(C3_PATH)}
    if set(c3_rows) != set(person_keys):
        raise ValueError("2026 C3 and E1 patient sets differ")
    predictions = final_rows(person_keys, image_counts, image_probabilities, c3_rows, roi_rows)

    identities = {row["person_key"]: row["hospital_patient_number"] for row in read_csv(IDENTITY_PATH)}
    if set(identities) != set(person_keys):
        raise ValueError("2026 identity map differs from prediction patients")
    hospital_rows = [
        {"hospital_patient_number": identities[row["person_key"]], **{k: v for k, v in row.items() if k != "person_key"}}
        for row in predictions
    ]
    hospital_roi_rows = [
        {"hospital_patient_number": identities[row["person_key"]], **{k: v for k, v in row.items() if k != "person_key"}}
        for row in roi_rows
    ]
    e1_rows = [
        {
            "person_key": key,
            **{
                f"prob_{slug}": float(value)
                for slug, value in zip(("normal", "ra", "ga", "spa", "oa", "injury"), image_probabilities[index], strict=True)
            },
        }
        for index, key in enumerate(person_keys)
    ]
    write_csv(ROI_PATH, roi_rows)
    write_csv(E1_PATH, e1_rows)
    write_csv(PRIVATE_PREDICTION_PATH, predictions)
    write_csv(HOSPITAL_DELIVERY_PATH, hospital_rows)
    write_csv(ROI_DELIVERY_PATH, hospital_roi_rows)

    image_top1 = Counter(row["image_top1"] for row in predictions)
    hierarchical_top1 = Counter(row["hierarchical_top1"] for row in predictions)
    manifest = {
        "experiment_code": "R2-2026-formal",
        "status": "formal_prediction_frozen_waiting_for_hospital_reference_review",
        "git_commit": revision,
        "git_dirty_at_start": False,
        "config_sha256": sha256_file(CONFIG_PATH),
        "patients": len(predictions),
        "images": len(roi_rows),
        "r0_model_manifest": r0_manifest,
        "e1_model_manifest": e1_manifest,
        "c3_component_sha256": sha256_file(C3_PATH),
        "roi_full_image_fallback_images": sum(int(row["full_image_fallback"]) for row in roi_rows),
        "roi_images_with_all_five_members": sum(int(row["available_r0_members"]) == 5 for row in roi_rows),
        "image_top1_counts": dict(image_top1),
        "hierarchical_top1_counts": dict(hierarchical_top1),
        "image_low_confidence_patients": sum(int(row["image_low_confidence"]) for row in predictions),
        "hierarchical_low_confidence_patients": sum(int(row["hierarchical_low_confidence"]) for row in predictions),
        "artifacts": {
            "private_prediction_sha256": sha256_file(PRIVATE_PREDICTION_PATH),
            "hospital_delivery_sha256": sha256_file(HOSPITAL_DELIVERY_PATH),
            "hospital_roi_delivery_sha256": sha256_file(ROI_DELIVERY_PATH),
            "private_roi_sha256": sha256_file(ROI_PATH),
            "private_e1_sha256": sha256_file(E1_PATH),
        },
        "reference_diagnoses_read": False,
        "route_selection_affected_by_2026": False,
        "patient_probabilities_publicly_reported": False,
        "clinical_use_authorized": False,
    }
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(MANIFEST_PATH), "manifest_sha256": sha256_file(MANIFEST_PATH), **manifest}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
