"""Evaluate ROI polygon segmentation without external COCO dependencies."""

import argparse
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.common_paths import (
    CATEGORY_MAPPING_FILE,
    CLEANED_DIR,
    ROI_POLY_SEG_ARTIFACTS_DIR,
    ROI_POLY_SEG_REPORTS_DIR,
    SPLITS_DIR,
    TOOLS_DIR,
)
from src.data_utils import load_category_mapping, load_isat_json, load_split_file
from src.metrics import compute_dice, compute_iou


def import_train_module():
    script_path = TOOLS_DIR / "14_train_roi_poly_seg.py"
    spec = importlib.util.spec_from_file_location("roi_poly_seg_train", script_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def choose_device(device_arg: str):
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def mask_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    pred = (pred_mask > 0).astype(np.float32)
    gt = (gt_mask > 0).astype(np.float32)
    return float(compute_iou(pred, gt))


def build_gt_targets(train_mod, img_path: Path, json_path: Path, category_to_id):
    data = load_isat_json(json_path)
    roi = data.get("ultrasound_rect")
    if not roi:
        raise ValueError(f"missing ultrasound_rect: {json_path}")

    x1, y1, x2, y2 = [int(roi[k]) for k in ("x1", "y1", "x2", "y2")]
    image = train_mod.Image.open(img_path).convert("RGB").crop((x1, y1, x2, y2))
    roi_w, roi_h = image.size

    items = []
    for obj in data.get("objects", []):
        category = obj.get("category", "")
        if category not in category_to_id:
            continue
        segmentation = obj.get("segmentation", [])
        if len(segmentation) < 3:
            continue

        shifted = []
        for px, py in segmentation:
            sx = min(max(float(px) - x1, 0.0), roi_w - 1)
            sy = min(max(float(py) - y1, 0.0), roi_h - 1)
            shifted.append([sx, sy])
        mask = train_mod.polygon_to_mask(shifted, roi_w, roi_h)
        if mask.sum() == 0:
            continue
        items.append({
            "category": category,
            "label_id": category_to_id[category] + 1,
            "mask": mask,
        })
    return image, items


def greedy_match(preds, gts, iou_thresh):
    matches = []
    used_pred = set()
    used_gt = set()
    candidate_pairs = []

    for pi, pred in enumerate(preds):
        for gi, gt in enumerate(gts):
            if pred["label_id"] != gt["label_id"]:
                continue
            iou = mask_iou(pred["mask"], gt["mask"])
            if iou >= iou_thresh:
                candidate_pairs.append((iou, pi, gi))

    candidate_pairs.sort(reverse=True, key=lambda x: x[0])
    for iou, pi, gi in candidate_pairs:
        if pi in used_pred or gi in used_gt:
            continue
        used_pred.add(pi)
        used_gt.add(gi)
        matches.append((pi, gi, iou))

    return matches, used_pred, used_gt


def main():
    parser = argparse.ArgumentParser(description="Evaluate ROI polygon segmentation")
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--score-thresh", type=float, default=0.5)
    parser.add_argument("--iou-thresh", type=float, default=0.5)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    args = parser.parse_args()

    if not CATEGORY_MAPPING_FILE.exists():
        raise FileNotFoundError("缺少类别映射，请先重建 shared_derived")

    train_mod = import_train_module()
    categories, category_to_id = load_category_mapping(CATEGORY_MAPPING_FILE)
    id_to_name = {idx + 1: name for idx, name in enumerate(categories)}
    device = choose_device(args.device)

    model = train_mod.create_model(num_classes=len(categories) + 1)
    ckpt = ROI_POLY_SEG_ARTIFACTS_DIR / "mask_rcnn" / "best.pth"
    if not ckpt.exists():
        raise FileNotFoundError("缺少 best.pth，请先训练")
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.to(device)
    model.eval()

    rel_paths = load_split_file(SPLITS_DIR / f"{args.split}.txt")
    if args.limit is not None:
        rel_paths = rel_paths[:args.limit]

    per_class = defaultdict(lambda: {
        "gt": 0,
        "pred": 0,
        "matched": 0,
        "dice_sum": 0.0,
        "iou_sum": 0.0,
        "false_positive": 0,
        "false_negative": 0,
    })

    total_gt = 0
    total_pred = 0
    total_matched = 0
    all_match_ious = []
    all_match_dices = []

    with torch.no_grad():
        for idx, rel_path in enumerate(rel_paths, start=1):
            img_path = CLEANED_DIR / rel_path
            json_path = img_path.with_suffix(".json")
            if not img_path.exists() or not json_path.exists():
                continue

            roi_image, gt_items = build_gt_targets(train_mod, img_path, json_path, category_to_id)
            image_tensor = train_mod.F.to_tensor(roi_image).to(device)
            output = model([image_tensor])[0]

            pred_items = []
            masks = output["masks"].detach().cpu().numpy()[:, 0]
            labels = output["labels"].detach().cpu().numpy()
            scores = output["scores"].detach().cpu().numpy()
            for mask, label, score in zip(masks, labels, scores):
                if float(score) < args.score_thresh:
                    continue
                pred_items.append({
                    "label_id": int(label),
                    "mask": (mask > 0.5).astype(np.uint8),
                })

            for gt in gt_items:
                name = gt["category"]
                per_class[name]["gt"] += 1
            for pred in pred_items:
                name = id_to_name.get(pred["label_id"], str(pred["label_id"]))
                per_class[name]["pred"] += 1

            matches, used_pred, used_gt = greedy_match(pred_items, gt_items, args.iou_thresh)
            for pi, gi, iou in matches:
                pred = pred_items[pi]
                gt = gt_items[gi]
                name = gt["category"]
                dice = float(compute_dice(pred["mask"].astype(np.float32), gt["mask"].astype(np.float32)))

                per_class[name]["matched"] += 1
                per_class[name]["dice_sum"] += dice
                per_class[name]["iou_sum"] += iou
                all_match_ious.append(iou)
                all_match_dices.append(dice)

            for pi, pred in enumerate(pred_items):
                if pi not in used_pred:
                    name = id_to_name.get(pred["label_id"], str(pred["label_id"]))
                    per_class[name]["false_positive"] += 1

            for gi, gt in enumerate(gt_items):
                if gi not in used_gt:
                    per_class[gt["category"]]["false_negative"] += 1

            total_gt += len(gt_items)
            total_pred += len(pred_items)
            total_matched += len(matches)

            if idx % 100 == 0:
                print(f"[{idx}/{len(rel_paths)}] processed")

    per_class_report = {}
    for name, stats in sorted(per_class.items()):
        matched = stats["matched"]
        per_class_report[name] = {
            "gt": stats["gt"],
            "pred": stats["pred"],
            "matched": matched,
            "precision": round(matched / stats["pred"], 4) if stats["pred"] else 0.0,
            "recall": round(matched / stats["gt"], 4) if stats["gt"] else 0.0,
            "mean_iou": round(stats["iou_sum"] / matched, 4) if matched else 0.0,
            "mean_dice": round(stats["dice_sum"] / matched, 4) if matched else 0.0,
            "false_positive": stats["false_positive"],
            "false_negative": stats["false_negative"],
        }

    summary = {
        "split": args.split,
        "score_thresh": args.score_thresh,
        "iou_thresh": args.iou_thresh,
        "device": str(device),
        "num_images": len(rel_paths),
        "total_gt": total_gt,
        "total_pred": total_pred,
        "total_matched": total_matched,
        "global_precision": round(total_matched / total_pred, 4) if total_pred else 0.0,
        "global_recall": round(total_matched / total_gt, 4) if total_gt else 0.0,
        "mean_iou_matched": round(float(np.mean(all_match_ious)), 4) if all_match_ious else 0.0,
        "mean_dice_matched": round(float(np.mean(all_match_dices)), 4) if all_match_dices else 0.0,
        "per_class": per_class_report,
    }

    ROI_POLY_SEG_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    thresh_tag = str(args.score_thresh).replace(".", "p")
    out_path = ROI_POLY_SEG_REPORTS_DIR / f"roi_poly_seg_eval_{args.split}_score_{thresh_tag}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps({
        "split": summary["split"],
        "num_images": summary["num_images"],
        "global_precision": summary["global_precision"],
        "global_recall": summary["global_recall"],
        "mean_iou_matched": summary["mean_iou_matched"],
        "mean_dice_matched": summary["mean_dice_matched"],
    }, ensure_ascii=False, indent=2))
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
