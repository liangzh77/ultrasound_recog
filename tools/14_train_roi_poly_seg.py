"""ROI polygon segmentation baseline.

训练时直接读取原图和 JSON，在内存中按 `ultrasound_rect` 裁剪有效超声视野，
并将多边形标注映射到 ROI 局部坐标系后训练 Mask R-CNN。

用法:
    python tools/14_train_roi_poly_seg.py
    python tools/14_train_roi_poly_seg.py --epochs 30 --batch-size 2
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torchvision
from PIL import Image, ImageDraw
from torch import amp
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights, maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models import ResNet50_Weights
from torchvision.transforms import functional as F
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.common_paths import (
    CATEGORY_MAPPING_FILE,
    CLEANED_DIR,
    ROI_POLY_SEG_ARTIFACTS_DIR,
    ROI_POLY_SEG_CONFIGS_DIR,
    ROI_POLY_SEG_LOGS_DIR,
    ROI_POLY_SEG_REPORTS_DIR,
    SPLITS_DIR,
)
from src.data_utils import load_category_mapping, load_isat_json, load_split_file


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collate_fn(batch):
    return tuple(zip(*batch))


def can_use_torchvision_cuda_nms() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0]], device="cuda")
        scores = torch.tensor([1.0], device="cuda")
        torchvision.ops.nms(boxes, scores, 0.5)
        return True
    except Exception:
        return False


def polygon_to_mask(segmentation: List[List[float]], width: int, height: int) -> np.ndarray:
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    draw.polygon([(p[0], p[1]) for p in segmentation], outline=1, fill=1)
    return np.array(mask, dtype=np.uint8)


class RoiPolygonDataset(Dataset):
    def __init__(
        self,
        split_file: Path,
        data_dir: Path,
        category_to_id: Dict[str, int],
        training: bool = True,
        roi_max_size: Optional[int] = None,
    ) -> None:
        self.data_dir = data_dir
        self.category_to_id = category_to_id
        self.training = training
        self.roi_max_size = roi_max_size
        self.samples: List[Tuple[Path, Path]] = []

        for rel_path in load_split_file(split_file):
            img_path = data_dir / rel_path
            json_path = img_path.with_suffix(".json")
            if img_path.exists() and json_path.exists():
                self.samples.append((img_path, json_path))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        img_path, json_path = self.samples[index]
        data = load_isat_json(json_path)

        roi = data.get("ultrasound_rect")
        if not roi:
            raise ValueError(f"缺少 ultrasound_rect: {json_path}")

        x1, y1, x2, y2 = [int(roi[k]) for k in ("x1", "y1", "x2", "y2")]
        image = Image.open(img_path).convert("RGB")
        image = image.crop((x1, y1, x2, y2))
        roi_w, roi_h = image.size

        masks = []
        boxes = []
        labels = []
        areas = []

        for obj in data.get("objects", []):
            category = obj.get("category", "")
            if category not in self.category_to_id:
                continue

            segmentation = obj.get("segmentation", [])
            if len(segmentation) < 3:
                continue

            shifted = []
            for px, py in segmentation:
                sx = min(max(float(px) - x1, 0.0), roi_w - 1)
                sy = min(max(float(py) - y1, 0.0), roi_h - 1)
                shifted.append([sx, sy])

            mask = polygon_to_mask(shifted, roi_w, roi_h)
            if mask.sum() == 0:
                continue

            ys, xs = np.where(mask > 0)
            xmin, xmax = float(xs.min()), float(xs.max())
            ymin, ymax = float(ys.min()), float(ys.max())
            if xmax <= xmin or ymax <= ymin:
                continue

            masks.append(mask)
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.category_to_id[category] + 1)
            areas.append(float(mask.sum()))

        if not masks:
            empty_mask = np.zeros((1, roi_h, roi_w), dtype=np.uint8)
            target = {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros((0,), dtype=torch.int64),
                "masks": torch.from_numpy(empty_mask[:0]),
                "image_id": torch.tensor([index]),
                "area": torch.zeros((0,), dtype=torch.float32),
                "iscrowd": torch.zeros((0,), dtype=torch.int64),
            }
        else:
            target = {
                "boxes": torch.tensor(boxes, dtype=torch.float32),
                "labels": torch.tensor(labels, dtype=torch.int64),
                "masks": torch.from_numpy(np.stack(masks)).to(torch.uint8),
                "image_id": torch.tensor([index]),
                "area": torch.tensor(areas, dtype=torch.float32),
                "iscrowd": torch.zeros((len(masks),), dtype=torch.int64),
            }

        if self.roi_max_size is not None and max(roi_w, roi_h) > self.roi_max_size:
            scale = self.roi_max_size / max(roi_w, roi_h)
            new_w = max(1, int(round(roi_w * scale)))
            new_h = max(1, int(round(roi_h * scale)))
            image = image.resize((new_w, new_h), Image.BILINEAR)

            if target["boxes"].numel() > 0:
                target["boxes"][:, [0, 2]] *= scale
                target["boxes"][:, [1, 3]] *= scale

                resized_masks = []
                for mask in target["masks"].numpy():
                    resized = Image.fromarray(mask).resize((new_w, new_h), Image.NEAREST)
                    resized_masks.append(np.array(resized, dtype=np.uint8))
                target["masks"] = torch.from_numpy(np.stack(resized_masks)).to(torch.uint8)
                target["area"] = target["area"] * (scale * scale)

            roi_w, roi_h = new_w, new_h

        if self.training and random.random() < 0.5:
            image = F.hflip(image)
            if target["boxes"].numel() > 0:
                boxes_t = target["boxes"].clone()
                boxes_t[:, [0, 2]] = roi_w - boxes_t[:, [2, 0]]
                target["boxes"] = boxes_t
                target["masks"] = torch.flip(target["masks"], dims=[2])

        image_tensor = F.to_tensor(image)
        return image_tensor, target


def create_model(
    num_classes: int,
    pretrained_backbone: bool = False,
    trainable_backbone_layers: int = 3,
):
    weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained_backbone else None
    weights_backbone = ResNet50_Weights.DEFAULT if pretrained_backbone else None
    kwargs = {
        "weights": weights,
        "weights_backbone": weights_backbone,
    }
    if pretrained_backbone:
        kwargs["trainable_backbone_layers"] = trainable_backbone_layers
    model = maskrcnn_resnet50_fpn(**kwargs)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask,
        hidden_layer,
        num_classes,
    )
    return model


def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
    epoch: int,
    use_amp: bool = False,
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    model.train()
    running = {"loss": 0.0, "loss_classifier": 0.0, "loss_box_reg": 0.0, "loss_mask": 0.0, "loss_objectness": 0.0, "loss_rpn_box_reg": 0.0}
    scaler = amp.GradScaler("cuda", enabled=use_amp and device.type == "cuda")

    steps = 0
    for images, targets in tqdm(loader, desc=f"Epoch {epoch} train", leave=False):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in target.items()} for target in targets]

        with amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running["loss"] += float(loss.item())
        for key in running:
            if key != "loss" and key in loss_dict:
                running[key] += float(loss_dict[key].item())
        steps += 1
        if max_batches is not None and steps >= max_batches:
            break

    for key in running:
        running[key] /= max(steps, 1)
    return running


@torch.no_grad()
def validate_one_epoch(
    model,
    loader,
    device,
    epoch: int,
    use_amp: bool = False,
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    model.train()
    running = {"val_loss": 0.0}
    steps = 0

    for images, targets in tqdm(loader, desc=f"Epoch {epoch} val", leave=False):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in target.items()} for target in targets]
        with amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
            loss_dict = model(images, targets)
            running["val_loss"] += float(sum(loss_dict.values()).item())
        steps += 1
        if max_batches is not None and steps >= max_batches:
            break

    running["val_loss"] /= max(steps, 1)
    return running


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="ROI polygon segmentation baseline")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--pretrained-backbone", action="store_true")
    parser.add_argument("--roi-max-size", type=int, default=512)
    parser.add_argument("--trainable-backbone-layers", type=int, default=2)
    parser.add_argument("--amp", action="store_true", help="在 CUDA 上启用自动混合精度")
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    args = parser.parse_args()

    if not CATEGORY_MAPPING_FILE.exists():
        raise FileNotFoundError("缺少类别映射，请先运行 01_clean_labels.py")

    set_seed(args.seed)
    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("指定了 --device cuda，但当前环境不可用 CUDA")
        if not can_use_torchvision_cuda_nms():
            raise RuntimeError(
                "当前 torchvision 环境不支持 CUDA NMS，Mask R-CNN 无法在 GPU 上运行。"
                "请改用 --device cpu，或安装与当前 PyTorch 匹配的 torchvision CUDA 版本。"
            )
        device = torch.device("cuda")
    else:
        if torch.cuda.is_available() and can_use_torchvision_cuda_nms():
            device = torch.device("cuda")
        else:
            if torch.cuda.is_available():
                print("警告: 检测到 CUDA 可用，但 torchvision CUDA NMS 不可用，自动回退到 CPU。")
            device = torch.device("cpu")

    use_amp = bool(args.amp and device.type == "cuda")
    if args.amp and device.type != "cuda":
        print("提示: 当前不是 CUDA 设备，已忽略 --amp。")

    print(f"Device: {device}")

    categories, category_to_id = load_category_mapping(CATEGORY_MAPPING_FILE)
    num_classes = len(categories) + 1

    train_ds = RoiPolygonDataset(
        SPLITS_DIR / "train.txt",
        CLEANED_DIR,
        category_to_id,
        training=True,
        roi_max_size=args.roi_max_size,
    )
    val_ds = RoiPolygonDataset(
        SPLITS_DIR / "val.txt",
        CLEANED_DIR,
        category_to_id,
        training=False,
        roi_max_size=args.roi_max_size,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    model = create_model(
        num_classes,
        pretrained_backbone=args.pretrained_backbone,
        trainable_backbone_layers=args.trainable_backbone_layers,
    ).to(device)
    if args.resume:
        model.load_state_dict(torch.load(args.resume, map_location=device))

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = None
    if args.epochs > 1:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    run_dir = ROI_POLY_SEG_ARTIFACTS_DIR / "mask_rcnn"
    run_dir.mkdir(parents=True, exist_ok=True)
    ROI_POLY_SEG_LOGS_DIR.mkdir(parents=True, exist_ok=True)
    ROI_POLY_SEG_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ROI_POLY_SEG_CONFIGS_DIR.mkdir(parents=True, exist_ok=True)

    config = {
        "model": "maskrcnn_resnet50_fpn",
        "num_classes": len(categories),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "device": str(device),
        "patience": args.patience,
        "pretrained_backbone": args.pretrained_backbone,
        "roi_max_size": args.roi_max_size,
        "trainable_backbone_layers": args.trainable_backbone_layers,
        "amp": use_amp,
        "max_train_batches": args.max_train_batches,
        "max_val_batches": args.max_val_batches,
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "storage_policy": {"save_cropped_images": False},
    }
    save_json(ROI_POLY_SEG_CONFIGS_DIR / "mask_rcnn_baseline.json", config)

    history_path = ROI_POLY_SEG_LOGS_DIR / "train_history.json"
    history = []
    if args.resume and history_path.exists():
        with open(history_path, "r", encoding="utf-8") as f:
            history = json.load(f).get("history", [])
    best_val = float("inf")
    if history:
        best_val = min(item.get("val_loss", float("inf")) for item in history)
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        started = time.time()
        train_stats = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            epoch,
            use_amp=use_amp,
            max_batches=args.max_train_batches,
        )
        val_stats = validate_one_epoch(
            model,
            val_loader,
            device,
            epoch,
            use_amp=use_amp,
            max_batches=args.max_val_batches,
        )
        lr_now = optimizer.param_groups[0]["lr"]
        if scheduler is not None:
            scheduler.step()

        record = {
            "epoch": epoch,
            "lr": lr_now,
            "seconds": round(time.time() - started, 2),
            **train_stats,
            **val_stats,
        }
        history.append(record)
        save_json(history_path, {"history": history})

        torch.save(model.state_dict(), run_dir / "last.pth")
        if record["val_loss"] < best_val:
            best_val = record["val_loss"]
            patience_counter = 0
            torch.save(model.state_dict(), run_dir / "best.pth")
        else:
            patience_counter += 1

        print(
            f"Epoch {epoch}/{args.epochs} "
            f"loss={record['loss']:.4f} "
            f"val_loss={record['val_loss']:.4f} "
            f"lr={record['lr']:.6f}"
        )

        if patience_counter >= args.patience:
            print(f"Early stopping: {args.patience} epochs without val improvement")
            break

    summary = {
        "best_val_loss": best_val,
        "best_checkpoint": str(run_dir / "best.pth"),
        "last_checkpoint": str(run_dir / "last.pth"),
    }
    save_json(ROI_POLY_SEG_REPORTS_DIR / "train_summary.json", summary)


if __name__ == "__main__":
    main()
