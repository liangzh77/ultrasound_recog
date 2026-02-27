"""方案 C: MedSAM 微调脚本。

冻结 image encoder，只训练 prompt encoder + mask decoder + 分类头。
encoder 使用 torch.no_grad() 前向传播，避免存储巨量激活张量。

前置条件:
    1. 下载 MedSAM 预训练权重: medsam_vit_b.pth
    2. pip install segment-anything monai

用法:
    python scripts/10_train_medsam.py --weights path/to/medsam_vit_b.pth
    python scripts/10_train_medsam.py --evaluate --weights path/to/medsam_vit_b.pth
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.label_mapping import DISEASE_CLASSES, DISEASE_CLASS_TO_ID, get_disease_from_label

MEDSAM_DIR = ROOT / "data" / "medsam"
EMBED_DIR = ROOT / "data" / "medsam_embeddings"
SAVE_DIR = ROOT / "runs" / "medsam"


class MedSAMDataset(Dataset):
    """MedSAM 微调数据集（加载原始图片）。"""

    def __init__(self, split: str, img_size: int = 1024):
        self.split_dir = MEDSAM_DIR / split
        self.img_size = img_size

        with open(self.split_dir / "bboxes.json", encoding="utf-8") as f:
            self.entries = json.load(f)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]

        # 加载图片
        img_path = self.split_dir / "images" / entry["image"]
        image = np.array(Image.open(img_path).convert("RGB"), dtype=np.float32)

        # 加载掩码
        mask_path = self.split_dir / "masks" / entry["mask"]
        mask = np.array(Image.open(mask_path), dtype=np.float32) / 255.0

        # 原始尺寸
        orig_h, orig_w = image.shape[:2]

        # Resize 到 1024x1024（MedSAM 输入尺寸）
        image = np.array(Image.fromarray(image.astype(np.uint8)).resize(
            (self.img_size, self.img_size), Image.BILINEAR
        ), dtype=np.float32)
        mask = np.array(Image.fromarray((mask * 255).astype(np.uint8)).resize(
            (256, 256), Image.NEAREST  # SAM 输出 256x256
        ), dtype=np.float32) / 255.0

        # MedSAM 使用 [0, 1] 归一化
        image = image / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)  # (3, H, W)
        mask = torch.from_numpy(mask).unsqueeze(0)  # (1, H, W)

        # Bbox prompt（缩放到 1024 尺度）
        bbox = entry["bbox"]  # [x_min, y_min, x_max, y_max]
        scale_x = self.img_size / orig_w
        scale_y = self.img_size / orig_h
        bbox_scaled = torch.tensor([
            bbox[0] * scale_x, bbox[1] * scale_y,
            bbox[2] * scale_x, bbox[3] * scale_y,
        ], dtype=torch.float32)

        # 疾病分类标签
        cat_name = entry["category"]
        disease = get_disease_from_label(cat_name)
        disease_id = DISEASE_CLASS_TO_ID.get(disease, 0)

        return image, mask, bbox_scaled, disease_id


class CachedMedSAMDataset(Dataset):
    """使用预计算 image embeddings 的数据集（速度更快）。"""

    def __init__(self, split: str, img_size: int = 1024):
        self.split_dir = MEDSAM_DIR / split
        self.embed_dir = EMBED_DIR / split
        self.img_size = img_size

        with open(self.split_dir / "bboxes.json", encoding="utf-8") as f:
            self.entries = json.load(f)

        # 加载图片尺寸缓存
        sizes_path = self.embed_dir / "image_sizes.json"
        with open(sizes_path, encoding="utf-8") as f:
            self.image_sizes = json.load(f)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]

        # 加载预计算 embedding
        embed_path = self.embed_dir / (Path(entry["image"]).stem + ".pt")
        image_embedding = torch.load(embed_path, weights_only=True).squeeze(0)

        # 加载掩码
        mask_path = self.split_dir / "masks" / entry["mask"]
        mask = np.array(Image.open(mask_path), dtype=np.float32) / 255.0
        mask = np.array(Image.fromarray((mask * 255).astype(np.uint8)).resize(
            (256, 256), Image.NEAREST
        ), dtype=np.float32) / 255.0
        mask = torch.from_numpy(mask).unsqueeze(0)

        # Bbox prompt（使用缓存的原始尺寸）
        orig_w, orig_h = self.image_sizes[entry["image"]]
        bbox = entry["bbox"]
        scale_x = self.img_size / orig_w
        scale_y = self.img_size / orig_h
        bbox_scaled = torch.tensor([
            bbox[0] * scale_x, bbox[1] * scale_y,
            bbox[2] * scale_x, bbox[3] * scale_y,
        ], dtype=torch.float32)

        # 疾病分类标签
        cat_name = entry["category"]
        disease = get_disease_from_label(cat_name)
        disease_id = DISEASE_CLASS_TO_ID.get(disease, 0)

        return image_embedding, mask, bbox_scaled, disease_id


class MedSAMWithClassifier(nn.Module):
    """MedSAM + 分类头的联合模型。"""

    def __init__(self, sam_model, num_classes: int = 7):
        super().__init__()
        self.sam = sam_model

        # 冻结 image encoder
        for param in self.sam.image_encoder.parameters():
            param.requires_grad = False

        # 分类头：从 image_encoder 输出做全局平均池化 → 分类
        encoder_dim = 256  # SAM ViT-B encoder output dim
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(encoder_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward_with_images(self, images, boxes):
        """从原始图片前向传播（encoder 用 no_grad 加速）。"""
        with torch.no_grad():
            image_embeddings = self.sam.image_encoder(images)
        # detach 断开计算图，分类头和 decoder 仍可反向传播
        image_embeddings = image_embeddings.detach().requires_grad_(True)
        return self._decode(image_embeddings, boxes)

    def forward_with_embeddings(self, image_embeddings, boxes):
        """从预计算 embeddings 前向传播。"""
        image_embeddings = image_embeddings.requires_grad_(True)
        return self._decode(image_embeddings, boxes)

    def _decode(self, image_embeddings, boxes):
        """Prompt encoder + mask decoder + classifier。"""
        batch_size = image_embeddings.shape[0]
        all_masks = []
        all_ious = []
        image_pe = self.sam.prompt_encoder.get_dense_pe()

        for i in range(batch_size):
            sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
                points=None,
                boxes=boxes[i:i+1],
                masks=None,
            )
            low_res_mask, iou_pred = self.sam.mask_decoder(
                image_embeddings=image_embeddings[i:i+1],
                image_pe=image_pe,
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )
            all_masks.append(low_res_mask)
            all_ious.append(iou_pred)

        low_res_masks = torch.cat(all_masks, dim=0)
        iou_predictions = torch.cat(all_ious, dim=0)

        # Classification from encoder features
        cls_logits = self.classifier(image_embeddings)

        return low_res_masks, iou_predictions, cls_logits


def dice_loss(pred, target, smooth=1e-5):
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1.0 - dice.mean()


def precompute_embeddings(args):
    """预计算所有图片的 image embeddings 并缓存到磁盘。"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    from segment_anything import sam_model_registry
    sam = sam_model_registry["vit_b"](checkpoint=args.weights)
    encoder = sam.image_encoder.to(device)
    encoder.eval()

    for split in ["train", "val"]:
        split_dir = MEDSAM_DIR / split
        out_dir = EMBED_DIR / split
        out_dir.mkdir(parents=True, exist_ok=True)

        with open(split_dir / "bboxes.json", encoding="utf-8") as f:
            entries = json.load(f)

        # 去重
        unique_images = sorted(set(e["image"] for e in entries))
        print(f"{split}: {len(unique_images)} unique images, {len(entries)} samples")

        # 保存图片原始尺寸
        image_sizes = {}
        done = 0

        for img_name in tqdm(unique_images, desc=f"Pre-computing {split}"):
            out_path = out_dir / (Path(img_name).stem + ".pt")

            img_path = split_dir / "images" / img_name
            img = Image.open(img_path).convert("RGB")
            image_sizes[img_name] = [img.width, img.height]

            if out_path.exists():
                done += 1
                continue

            image = np.array(img.resize((1024, 1024), Image.BILINEAR), dtype=np.float32)
            image = image / 255.0
            image_t = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(device)

            with torch.no_grad():
                embedding = encoder(image_t)

            torch.save(embedding.cpu().half(), out_path)  # float16 节省磁盘
            done += 1

        # 保存尺寸信息
        with open(out_dir / "image_sizes.json", "w", encoding="utf-8") as f:
            json.dump(image_sizes, f)

        print(f"  {split} done: {done} embeddings cached")

    print("All embeddings pre-computed!")


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 检查是否有预计算 embeddings
    use_cached = (EMBED_DIR / "train").exists() and (EMBED_DIR / "val").exists()

    # 加载 MedSAM
    from segment_anything import sam_model_registry
    sam = sam_model_registry["vit_b"](checkpoint=args.weights)

    model = MedSAMWithClassifier(sam, num_classes=len(DISEASE_CLASSES))
    model = model.to(device)

    # 数据加载
    if use_cached:
        print("使用预计算 embeddings 加速训练")
        train_ds = CachedMedSAMDataset("train")
        val_ds = CachedMedSAMDataset("val")
    else:
        print("未找到预计算 embeddings，使用原始图片（较慢）")
        print("提示: 运行 --precompute 先预计算 embeddings")
        train_ds = MedSAMDataset("train")
        val_ds = MedSAMDataset("val")

    print(f"训练样本: {len(train_ds)}, 验证样本: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=2)

    # 优化器（只训练 decoder + classifier）
    seg_params = list(model.sam.mask_decoder.parameters()) + list(model.sam.prompt_encoder.parameters())
    cls_params = list(model.classifier.parameters())

    optimizer = torch.optim.AdamW([
        {"params": seg_params, "lr": args.lr},
        {"params": cls_params, "lr": args.lr * 10},
    ], weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    cls_criterion = nn.CrossEntropyLoss()

    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    best_dice = 0
    lambda_seg, lambda_cls = 1.0, 0.5

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_seg_loss = 0
        epoch_cls_loss = 0
        n_batches = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):
            if use_cached:
                embeddings, masks, bboxes, disease_ids = batch
                embeddings = embeddings.float().to(device)
            else:
                images, masks, bboxes, disease_ids = batch
                images = images.to(device)
            masks = masks.to(device)
            bboxes = bboxes.to(device)
            disease_ids = disease_ids.to(device)

            if use_cached:
                low_res_masks, _, cls_logits = model.forward_with_embeddings(embeddings, bboxes)
            else:
                low_res_masks, _, cls_logits = model.forward_with_images(images, bboxes)

            seg_loss = dice_loss(low_res_masks, masks) + F.binary_cross_entropy_with_logits(low_res_masks, masks)
            cls_loss = cls_criterion(cls_logits, disease_ids)
            total_loss = lambda_seg * seg_loss + lambda_cls * cls_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            epoch_seg_loss += seg_loss.item()
            epoch_cls_loss += cls_loss.item()
            n_batches += 1

        scheduler.step()

        # 验证
        model.eval()
        val_dice_sum = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                if use_cached:
                    embeddings, masks, bboxes, disease_ids = batch
                    embeddings = embeddings.float().to(device)
                else:
                    images, masks, bboxes, disease_ids = batch
                    images = images.to(device)
                masks = masks.to(device)
                bboxes = bboxes.to(device)
                disease_ids = disease_ids.to(device)

                if use_cached:
                    low_res_masks, _, cls_logits = model.forward_with_embeddings(embeddings, bboxes)
                else:
                    low_res_masks, _, cls_logits = model.forward_with_images(images, bboxes)

                # Dice
                pred_masks = (torch.sigmoid(low_res_masks) > 0.5).float()
                intersection = (pred_masks * masks).sum(dim=(2, 3))
                union = pred_masks.sum(dim=(2, 3)) + masks.sum(dim=(2, 3))
                dice = (2 * intersection + 1e-5) / (union + 1e-5)
                val_dice_sum += dice.sum().item()

                # Classification accuracy
                _, predicted = cls_logits.max(1)
                val_correct += predicted.eq(disease_ids).sum().item()
                val_total += disease_ids.size(0)

        avg_dice = val_dice_sum / max(val_total, 1)
        avg_acc = val_correct / max(val_total, 1)

        print(f"Epoch {epoch}/{args.epochs} | "
              f"Seg Loss: {epoch_seg_loss/n_batches:.4f} | "
              f"Cls Loss: {epoch_cls_loss/n_batches:.4f} | "
              f"Val Dice: {avg_dice:.4f} | Val Acc: {avg_acc:.4f}")

        if avg_dice > best_dice:
            best_dice = avg_dice
            torch.save(model.state_dict(), SAVE_DIR / "best.pth")
            print(f"  >> Best model saved (dice={best_dice:.4f})")

        torch.save(model.state_dict(), SAVE_DIR / "last.pth")

    print(f"\n训练完成! Best val dice: {best_dice:.4f}")


def main():
    parser = argparse.ArgumentParser(description="MedSAM 微调")
    parser.add_argument("--weights", type=str, required=True, help="MedSAM 预训练权重路径")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--precompute", action="store_true", help="预计算 image embeddings")
    args = parser.parse_args()

    if args.precompute:
        precompute_embeddings(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
