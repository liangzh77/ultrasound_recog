"""方案 B 补充: 独立疾病分类器训练。

使用 EfficientNet-B4 进行 7 类疾病分类。
配合 nnU-Net 分割模型组成完整方案 B。

用法:
    python scripts/09_train_classifier.py
    python scripts/09_train_classifier.py --evaluate
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.label_mapping import DISEASE_CLASSES, DISEASE_CLASS_TO_ID

CLEANED_DIR = ROOT / "data" / "cleaned"
SPLITS_DIR = ROOT / "data" / "splits"
SAVE_DIR = ROOT / "runs" / "classifier"


class UltrasoundDataset(Dataset):
    """超声图像疾病分类数据集。"""

    def __init__(self, split_file: Path, data_dir: Path, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.samples = []  # [(img_path, disease_id)]

        with open(split_file, encoding="utf-8") as f:
            for line in f:
                rel_path = line.strip()
                if not rel_path:
                    continue
                # 从路径提取疾病类别（第一级目录）
                disease = Path(rel_path).parts[0]
                if disease in DISEASE_CLASS_TO_ID:
                    img_path = data_dir / rel_path
                    if img_path.exists():
                        self.samples.append((img_path, DISEASE_CLASS_TO_ID[disease]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

    def get_class_weights(self) -> torch.Tensor:
        """计算类别权重用于处理不平衡。"""
        counts = np.zeros(len(DISEASE_CLASSES))
        for _, label in self.samples:
            counts[label] += 1
        weights = 1.0 / (counts + 1e-6)
        weights = weights / weights.sum() * len(DISEASE_CLASSES)
        return torch.FloatTensor(weights)

    def get_sample_weights(self) -> list[float]:
        """计算每个样本的采样权重（用于 WeightedRandomSampler）。"""
        counts = np.zeros(len(DISEASE_CLASSES))
        for _, label in self.samples:
            counts[label] += 1
        class_weights = 1.0 / (counts + 1e-6)
        return [class_weights[label] for _, label in self.samples]


def get_transforms(is_train: bool):
    """获取数据增强变换。"""
    if is_train:
        return transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.1),
            transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])


def create_model(num_classes: int, device: torch.device):
    """创建 EfficientNet-B4 分类器。"""
    import timm
    model = timm.create_model("efficientnet_b4", pretrained=True, num_classes=num_classes)
    return model.to(device)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="  Train", leave=False):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    for images, labels in tqdm(loader, desc="  Eval", leave=False):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += images.size(0)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return total_loss / total, correct / total, all_preds, all_labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluate", action="store_true", help="仅在测试集上评估")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    num_classes = len(DISEASE_CLASSES)

    if args.evaluate:
        # 仅评估
        model = create_model(num_classes, device)
        ckpt = SAVE_DIR / "best.pth"
        model.load_state_dict(torch.load(ckpt, map_location=device))

        test_ds = UltrasoundDataset(SPLITS_DIR / "test.txt", CLEANED_DIR, get_transforms(False))
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, num_workers=2)
        criterion = nn.CrossEntropyLoss()

        loss, acc, preds, labels = evaluate(model, test_loader, criterion, device)
        print(f"Test Loss: {loss:.4f}, Accuracy: {acc:.4f}")

        # 打印分类报告
        from sklearn.metrics import classification_report
        print(classification_report(labels, preds, target_names=DISEASE_CLASSES))
        return

    # 训练
    train_ds = UltrasoundDataset(SPLITS_DIR / "train.txt", CLEANED_DIR, get_transforms(True))
    val_ds = UltrasoundDataset(SPLITS_DIR / "val.txt", CLEANED_DIR, get_transforms(False))

    print(f"训练样本: {len(train_ds)}, 验证样本: {len(val_ds)}")

    # 加权采样处理类别不平衡
    sample_weights = train_ds.get_sample_weights()
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=2)

    model = create_model(num_classes, device)
    class_weights = train_ds.get_class_weights().to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    best_acc = 0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        print(f"Epoch {epoch}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), SAVE_DIR / "best.pth")
            print(f"  >> Best model saved (acc={best_acc:.4f})")

        torch.save(model.state_dict(), SAVE_DIR / "last.pth")

    print(f"\n训练完成! Best val acc: {best_acc:.4f}")


if __name__ == "__main__":
    main()
