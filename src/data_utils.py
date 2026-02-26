"""数据加载与转换工具函数。"""

import json
from pathlib import Path
from typing import Any


def load_isat_json(json_path: str | Path) -> dict[str, Any]:
    """加载一个 ISAT 格式的标注文件。"""
    with open(json_path, encoding="utf-8") as f:
        return json.load(f)


def load_split_file(split_path: str | Path) -> list[str]:
    """加载拆分文件，返回相对路径列表。"""
    with open(split_path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def load_category_mapping(mapping_path: str | Path) -> tuple[list[str], dict[str, int]]:
    """加载类别映射文件，返回 (categories, category_to_id)。"""
    with open(mapping_path, encoding="utf-8") as f:
        data = json.load(f)
    return data["categories"], data["category_to_id"]


def polygon_to_flat(segmentation: list[list[float]]) -> list[float]:
    """将 ISAT 的 [[x1,y1],[x2,y2],...] 转为 COCO 的 [x1,y1,x2,y2,...] 格式。"""
    flat = []
    for point in segmentation:
        flat.extend(point)
    return flat


def polygon_area(segmentation: list[list[float]]) -> float:
    """用 Shoelace 公式计算多边形面积。"""
    n = len(segmentation)
    if n < 3:
        return 0.0
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += segmentation[i][0] * segmentation[j][1]
        area -= segmentation[j][0] * segmentation[i][1]
    return abs(area) / 2.0


def polygon_bbox(segmentation: list[list[float]]) -> list[float]:
    """计算多边形的 bbox [x, y, width, height]（COCO 格式）。"""
    xs = [p[0] for p in segmentation]
    ys = [p[1] for p in segmentation]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    return [x_min, y_min, x_max - x_min, y_max - y_min]
