"""标签映射与清洗规则。

定义了：
1. LABEL_FIX_MAP: 错误标签 → 正确标签的直接映射
2. DISEASE_PREFIX_MAP: 疾病目录名 → 标签前缀
3. CATEGORY_ID_MAP: 清洗后的所有类别 → 整数 ID
4. DISEASE_CLASSES: 7 种疾病分类
"""

# ============================================================
# 1. 标签直接修复映射（错别字、格式不一致、缺前缀）
# ============================================================
LABEL_FIX_MAP = {
    # 错别字：内测 → 内侧
    "N内测副韧带": "N内侧副韧带",
    "N-内测半月板": "N内侧半月板",
    "内测半月板": "N内侧半月板",       # 同时补前缀
    # 拼写错误
    "SPA-股二头肌建": "SPA-股二头肌腱",  # 建 → 腱
    # 占位符/无效标签
    "滑膜那种": "滑膜囊肿",             # 需人工确认，暂映射为滑膜囊肿
    # 注意：以下需要根据疾病目录动态补前缀，在 ORPHAN_LABELS 中处理
}

# 需要根据所在疾病目录动态补前缀的标签
ORPHAN_LABELS = {
    "腘窝囊肿",         # 缺少疾病前缀
    "内测半月板",        # 已在 LABEL_FIX_MAP 中处理
    "滑膜囊肿",         # 修复后仍可能缺前缀
    "半膜肌腱",
    "滑膜那种",
}

# 疑似错误但无法确认的标签（保留并在报告中标记）
SUSPICIOUS_LABELS = {
    "SPA-髌下深囊炎",   # 疑似 OCR/拼写错误
    "SPA-鹅足腱滑囊炎", # 命名风格不一致（其他疾病不加"炎"）
    "滑膜那种",         # 原始值为占位符
}

# ============================================================
# 2. 疾病目录名 → 标签前缀映射
# ============================================================
DISEASE_PREFIX_MAP = {
    "正常": "N",
    "类风湿性关节炎": "RA",
    "骨性关节炎": "OA",
    "痛风性关节炎": "GA",
    "脊柱关节炎": "SPA-",
    "损伤": "损伤-",
    "滑膜囊肿": "滑膜囊肿-",
}

# ============================================================
# 3. 七种疾病分类
# ============================================================
DISEASE_CLASSES = [
    "正常",
    "类风湿性关节炎",
    "骨性关节炎",
    "痛风性关节炎",
    "脊柱关节炎",
    "损伤",
    "滑膜囊肿",
]

DISEASE_CLASS_TO_ID = {name: i for i, name in enumerate(DISEASE_CLASSES)}

# ============================================================
# 4. 从标签前缀提取疾病类型
# ============================================================
# 按前缀长度降序排列，确保先匹配更长的前缀
LABEL_PREFIX_TO_DISEASE = [
    ("滑膜囊肿-", "滑膜囊肿"),
    ("损伤-", "损伤"),
    ("SPA-", "脊柱关节炎"),
    ("RA", "类风湿性关节炎"),
    ("OA", "骨性关节炎"),
    ("GA", "痛风性关节炎"),
    ("N", "正常"),
]


def get_disease_from_label(label: str) -> str | None:
    """从标签名推断疾病类型。"""
    for prefix, disease in LABEL_PREFIX_TO_DISEASE:
        if label.startswith(prefix):
            return disease
    return None


def get_anatomy_from_label(label: str) -> str:
    """从标签名提取解剖结构名（去除疾病前缀）。"""
    for prefix, _ in LABEL_PREFIX_TO_DISEASE:
        if label.startswith(prefix):
            return label[len(prefix):]
    return label


def fix_label(label: str, disease_dir: str = "") -> str:
    """修复单个标签。

    Args:
        label: 原始标签
        disease_dir: 所在的疾病目录名（用于补前缀）

    Returns:
        修正后的标签
    """
    # 先做直接映射修复
    if label in LABEL_FIX_MAP:
        label = LABEL_FIX_MAP[label]

    # 检查是否是缺前缀的孤立标签，根据疾病目录补前缀
    if label in ORPHAN_LABELS and disease_dir:
        prefix = DISEASE_PREFIX_MAP.get(disease_dir, "")
        if prefix and not label.startswith(prefix):
            label = prefix + label

    return label


# ============================================================
# 5. 清洗后的完整类别列表（训练时使用）
#    运行 01_clean_labels.py 后会自动生成并更新
# ============================================================
# 占位符 —— 清洗脚本会收集实际出现的所有类别并写入此处
ALL_CATEGORIES = None  # 由 01_clean_labels.py 填充
CATEGORY_TO_ID = None  # 由 01_clean_labels.py 填充
