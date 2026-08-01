"""标签映射与清洗规则。

定义了：
1. LABEL_FIX_MAP: 错误标签 → 正确标签的直接映射
2. LABEL_PREFIX_TO_DISEASE: 识别历史标签中的疾病前缀
3. fix_label: 将像素区域标签规范化为疾病无关名称
4. DISEASE_CLASSES: 历史 7 种疾病分类
"""

# ============================================================
# 1. 标签直接修复映射（错别字、格式不一致、缺前缀）
# ============================================================
LABEL_FIX_MAP = {
    # 错别字：内测 → 内侧
    "N内测副韧带": "N内侧副韧带",
    "N-内测半月板": "N内侧半月板",
    "内测半月板": "内侧半月板",
    # 拼写错误
    "SPA-股二头肌建": "SPA-股二头肌腱",  # 建 → 腱
    "GA股二头肌建": "GA股二头肌腱",      # 建 → 腱
    "SPA-斌下肾囊炎": "SPA-髌下深囊炎",  # 客户增量数据重新带入的旧错别字
    # 占位符/无效标签
    "滑膜那种": "滑膜囊肿",             # 需人工确认，暂映射为滑膜囊肿
}

# 去除疾病前缀后仍需合并的同义写法。
REGION_NAME_FIX_MAP = {
    "髌骨前浅筋膜": "髌前浅筋膜",
}

# 疑似错误但无法确认的标签（保留并在报告中标记）
SUSPICIOUS_LABELS = {
    "滑膜那种",         # 原始值为占位符
}

# ============================================================
# 2. 疾病目录名 → 历史标签前缀映射
# ============================================================
# 仅用于识别旧数据来源。新的像素区域训练标签不再添加疾病前缀。
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
    """将单个像素区域标签规范化为疾病无关名称。

    Args:
        label: 原始标签
        disease_dir: 保留的兼容参数；主要诊断由目录表达，不写入区域标签

    Returns:
        修正错别字并去除疾病前缀后的区域名称
    """
    label = label.strip()

    # 先修复包含疾病前缀的历史错别字，再统一去除前缀。
    if label in LABEL_FIX_MAP:
        label = LABEL_FIX_MAP[label]

    label = get_anatomy_from_label(label).strip()
    return REGION_NAME_FIX_MAP.get(label, label)


# ============================================================
# 5. 清洗后的完整类别列表（训练时使用）
#    运行 01_clean_labels.py 后会自动生成并更新
# ============================================================
# 占位符 —— 清洗脚本会收集实际出现的所有类别并写入此处
ALL_CATEGORIES = None  # 由 01_clean_labels.py 填充
CATEGORY_TO_ID = None  # 由 01_clean_labels.py 填充
