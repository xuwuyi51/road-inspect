"""road-inspect：轻量级道路灾害巡查程序。

M1 范围：采集导入 → 人工标注 → 复核 → 数据集冻结与导出（YOLO/COCO/LabelMe）。
设计与决策见 docs/（架构、数据模型、标注流程、ADR）。
"""

__version__ = "0.1.0"

#: 内置类别顺序与 `db/schema.sql` 的初始数据保持一致（YOLO 索引由 order_index 决定）
DEFAULT_CLASSES = [
    ("longitudinal_crack", "纵向裂缝", "Longitudinal Crack", "#1f77b4", 1, 0),
    ("transverse_crack", "横向裂缝", "Transverse Crack", "#d62728", 1, 1),
    ("pothole", "坑洞", "Pothole", "#2ca02c", 0, 2),
    ("garbage", "垃圾", "Garbage", "#9467bd", 0, 3),
    ("alligator_crack", "网裂（预留）", "Alligator Crack", "#ff7f0e", 1, 4),
]
