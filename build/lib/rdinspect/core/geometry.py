"""几何换算：内部统一使用「归一化 bbox (x1,y1,x2,y2) ∈ [0,1]」。

三种交换格式（见 docs/format-samples/README.md 与 docs/03-data-model.md）：
  * YOLO   : `class cx cy w h`（归一化中心点 + 宽高）
  * COCO   : `bbox = [x, y, w, h]`（像素） + `category_id`
  * LabelMe: `points = [[x1,y1],[x2,y2]]`（像素）
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

BBox = dict[str, float]

EPS = 1e-9


class GeometryError(ValueError):
    """非法坐标（越界、零面积、顺序颠倒）。"""


def validate_bbox(x1: float, y1: float, x2: float, y2: float) -> BBox:
    if not all(isinstance(v, (int, float)) for v in (x1, y1, x2, y2)):
        raise GeometryError(f"坐标必须是数字: {(x1, y1, x2, y2)}")
    if not (0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0):
        raise GeometryError(f"非法 bbox（要求 0<=x1<x2<=1 且 0<=y1<y2<=1）: {(x1, y1, x2, y2)}")
    return {"x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2)}


def bbox_from_xywh_norm(cx: float, cy: float, w: float, h: float) -> BBox:
    """YOLO 归一化 (cx,cy,w,h) → 内部 bbox。"""
    return validate_bbox(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def bbox_to_xywh_norm(bbox: BBox) -> tuple[float, float, float, float]:
    """内部 bbox → YOLO 归一化 (cx,cy,w,h)。"""
    return ((bbox["x1"] + bbox["x2"]) / 2, (bbox["y1"] + bbox["y2"]) / 2,
            bbox["x2"] - bbox["x1"], bbox["y2"] - bbox["y1"])


def bbox_from_coco(bbox_px: Sequence[float], width: int, height: int) -> BBox:
    """COCO 像素 [x,y,w,h] → 内部归一化 bbox。"""
    x, y, w, h = (float(v) for v in bbox_px)
    if width <= 0 or height <= 0:
        raise GeometryError(f"图像尺寸非法: {width}x{height}")
    return validate_bbox(x / width, y / height, (x + w) / width, (y + h) / height)


def bbox_to_coco(bbox: BBox, width: int, height: int) -> list[float]:
    """内部归一化 bbox → COCO 像素 [x,y,w,h]（保留 2 位小数）。"""
    x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
    return [round(x1 * width, 2), round(y1 * height, 2),
            round((x2 - x1) * width, 2), round((y2 - y1) * height, 2)]


def bbox_from_labelme(points: Iterable[Sequence[float]], width: int, height: int) -> BBox:
    """LabelMe 像素点集（可为多边形）→ 外接矩形（内部归一化）。"""
    pts = [(float(p[0]), float(p[1])) for p in points]
    if len(pts) < 2:
        raise GeometryError("LabelMe shape 至少需要 2 个点")
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    if width <= 0 or height <= 0:
        raise GeometryError(f"图像尺寸非法: {width}x{height}")
    return validate_bbox(min(xs) / width, min(ys) / height, max(xs) / width, max(ys) / height)


def bbox_to_labelme(bbox: BBox, width: int, height: int) -> list[list[float]]:
    """内部归一化 bbox → LabelMe 像素点（左上、右下）。"""
    return [[round(bbox["x1"] * width, 2), round(bbox["y1"] * height, 2)],
            [round(bbox["x2"] * width, 2), round(bbox["y2"] * height, 2)]]


def bbox_area(bbox: BBox) -> float:
    return max(0.0, bbox["x2"] - bbox["x1"]) * max(0.0, bbox["y2"] - bbox["y1"])


def bbox_iou(a: BBox, b: BBox) -> float:
    """归一化坐标系下的 IoU（同图内可比较）。"""
    ix1, iy1 = max(a["x1"], b["x1"]), max(a["y1"], b["y1"])
    ix2, iy2 = min(a["x2"], b["x2"]), min(a["y2"], b["y2"])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = bbox_area(a) + bbox_area(b) - inter
    return inter / union if union > EPS else 0.0


def yolo_line(class_index: int, bbox: BBox, *, digits: int = 6) -> str:
    cx, cy, w, h = bbox_to_xywh_norm(bbox)
    return f"{int(class_index)} {cx:.{digits}f} {cy:.{digits}f} {w:.{digits}f} {h:.{digits}f}"


def parse_yolo_line(line: str) -> tuple[int, BBox]:
    parts = line.split()
    if len(parts) < 5:
        raise GeometryError(f"YOLO 行格式错误（需要 5 个字段）: {line!r}")
    class_index = int(float(parts[0]))
    cx, cy, w, h = (float(v) for v in parts[1:5])
    return class_index, bbox_from_xywh_norm(cx, cy, w, h)


def tiled_origins(width: int, height: int, tile: int, overlap: float) -> list[tuple[int, int, int, int]]:
    """按 tile 尺寸与重叠率计算切片窗口 (x, y, w, h)，覆盖整图且不越界。"""
    if tile <= 0:
        raise GeometryError("tile 必须为正整数")
    if not 0.0 <= overlap < 1.0:
        raise GeometryError("overlap 必须在 [0,1)")
    step = max(1, int(round(tile * (1 - overlap))))
    windows: list[tuple[int, int, int, int]] = []
    y = 0
    while True:
        h = min(tile, height - y)
        if h <= 0:
            break
        x = 0
        while True:
            w = min(tile, width - x)
            if w <= 0:
                break
            windows.append((x, y, w, h))
            if x + w >= width:
                break
            x += step
        if y + h >= height:
            break
        y += step
    return windows


def tile_to_orig(tile_bbox: BBox, window: tuple[int, int, int, int],
                 image_width: int, image_height: int) -> BBox:
    """切片内归一化 bbox → 原图归一化 bbox。"""
    x, y, w, h = window
    x1 = (x + tile_bbox["x1"] * w) / image_width
    y1 = (y + tile_bbox["y1"] * h) / image_height
    x2 = (x + tile_bbox["x2"] * w) / image_width
    y2 = (y + tile_bbox["y2"] * h) / image_height
    return validate_bbox(min(max(x1, 0.0), 1.0 - EPS), min(max(y1, 0.0), 1.0 - EPS),
                         min(max(x2, EPS), 1.0), min(max(y2, EPS), 1.0))


def class_index_map(classes: Sequence[dict[str, Any]]) -> dict[str, int]:
    """类别顺序（YOLO 索引）由 order_index 决定，名称是唯一键。"""
    ordered = sorted(classes, key=lambda c: (c["order_index"], c["code"]))
    return {cls["code"]: index for index, cls in enumerate(ordered)}


def quantize(value: float, digits: int = 6) -> float:
    return round(float(value) + 0.0, digits)
