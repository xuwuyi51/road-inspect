"""切片推理（SAHI 策略）与结果合并（ADR-0006）。

细裂缝/航拍小目标在整图缩放到 640 后会系统性漏检，因此：
  * 长边超过阈值时把原图切成有重叠的 tile，逐片推理，再把框平移回原图坐标；
  * 跨片重复目标用类别感知的 NMS 合并（同一目标被多个 tile 命中是常态）。
零第三方依赖：NMS 与几何全部用标准库 + numpy 之外的纯 Python 实现（numpy 仅在 IoU 批量计算时可选）。
"""

from __future__ import annotations

from dataclasses import replace

from PIL import Image

from ..core.geometry import tiled_origins
from .detector import Detection, Detector

#: 过小的尾部 tile 无信息量，直接跳过（避免大量无效推理）
MIN_TILE_SIDE_PX = 64
#: "auto" 模式的切片阈值：长边超过它才切片
AUTO_TILE_LONG_EDGE = 1920


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def nms(detections: list[Detection], iou_threshold: float = 0.5) -> list[Detection]:
    """类别感知 NMS：同类重叠超过阈值时保留分数更高的框。"""
    kept: list[Detection] = []
    for candidate in sorted(detections, key=lambda det: det.score, reverse=True):
        if any(other.class_index == candidate.class_index and iou(other.bbox, candidate.bbox) > iou_threshold
               for other in kept):
            continue
        kept.append(candidate)
    # 输出顺序稳定（便于确定性测试）：按坐标排序
    return sorted(kept, key=lambda det: (det.bbox[1], det.bbox[0], det.class_index))


def should_slice(image: Image.Image, tile_config) -> bool:
    """按 tile.enabled（True/False/"auto"）判断是否需要切片。"""
    enabled = getattr(tile_config, "enabled", False)
    if enabled is False:
        return False
    if enabled is True:
        return True
    return max(image.size) > AUTO_TILE_LONG_EDGE


def predict_sliced(image: Image.Image, detector: Detector, *, tile_config, min_tile_side: int = MIN_TILE_SIDE_PX
                   ) -> tuple[list[Detection], int]:
    """切片推理 + 合并。

    @returns (原图坐标系下的检测结果, 使用的切片数；未切片时为 1)
    """
    image = image.convert("RGB")
    if not should_slice(image, tile_config):
        return detector.predict(image), 1

    width, height = image.size
    windows = tiled_origins(width, height, int(getattr(tile_config, "size", 1024)),
                            float(getattr(tile_config, "overlap", 0.2)))
    collected: list[Detection] = []
    used = 0
    for (x, y, w, h) in windows:
        if w < min_tile_side or h < min_tile_side:
            continue
        crop = image.crop((x, y, x + w, y + h))
        used += 1
        for detection in detector.predict(crop):
            x1, y1, x2, y2 = detection.bbox
            collected.append(replace(detection, bbox=(x1 + x, y1 + y, x2 + x, y2 + y)))
    merged = nms(collected, float(getattr(tile_config, "merge_iou", 0.5)))
    return merged, max(1, used)
