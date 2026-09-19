"""SAM 掩膜辅助（ADR-0001）：以检测框为提示生成裂缝掩膜，人工在标注台修正。

掩膜用途是派生指标（裂缝长度/平均宽度/面积），不参与首版检测训练。
未安装 ultralytics 时抛 DetectorUnavailable，由上层降级为「只出框」。
"""

from __future__ import annotations

import uuid
from pathlib import Path

import numpy as np
from PIL import Image

from ..config import Config
from .detector import DetectorUnavailable, ensure_runtime_env, ml_available, resolve_device


class SamMasker:
    """ultralytics SAM 适配器（box prompt → 二值掩膜）。"""

    def __init__(self, weights: str = "sam2.1_t.pt", *, device: str = "auto") -> None:
        if not ml_available():
            raise DetectorUnavailable("未安装 ultralytics/torch，无法使用 SAM 掩膜辅助")
        ensure_runtime_env(Path(weights).parent if weights else None)
        try:
            from ultralytics import SAM  # noqa: PLC0415
        except Exception as exc:  # pragma: no cover - 依赖损坏
            raise DetectorUnavailable(f"导入 ultralytics.SAM 失败: {exc}") from exc
        self.weights = weights
        self.device = resolve_device(device)
        try:
            self._model = SAM(weights)
        except Exception as exc:
            raise DetectorUnavailable(f"加载 SAM 权重 {weights} 失败: {exc}") from exc

    def mask_for_box(self, image: Image.Image, bbox_px: tuple[float, float, float, float]) -> np.ndarray:
        """返回与图像同尺寸的布尔掩膜（True = 目标）。"""
        results = self._model.predict(source=np.asarray(image.convert("RGB")), bboxes=[list(bbox_px)],
                                      device=self.device, verbose=False)
        if not results:
            return np.zeros((image.height, image.width), dtype=bool)
        masks = getattr(results[0], "masks", None)
        if masks is None or getattr(masks, "data", None) is None or len(masks.data) == 0:
            return np.zeros((image.height, image.width), dtype=bool)
        array = masks.data[0]
        mask = array.cpu().numpy() if hasattr(array, "cpu") else np.asarray(array)
        if mask.shape != (image.height, image.width):
            mask_image = Image.fromarray((mask > 0.5).astype(np.uint8) * 255)
            mask = np.asarray(mask_image.resize((image.width, image.height), Image.NEAREST)) > 127
        return mask > 0.5


def save_mask(config: Config, image_id: int, mask: np.ndarray) -> str:
    """把掩膜写成单通道 PNG，返回相对 data_dir 的路径。"""
    target_dir = config.masks_dir / str(image_id)
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{uuid.uuid4().hex[:12]}.png"
    Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(target, format="PNG", optimize=True)
    return config.rel_data_path(target)


def mask_metrics(mask: np.ndarray, *, pixel_size_m: float | None = None) -> dict[str, float | None]:
    """掩膜的派生指标：面积占比、等效长度/宽度（按像素估算；有标定则换算为物理单位）。"""
    total = int(mask.size)
    inside = int(mask.sum())
    if inside == 0:
        return {"area_ratio": 0.0, "length_px": 0.0, "width_px": 0.0,
                "length_m": None, "width_m": None}
    ys, xs = np.nonzero(mask)
    length_px = float(max(xs.max() - xs.min(), ys.max() - ys.min()) + 1)
    width_px = float(inside / length_px)
    return {
        "area_ratio": round(inside / total, 6),
        "length_px": round(length_px, 2),
        "width_px": round(width_px, 2),
        "length_m": None if pixel_size_m is None else round(length_px * pixel_size_m, 3),
        "width_m": None if pixel_size_m is None else round(width_px * pixel_size_m, 4),
    }


def mask_to_bbox(mask: np.ndarray, width: int, height: int) -> dict[str, float] | None:
    """掩膜外接矩形（归一化），用于和框标注对齐校对。"""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return {
        "x1": round(float(xs.min()) / width, 6), "y1": round(float(ys.min()) / height, 6),
        "x2": round(float(xs.max() + 1) / width, 6), "y2": round(float(ys.max() + 1) / height, 6),
    }


def load_mask(path: Path) -> np.ndarray:
    """读回掩膜（单通道 PNG → bool）。"""
    with Image.open(path) as image:
        return np.asarray(image.convert("L")) > 127
