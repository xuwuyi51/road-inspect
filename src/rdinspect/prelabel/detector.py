"""检测器抽象：把"模型如何推理"与"预标注流程"解耦（ADR-0003 的替换点）。

- `Detector` 协议只有 `predict(image) -> [Detection]`（像素坐标），便于用假实现做单测；
- `UltralyticsDetector` 是唯一的真实实现（惰性导入 ultralytics/torch）；
- 未安装 ML 依赖时抛 `DetectorUnavailable`，上层转换为 501，绝不影响人工标注流程。
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence

from PIL import Image


class DetectorUnavailable(RuntimeError):
    """ML 依赖缺失或模型加载失败（不是用户输入错误）。"""


@dataclass(frozen=True)
class Detection:
    """一次检测结果（像素坐标，原图坐标系）。"""

    class_index: int
    class_name: str
    score: float
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2


class Detector(Protocol):
    """预标注使用的检测器接口。"""

    names: dict[int, str]

    def predict(self, image: Image.Image) -> list[Detection]:  # pragma: no cover - 协议
        ...


def ensure_runtime_env(base_dir: str | Path | None = None) -> Path:
    """确保 ultralytics/matplotlib 的配置目录可写且不落在用户 home 之外。

    默认位置是 `<权重目录或 CWD>/.ultralytics`；应用层（config.ensure_dirs）会用 data_dir 覆盖它。
    这样即使有人直接构造 `UltralyticsDetector`（不经配置加载），也不会因写入 ~/.config 失败。
    """
    import os  # noqa: PLC0415

    base = Path(base_dir) if base_dir is not None else Path(os.environ.get("YOLO_CONFIG_DIR") or Path.cwd())
    if base.name in (".ultralytics", "ultralytics") and Path(os.environ.get("YOLO_CONFIG_DIR", "") or "") == str(base):
        config_dir = base
    else:
        config_dir = base if base.name == "ultralytics" else base / ".ultralytics"
    config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(config_dir))
    os.environ.setdefault("MPLCONFIGDIR", str(config_dir / "mpl"))
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    return config_dir


def ml_available() -> bool:
    """是否有可用的 ultralytics（不导入 torch，避免测试环境开销）。"""
    return importlib.util.find_spec("ultralytics") is not None


def resolve_device(prefer: str = "auto") -> str:
    """auto → 有 CUDA 用 cuda，否则 cpu。"""
    if prefer != "auto":
        return prefer
    try:
        import torch  # noqa: PLC0415

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


class UltralyticsDetector:
    """ultralytics YOLO 适配器（AGPL，内部自用；见 ADR-0003）。"""

    def __init__(self, weights: str | Path, *, device: str = "auto", imgsz: int = 640,
                 conf: float = 0.25, iou: float = 0.5, max_detections: int = 100) -> None:
        if not ml_available():
            raise DetectorUnavailable(
                "未安装 ultralytics/torch：请执行 .venv/bin/pip install -e '.[ml]' 或 "
                "pip install ultralytics（权重与依赖见 docs/05-model-plan.md）"
            )
        ensure_runtime_env(Path(weights).parent if str(weights) not in ("", ".") else None)
        try:
            from ultralytics import YOLO  # noqa: PLC0415
        except Exception as exc:  # pragma: no cover - 依赖损坏
            raise DetectorUnavailable(f"导入 ultralytics 失败: {exc}") from exc
        self.weights = str(weights)
        self.device = resolve_device(device)
        self.imgsz = int(imgsz)
        self.conf = float(conf)
        self.iou = float(iou)
        self.max_detections = int(max_detections)
        try:
            self._model = YOLO(self.weights)
        except Exception as exc:
            raise DetectorUnavailable(f"加载权重 {self.weights} 失败: {exc}") from exc
        names = getattr(self._model, "names", None) or {}
        self.names = {int(key): str(value) for key, value in dict(names).items()}

    def predict(self, image: Image.Image) -> list[Detection]:
        import numpy as np  # noqa: PLC0415

        results = self._model.predict(
            source=np.asarray(image.convert("RGB")),
            imgsz=self.imgsz, conf=self.conf, iou=self.iou, max_det=self.max_detections,
            device=self.device, verbose=False,
        )
        if not results:
            return []
        boxes = getattr(results[0], "boxes", None)
        if boxes is None:
            return []
        detections: list[Detection] = []
        xyxy = boxes.xyxy.tolist() if hasattr(boxes, "xyxy") else []
        classes = boxes.cls.tolist() if hasattr(boxes, "cls") else []
        confidences = boxes.conf.tolist() if hasattr(boxes, "conf") else []
        for index, box in enumerate(xyxy):
            class_index = int(classes[index]) if index < len(classes) else -1
            detections.append(Detection(
                class_index=class_index,
                class_name=self.names.get(class_index, str(class_index)),
                score=float(confidences[index]) if index < len(confidences) else 0.0,
                bbox=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
            ))
        return detections


def model_version_of(weights: str | Path) -> tuple[str, str]:
    """从权重路径推导 (name, version)：name = 文件名去扩展名，version = 8 位内容指纹。"""
    import hashlib  # noqa: PLC0415

    path = Path(weights)
    name = path.stem
    try:
        stat = path.stat()
        fingerprint = hashlib.sha256(f"{path.resolve()}:{stat.st_size}:{int(stat.st_mtime)}".encode()).hexdigest()[:8]
    except OSError:
        fingerprint = hashlib.sha256(str(path).encode()).hexdigest()[:8]
    return name, fingerprint


def map_class(class_name: str, aliases: dict[str, str], known_codes: Sequence[str]) -> str | None:
    """把检测器的类别名映射到本项目类别 code；无法映射返回 None（调用方计数丢弃）。"""
    key = str(class_name).strip().lower()
    if key in aliases and aliases[key] in known_codes:
        return aliases[key]
    for alias, target in aliases.items():
        if alias and alias in key and target in known_codes:
            return target
    return None


def to_normalized_bbox(bbox_px: tuple[float, float, float, float], width: int, height: int,
                       *, min_side_px: float = 2.0) -> dict[str, float] | None:
    """像素框 → 归一化框；过小或越界（裁剪后仍无效）返回 None。"""
    x1, y1, x2, y2 = bbox_px
    x1, x2 = sorted((max(0.0, min(float(width), x1)), max(0.0, min(float(width), x2))))
    y1, y2 = sorted((max(0.0, min(float(height), y1)), max(0.0, min(float(height), y2))))
    if (x2 - x1) < min_side_px or (y2 - y1) < min_side_px:
        return None
    nx1, ny1, nx2, ny2 = x1 / width, y1 / height, x2 / width, y2 / height
    if not (0.0 <= nx1 < nx2 <= 1.0 and 0.0 <= ny1 < ny2 <= 1.0):
        return None
    return {"x1": round(nx1, 6), "y1": round(ny1, 6), "x2": round(nx2, 6), "y2": round(ny2, 6)}


def detector_summary(detector: Any) -> dict[str, Any]:
    """给 runs.config_json 的检测器描述（不写入模型权重内容）。"""
    return {
        "class": type(detector).__name__,
        "weights": getattr(detector, "weights", None),
        "device": getattr(detector, "device", None),
        "imgsz": getattr(detector, "imgsz", None),
        "conf": getattr(detector, "conf", None),
        "iou": getattr(detector, "iou", None),
        "names": list(getattr(detector, "names", {}).values())[:20],
    }
