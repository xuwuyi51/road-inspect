"""边缘离线推理（M3 先落地 ONNX 运行时，M4 的 `rdinspect infer` CLI 直接复用它）。

为什么单独写而不是继续用 ultralytics：

* 边缘/车载环境只装 ``onnxruntime + numpy + pillow/opencv``，不装 torch（见 configs/edge.yaml）；
* 导出包里的 ``preprocess.json`` 是**唯一**的预处理契约，必须由本模块亲自实现并逐项验证，
  而不是依赖训练框架的默认行为（否则「工作站能跑、边缘跑出别的框」这类问题无法定位）；
* 预处理与 ultralytics 的 LetterBox 保持逐字节一致（stride 32 对齐、居中、pad 114、RGB、/255），
  这样 ONNX 与 .pt 的输出才能用 1e-3 的容差做验收。

约定的通道顺序：**RGB**。ultralytics 在吃 numpy 输入时按 BGR 解释，所以工作站侧
:class:`rdinspect.prelabel.detector.UltralyticsDetector` 会把 RGB 翻成 BGR；
而 ONNX 模型（torch 导出）期望的是 RGB，本模块直接喂 RGB，两者最终输入一致。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image

#: LetterBox 的填充值（与 ultralytics 默认一致）
PAD_VALUE = 114
#: YOLO 系列的下采样步长，LetterBox 会按它对齐缩放后的边长
STRIDE = 32


@dataclass(frozen=True)
class LetterboxMeta:
    """letterbox 的几何信息（用于把预测框映射回原图）。"""

    ratio: float
    pad_x: float
    pad_y: float
    resized: tuple[int, int]      # (w, h) 缩放后、未填充的尺寸
    original: tuple[int, int]     # (w, h) 原图尺寸
    target: tuple[int, int]       # (w, h) letterbox 目标尺寸

    def undo(self, x: float, y: float) -> tuple[float, float]:
        """把 letterbox 坐标还原到原图坐标。"""
        return (x - self.pad_x) / self.ratio, (y - self.pad_y) / self.ratio

    def as_dict(self) -> dict[str, Any]:
        return {"ratio": round(self.ratio, 8), "pad_x": self.pad_x, "pad_y": self.pad_y,
                "resized": list(self.resized), "original": list(self.original), "target": list(self.target)}


def letterbox(image: Image.Image, imgsz: int | tuple[int, int] = 640, *,
              auto: bool = False, stride: int = STRIDE, pad_value: int = PAD_VALUE,
              scaleup: bool = True) -> tuple[np.ndarray, LetterboxMeta]:
    """等比缩放 + 居中填充 → RGB NCHW float32 /255。

    与 ultralytics ``LetterBox(auto=False, center=True, scaleup=True)`` 的几何**逐项一致**：

    ``r = min(target_h/h, target_w/w)``；``new_unpad = (round(w*r), round(h*r))``；
    ``dw, dh = target - new_unpad``；仅 ``auto=True`` 时才按 stride 取模；
    居中时 ``top = round(dh/2 - 0.1)``、``left = round(dw/2 - 0.1)``（右/下补足余量）。
    """
    if isinstance(imgsz, int):
        target_w = target_h = int(imgsz)
    else:
        target_h, target_w = int(imgsz[0]), int(imgsz[1])
    rgb = image.convert("RGB")
    width, height = rgb.size
    ratio = min(target_h / height, target_w / width)
    if not scaleup:
        ratio = min(ratio, 1.0)
    new_w, new_h = round(width * ratio), round(height * ratio)
    pad_w, pad_h = target_w - new_w, target_h - new_h
    if auto:
        pad_w, pad_h = pad_w % stride, pad_h % stride
    left = round(pad_w / 2 - 0.1)
    top = round(pad_h / 2 - 0.1)
    canvas = np.full((target_h, target_w, 3), pad_value, dtype=np.uint8)
    if new_w > 0 and new_h > 0:
        try:
            import cv2  # noqa: PLC0415 - 与 ultralytics 使用同一插值实现，保证像素级一致

            resized = cv2.resize(np.asarray(rgb), (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        except ImportError:  # pragma: no cover - 无 cv2 的边缘最小环境
            resized = np.asarray(rgb.resize((new_w, new_h), Image.BILINEAR))
        canvas[top:top + new_h, left:left + new_w] = resized
    tensor = canvas.astype(np.float32).transpose(2, 0, 1)[None, ...] / 255.0
    meta = LetterboxMeta(ratio=ratio, pad_x=float(left), pad_y=float(top),
                         resized=(new_w, new_h), original=(width, height), target=(target_w, target_h))
    return tensor, meta


def decode_predictions(output: np.ndarray, meta: LetterboxMeta, *, conf: float,
                       class_names: Sequence[str]) -> list[dict[str, Any]]:
    """YOLO 检测头输出 (1, 4+nc, N) 或 (1, N, 4+nc) → 原图像素坐标的候选框（未做 NMS）。"""
    array = np.asarray(output)
    if array.ndim == 3:
        array = array[0]
    if array.ndim != 2:
        raise ValueError(f"无法解析的模型输出形状: {np.asarray(output).shape}")
    channels = 4 + len(class_names)
    if array.shape[0] == channels and array.shape[1] != channels:
        array = array.T                      # (4+nc, N) → (N, 4+nc)
    elif array.shape[1] != channels and array.shape[0] < array.shape[1]:
        array = array.T                      # 通道数未知时按"通道在前"处理
    boxes_xywh, scores = array[:, :4], array[:, 4:]
    if scores.size == 0:
        return []
    class_ids = scores.argmax(axis=1)
    confidences = scores[np.arange(scores.shape[0]), class_ids]
    keep = confidences >= float(conf)
    results: list[dict[str, Any]] = []
    for index in np.nonzero(keep)[0]:
        cx, cy, bw, bh = (float(value) for value in boxes_xywh[index])
        x1, y1 = meta.undo(cx - bw / 2.0, cy - bh / 2.0)
        x2, y2 = meta.undo(cx + bw / 2.0, cy + bh / 2.0)
        class_id = int(class_ids[index])
        name = class_names[class_id] if 0 <= class_id < len(class_names) else str(class_id)
        results.append({"class_index": class_id, "class_name": name,
                        "score": float(confidences[index]), "bbox": (x1, y1, x2, y2)})
    return results


def disable_telemetry(onnxruntime_module: Any) -> bool:
    """关闭 onnxruntime 遥测。

    不关掉的话，它会尝试持久化"遥测设备 ID"，在无法写入用户目录时**在进程 CWD 落一个
    名为 ``:memory:.ses`` 的文件**（本机实测），既是垃圾文件也暴露运行痕迹。失败不影响推理。
    """
    try:
        onnxruntime_module.disable_telemetry_events()
        return True
    except Exception:  # noqa: BLE001 - 老版本没有这个 API
        return False


class OnnxDetector:
    """导出包的 ONNX 检测器：与 :class:`~rdinspect.prelabel.detector.Detector` 协议兼容。"""

    def __init__(self, model_path: str | Path, *, class_names: Sequence[str] | None = None,
                 imgsz: int = 640, conf: float = 0.25, iou: float = 0.5, max_detections: int = 100,
                 providers: Sequence[str] | None = None, threads: int | None = None,
                 intra_op_threads: int | None = None) -> None:
        try:
            import onnxruntime  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - 边缘环境缺依赖
            raise RuntimeError("未安装 onnxruntime：pip install onnxruntime（边缘端见 docs/08-deployment.md）") from exc
        disable_telemetry(onnxruntime)
        self.weights = str(model_path)
        self.imgsz = int(imgsz)
        self.conf = float(conf)
        self.iou = float(iou)
        self.max_detections = int(max_detections)
        options = onnxruntime.SessionOptions()
        options.log_severity_level = 3
        # 边缘 CPU 预算有限：显式限制算子内线程数（0 = 交给 runtime 决定）
        resolved_threads = threads if threads is not None else intra_op_threads
        if resolved_threads:
            options.intra_op_num_threads = int(resolved_threads)
            options.inter_op_num_threads = 1
        available = onnxruntime.get_available_providers()
        chosen = [name for name in (providers or ["CPUExecutionProvider"]) if name in available]
        self._session = onnxruntime.InferenceSession(str(model_path), sess_options=options,
                                                     providers=chosen or ["CPUExecutionProvider"])
        input_meta = self._session.get_inputs()[0]
        self._input_name = input_meta.name
        static_sides = [dim for dim in list(getattr(input_meta, "shape", []) or [])[-2:]
                        if isinstance(dim, int) and dim > 0]
        if len(static_sides) == 2 and static_sides != [self.imgsz, self.imgsz]:
            raise RuntimeError(
                f"模型输入尺寸固定为 {static_sides}，与请求的 imgsz={self.imgsz} 不一致："
                f"请去掉 imgsz 覆盖（默认用导出包的 preprocess.json）或重新导出对应尺寸的包")
        names = list(class_names or [])
        self.names = {index: str(name) for index, name in enumerate(names)}
        self.providers = list(self._session.get_providers())
        self.threads = int(resolved_threads) if resolved_threads else None

    def predict_raw(self, image: Image.Image) -> tuple[np.ndarray, LetterboxMeta]:
        tensor, meta = letterbox(image, self.imgsz)
        return self._session.run(None, {self._input_name: tensor})[0], meta

    def predict(self, image: Image.Image) -> list[Any]:
        """返回 :class:`~rdinspect.prelabel.detector.Detection` 列表（像素坐标、含 NMS）。"""
        from ..prelabel.detector import Detection  # 局部导入避免循环
        from ..prelabel.sahi import nms

        output, meta = self.predict_raw(image)
        raw = decode_predictions(output, meta, conf=self.conf, class_names=list(self.names.values()))
        detections = [Detection(class_index=item["class_index"], class_name=item["class_name"],
                                score=item["score"], bbox=item["bbox"]) for item in raw]
        kept = nms(detections, self.iou)
        return kept[: self.max_detections]


def load_export_package(package_dir: str | Path) -> dict[str, Any]:
    """读取导出包（M4 的启动前校验：manifest 与模型/类别不匹配时拒绝启动）。"""
    root = Path(package_dir)
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"导出包缺少 manifest.json: {root}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    preprocess_path = root / "preprocess.json"
    preprocess = json.loads(preprocess_path.read_text(encoding="utf-8")) if preprocess_path.exists() else {}
    labels_path = root / "labels.txt"
    labels = [line.strip() for line in labels_path.read_text(encoding="utf-8").splitlines() if line.strip()] \
        if labels_path.exists() else []
    model_path = root / str(manifest.get("model_file") or "model.onnx")
    if not model_path.exists():
        raise FileNotFoundError(f"导出包缺少模型文件: {model_path}")
    return {"root": str(root), "manifest": manifest, "preprocess": preprocess, "labels": labels,
            "model_path": str(model_path)}


def package_detector(package_dir: str | Path, *, conf: float | None = None, iou: float | None = None) -> OnnxDetector:
    """按导出包里的 preprocess.json 构造检测器（参数以包为准，调用方只能收紧不能放松）。"""
    package = load_export_package(package_dir)
    preprocess = package["preprocess"] or {}
    postprocess = preprocess.get("postprocess") or {}
    size = preprocess.get("input_size") or [640, 640]
    return OnnxDetector(package["model_path"], class_names=package["labels"],
                        imgsz=int(size[0]) if isinstance(size, (list, tuple)) else int(size),
                        conf=float(conf if conf is not None else postprocess.get("conf", 0.25)),
                        iou=float(iou if iou is not None else postprocess.get("iou", 0.5)),
                        max_detections=int(postprocess.get("max_detections", 100)))
