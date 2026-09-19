"""边缘离线推理包（M3 先落地 ONNX 运行时，M4 补齐 `rdinspect infer` CLI）。

只用 ``onnxruntime + numpy + pillow/cv2``：不依赖 torch/ultralytics，可在无网车载设备上运行。
"""

from __future__ import annotations

__all__ = ["onnx_runtime"]
