"""ONNX 导出与导出包（M3）：模型 + labels + preprocess + manifest + 一致性验收。

导出包是**边缘端唯一需要的产物**（M4 的 `rdinspect infer` 直接读它）：

```
exports/<name>-<version>/
├── model.onnx          # opset 17，可选 dynamic batch
├── labels.txt          # 类别 code，行号 = 模型类别下标（顺序与数据集冻结顺序一致）
├── preprocess.json     # 预处理/后处理契约（letterbox 参数、归一化、NMS 阈值）
├── manifest.json       # 溯源：数据集清单哈希、训练/导出 run、门禁结论、权重 sha256
├── parity.json         # ONNX vs .pt 的一致性验收结果
└── README.md           # 边缘端用法（10 行可跑）
```

一致性验收（roadmap 验收项）：同尺寸同阈值下，ONNX 与 .pt 的**逐框归一化坐标**最大绝对误差 ≤
``export.tolerance``（默认 1e-3），且不出现单侧独有框。验收不通过时导出仍会落盘（便于排查），
但 ``parity.passed=false`` 会写入 manifest，并且不会更新模型的 ``onnx_path``。
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from PIL import Image

from ..config import Config
from ..errors import ConflictError, NotFoundError
from ..prelabel.detector import (Detection, DetectorUnavailable, ensure_runtime_env,  # noqa: F401
                                 ml_available, to_normalized_bbox)
from ..storage.files import sha256_file
from ..storage.repo import Repo
from . import matching
from .gate import require_promotable
from .runner import class_names_for, resolve_train_device, slugify

EXPORT_VERSION = "0.1.0"
#: 导出包格式版本：边缘端据此拒绝"比运行时更新"的包（见 edge/package.py）
EXPORT_SCHEMA_VERSION = 1


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def export_onnx(config: Config, repo: Repo, model_id: int, *, opset: int | None = None,
                dynamic_batch: bool | None = None, simplify: bool | None = None,
                half: bool | None = None, imgsz: int | None = None, device: str | None = None,
                verify: bool = True, tolerance: float | None = None, parity_images: int = 8,
                actor: str = "system") -> dict[str, Any]:
    """导出 ONNX + 组装导出包 + （默认）做一致性验收。"""
    row = require_promotable(config, repo, model_id)
    if not ml_available():
        # 依赖缺失属于"环境不可用"（501 / CLI 退出码 3），不是"状态不允许"（409）
        raise DetectorUnavailable("未安装 ultralytics/torch，无法导出 ONNX（pip install -e '.[ml]'）")
    export_cfg = config.train.export
    resolved_opset = int(opset if opset is not None else export_cfg.opset)
    resolved_dynamic = export_cfg.dynamic_batch if dynamic_batch is None else bool(dynamic_batch)
    resolved_simplify = export_cfg.simplify if simplify is None else bool(simplify)
    resolved_half = export_cfg.half if half is None else bool(half)
    train_params = (json.loads(row.get("metrics_json") or "{}") or {}).get("train_params") or {}
    resolved_imgsz = int(imgsz or train_params.get("imgsz") or config.train.imgsz)
    resolved_tolerance = float(tolerance if tolerance is not None else export_cfg.tolerance)
    weights = str(row["weights_path"])
    dataset = repo.get_dataset(dataset_id=row["dataset_id"]) if row.get("dataset_id") else None
    # 类别顺序校验前置：否则会在导出完 ONNX 之后才报错，白烧一次导出算力
    records = _class_records(repo, row)

    run = repo.create_run("export", dataset_id=row.get("dataset_id"), model_id=int(model_id),
                          config_json={"model": f"{row['name']}:{row['version']}", "weights": weights,
                                       "opset": resolved_opset, "dynamic_batch": resolved_dynamic,
                                       "simplify": resolved_simplify, "half": resolved_half,
                                       "imgsz": resolved_imgsz})
    run_id = int(run["id"])
    try:
        onnx_path = _run_export(config, weights, opset=resolved_opset, dynamic=resolved_dynamic,
                                simplify=resolved_simplify, half=resolved_half, imgsz=resolved_imgsz,
                                device=resolve_train_device(device or config.train.device))
        package = build_export_package(
            config, repo, row, onnx_path, opset=resolved_opset, dynamic_batch=resolved_dynamic,
            half=resolved_half, simplify=resolved_simplify, imgsz=resolved_imgsz, dataset=dataset,
            export_run_id=run_id, tolerance=resolved_tolerance, class_records=records,
        )
        parity = None
        if verify:
            images = _parity_images(config, repo, dataset, weights=weights, limit=parity_images)
            parity = verify_onnx_parity(config, repo, package_dir=package["dir"], weights=weights,
                                        images=images, tolerance=resolved_tolerance,
                                        imgsz=resolved_imgsz, conf=config.train.evaluate.conf,
                                        iou=config.train.evaluate.iou)
            package = _finalize_package(package["dir"], parity=parity)
    except Exception as exc:  # noqa: BLE001 - 失败落库便于排查
        repo.finish_run(run_id, status="failed", metrics={"model_id": int(model_id)},
                        error=f"{type(exc).__name__}: {exc}")
        raise

    if parity is None or parity["passed"]:
        repo.update_model_version(int(model_id), onnx_path=package["model_path"], actor=actor)
    repo.finish_run(run_id, status="succeeded",
                    metrics={"model_id": int(model_id), "package": package, "parity": parity,
                             "opset": resolved_opset, "dynamic_batch": resolved_dynamic})
    return {"model_id": int(model_id), "run_id": run_id, "package": package, "parity": parity,
            "registered": parity is None or parity["passed"]}


def _run_export(config: Config, weights: str, *, opset: int, dynamic: bool, simplify: bool, half: bool,
                imgsz: int, device: str) -> Path:
    ensure_runtime_env()
    from .compat import apply_compat_patches, preload_ultralytics  # noqa: PLC0415

    preload_ultralytics()
    apply_compat_patches()
    from ultralytics import YOLO  # noqa: PLC0415

    try:
        model = YOLO(weights)
    except Exception as exc:  # noqa: BLE001 - 权重损坏/不可读属于"模型不可用"
        raise DetectorUnavailable(f"加载权重失败 {weights}: {exc}") from exc
    kwargs: dict[str, Any] = {"format": "onnx", "opset": int(opset), "dynamic": bool(dynamic),
                              "simplify": bool(simplify), "imgsz": int(imgsz), "device": device}
    if half:  # 只有在真要 FP16 时才传，避免 ultralytics 的 half→quantize 弃用告警
        kwargs["half"] = True
    exported = model.export(**kwargs)
    path = Path(str(exported))
    if not path.exists():
        raise RuntimeError(f"ONNX 导出未产生文件: {exported}")
    return path


def _class_records(repo: Repo, model_row: dict[str, Any]) -> list[dict[str, Any]]:
    """模型类别 → 注册表类别（顺序必须与训练时一致，否则拒绝导出）。"""
    classes = repo.list_classes()
    order = class_names_for(repo)
    labels = (json.loads(model_row.get("labels_json") or "{}") or {}).get("names")
    if labels and list(labels) != order:
        raise ConflictError(
            "模型类别顺序与当前类别表不一致（可能新增/调整过类别）："
            f"模型 {list(labels)} ≠ 注册表 {order}；请用新类别顺序重新冻结数据集并训练（见 ADR-0005）")
    by_code = {cls["code"]: cls for cls in classes}
    records: list[dict[str, Any]] = []
    for code in order:
        cls = by_code.get(code, {})
        records.append({"code": code, "name_zh": cls.get("name_zh"), "name_en": cls.get("name_en"),
                        "is_crack": bool(cls.get("is_crack")), "order_index": cls.get("order_index")})
    return records


def build_export_package(config: Config, repo: Repo, model_row: dict[str, Any], onnx_path: Path, *,
                         opset: int, dynamic_batch: bool, half: bool, simplify: bool, imgsz: int,
                         dataset: dict[str, Any] | None, export_run_id: int, tolerance: float,
                         class_records: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """组装导出包目录（幂等：同名目录被覆盖重建）。"""
    records = class_records if class_records is not None else _class_records(repo, model_row)
    labels = [record["code"] for record in records]
    # 目录名带 imgsz：同一个 model version 可能按 320/640 各导一份，否则会互相覆盖
    package_dir = (config.exports_dir
                   / f"{slugify(str(model_row['name']))}-{slugify(str(model_row['version']))}-{int(imgsz)}")
    if package_dir.exists():
        shutil.rmtree(package_dir)
    package_dir.mkdir(parents=True, exist_ok=True)
    target_model = package_dir / "model.onnx"
    # ⚠️ 必须复制而不是硬链接：ultralytics 每次导出都写同一个 `<stem>.onnx`，
    # 硬链接会让先后导出的两个包（如 320 与 640）指向同一 inode——后一次导出会悄悄改掉前一个包的模型，
    # manifest 里的 sha256 随即失效（M4 的包校验正是在这里发现了它）。
    shutil.copy2(onnx_path, target_model)
    (package_dir / "labels.txt").write_text("\n".join(labels) + "\n", encoding="utf-8")
    preprocess = {
        "input_size": [imgsz, imgsz],
        "channel_order": "RGB",
        "layout": "NCHW",
        "dtype": "float32",
        "scale": 1.0 / 255.0,
        "mean": [0.0, 0.0, 0.0],
        "std": [1.0, 1.0, 1.0],
        "resize": {"mode": "letterbox", "keep_aspect": True, "center": True,
                   "pad_value": 114, "interpolation": "linear", "scaleup": True, "align_stride": None},
        "postprocess": {"head": "yolo_detect_raw", "output_layout": "[1, 4+nc, N]",
                        "box_format": "cxcywh", "conf": config.train.evaluate.conf,
                        "iou": config.train.evaluate.iou, "max_detections": 100,
                        "nms": "class_aware_greedy"},
    }
    (package_dir / "preprocess.json").write_text(
        json.dumps(preprocess, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = {
        "name": model_row["name"], "version": model_row["version"], "task": model_row.get("task", "detection"),
        "created_at": _now(), "rdinspect_version": EXPORT_VERSION,
        "schema_version": EXPORT_SCHEMA_VERSION,
        "model_file": "model.onnx", "model_sha256": sha256_file(target_model),
        "model_size_bytes": target_model.stat().st_size,
        "weights_path": model_row.get("weights_path"), "weights_sha256": model_row.get("sha256"),
        "opset": int(opset), "dynamic_batch": bool(dynamic_batch), "half": bool(half),
        "simplify": bool(simplify), "imgsz": int(imgsz),
        "nc": len(labels), "labels": labels, "classes": records,
        "dataset": ({"name": dataset["name"], "manifest_hash": dataset.get("manifest_hash"),
                     "id": int(dataset["id"])} if dataset else None),
        "train_run_id": model_row.get("run_id"), "export_run_id": int(export_run_id),
        "gate": (json.loads(model_row.get("gate_json") or "{}") or None),
        "metrics": {"map50": (json.loads(model_row.get("metrics_json") or "{}").get("evaluation") or {})
                    .get("ultralytics", {}).get("map50"),
                    "status": model_row.get("status")},
        "parity_tolerance": float(tolerance),
    }
    (package_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                                               encoding="utf-8")
    (package_dir / "README.md").write_text(_readme(manifest, config), encoding="utf-8")
    return {"dir": str(package_dir), "model_path": str(target_model), "labels": labels,
            "manifest_path": str(package_dir / "manifest.json"),
            "preprocess_path": str(package_dir / "preprocess.json"),
            # 键集保持稳定：verify=false 时也返回这些键（值为 None），调用方不必分支
            "parity_path": None, "manifest": manifest,
            "model_sha256": manifest["model_sha256"], "model_size_bytes": manifest["model_size_bytes"]}


def _readme(manifest: dict[str, Any], config: Config) -> str:
    return f"""# 导出包 · {manifest['name']}:{manifest['version']}

- 类别（行号 = 类别下标）：{', '.join(manifest['labels'])}
- 输入：{manifest['imgsz']}×{manifest['imgsz']}，RGB，letterbox 居中填充 114，/255（见 preprocess.json）
- 数据集清单哈希：{(manifest.get('dataset') or {}).get('manifest_hash')}
- 模型 sha256：`{manifest['model_sha256']}`

## 用法（边缘端只需 onnxruntime + numpy + pillow）

```python
from rdinspect.edge.onnx_runtime import package_detector
detector = package_detector(r"{config.exports_dir}/{manifest['name']}-{manifest['version']}")
boxes = detector.predict(image)         # image: PIL.Image；返回像素坐标 Detection
```

M4 会在此之上提供 `rdinspect infer --package <dir> --input <dir|video>`，并输出 JSONL/CSV。
"""


def _finalize_package(package_dir: str | Path, *, parity: dict[str, Any]) -> dict[str, Any]:
    """把一致性验收结果写进包（parity.json + manifest.parity）。"""
    root = Path(package_dir)
    (root / "parity.json").write_text(json.dumps(parity, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["parity"] = {key: parity[key] for key in
                          ("passed", "tolerance", "images", "max_bbox_delta", "max_raw_delta",
                           "max_workstation_delta", "unmatched_pt", "unmatched_onnx",
                           "checked_at", "method") if key in parity}
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"dir": str(root), "model_path": str(root / "model.onnx"),
            "labels": manifest.get("labels") or [], "manifest_path": str(manifest_path),
            "preprocess_path": str(root / "preprocess.json"),
            "parity_path": str(root / "parity.json"),
            "model_sha256": manifest.get("model_sha256"), "model_size_bytes": manifest.get("model_size_bytes"),
            "manifest": manifest}


def _parity_images(config: Config, repo: Repo, dataset: dict[str, Any] | None, *, weights: str,
                   limit: int) -> list[Path]:
    """挑选一致性验收用的图片：优先该模型的 test 划分，其次 val，最后训练数据目录。"""
    if dataset is not None and limit > 0:
        for split in ("test", "val", "train"):
            items = repo.dataset_items(int(dataset["id"]), split=split)
            if not items:
                continue
            paths = [config.abs_data_path(item["path"]) for item in items[:limit]]
            if paths:
                return paths
    raise NotFoundError(f"模型 {weights} 没有可用于一致性验收的数据集样本")


def verify_onnx_parity(config: Config, repo: Repo, *, package_dir: str | Path, weights: str,
                       images: Sequence[Path], tolerance: float, imgsz: int, conf: float,
                       iou: float) -> dict[str, Any]:
    """一致性验收：同一份 letterbox 输入下，ONNX 与 .pt 的输出与解码框必须一致。

    做法上刻意**不经过 ultralytics 的 predict**：导出包对外承诺的是「preprocess.json 描述的
    预处理 + ONNX 图」，所以验收也必须用包里的预处理（居中 letterbox）。步骤：

    1. 用包里的 imgsz 做 letterbox → 同一个张量分别喂 torch 与 onnxruntime；
    2. 比原始输出（letterbox 像素/logit 空间）的最大绝对误差；
    3. 用**同一套**解码 + NMS 得到两边的框，再比归一化坐标；
    4. 额外报告与工作站检测器（ultralytics predict，rect=False 已与包对齐）的差异。
    """
    from ..edge.onnx_runtime import (decode_predictions, letterbox, load_export_package,  # noqa: PLC0415
                                     package_detector)
    from ..prelabel.sahi import nms  # noqa: PLC0415
    from ..prelabel.detector import ensure_runtime_env, ml_available  # noqa: PLC0415

    if not ml_available():
        raise DetectorUnavailable("未安装 torch/ultralytics，无法做 ONNX↔.pt 一致性验收")
    ensure_runtime_env()
    from .compat import apply_compat_patches, preload_ultralytics  # noqa: PLC0415

    preload_ultralytics()
    apply_compat_patches()
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415
    from ultralytics import YOLO  # noqa: PLC0415

    package = load_export_package(package_dir)
    labels = list(package["labels"])
    preprocess = package["preprocess"] or {}
    package_imgsz = int((preprocess.get("input_size") or [imgsz])[0])
    postprocess = preprocess.get("postprocess") or {}
    package_conf = float(postprocess.get("conf", conf))
    package_iou = float(postprocess.get("iou", iou))

    torch_model = YOLO(weights).model.eval()
    onnx_detector = package_detector(package_dir)
    from ..prelabel.detector import UltralyticsDetector  # noqa: PLC0415

    workstation = UltralyticsDetector(weights, device="cpu", imgsz=package_imgsz, conf=conf, iou=iou)

    per_image: list[dict[str, Any]] = []
    max_raw_delta = max_bbox_delta = max_workstation_delta = 0.0
    unmatched_onnx = unmatched_torch = unmatched_workstation = 0
    total_boxes = 0
    for path in images:
        with Image.open(path) as handle:
            image = handle.convert("RGB")
        width, height = image.size
        tensor, meta = letterbox(image, package_imgsz)
        with torch.no_grad():
            torch_out = torch_model(torch.from_numpy(tensor))[0].detach().cpu().numpy()
        onnx_out = onnx_detector._session.run(  # noqa: SLF001 - 直接拿原始输出做逐元素比较
            None, {onnx_detector._input_name: tensor})[0]  # noqa: SLF001
        raw_delta = float(np.abs(np.asarray(torch_out) - np.asarray(onnx_out)).max())

        def _decode(output: Any) -> list[dict[str, Any]]:
            raw = decode_predictions(output, meta, conf=package_conf, class_names=labels)
            boxes = [Detection(class_index=item["class_index"], class_name=item["class_name"],
                               score=item["score"], bbox=item["bbox"]) for item in raw]
            kept = nms(boxes, package_iou)[: int(postprocess.get("max_detections", 100))]
            return [{"class_code": det.class_name, "score": float(det.score),
                     "bbox": to_normalized_bbox(det.bbox, width, height)
                             or {"x1": 0.0, "y1": 0.0, "x2": 0.0, "y2": 0.0}} for det in kept]

        torch_preds, onnx_preds = _decode(torch_out), _decode(onnx_out)
        matched = matching.match_detections(torch_preds, onnx_preds, iou_thr=0.5)
        image_delta = 0.0
        for pair in matched["pairs"]:
            left, right = torch_preds[int(pair["pred_index"])], onnx_preds[int(pair["gt_index"])]
            for key in ("x1", "y1", "x2", "y2"):
                image_delta = max(image_delta, abs(float(left["bbox"][key]) - float(right["bbox"][key])))
        workstation_preds = [{"class_code": det.class_name, "score": float(det.score),
                              "bbox": to_normalized_bbox(det.bbox, width, height)
                                      or {"x1": 0.0, "y1": 0.0, "x2": 0.0, "y2": 0.0}}
                             for det in workstation.predict(image)]
        ws_match = matching.match_detections(onnx_preds, workstation_preds, iou_thr=0.5)
        ws_delta = 0.0
        for pair in ws_match["pairs"]:
            left, right = onnx_preds[int(pair["pred_index"])], workstation_preds[int(pair["gt_index"])]
            for key in ("x1", "y1", "x2", "y2"):
                ws_delta = max(ws_delta, abs(float(left["bbox"][key]) - float(right["bbox"][key])))

        max_raw_delta = max(max_raw_delta, raw_delta)
        max_bbox_delta = max(max_bbox_delta, image_delta)
        max_workstation_delta = max(max_workstation_delta, ws_delta)
        unmatched_onnx += len(matched["extra"])
        unmatched_torch += len(matched["missed"])
        unmatched_workstation += len(ws_match["missed"]) + len(ws_match["extra"])
        total_boxes += len(torch_preds)
        per_image.append({"image": str(path), "torch_boxes": len(torch_preds), "onnx_boxes": len(onnx_preds),
                          "matched": len(matched["pairs"]), "max_raw_delta": round(raw_delta, 8),
                          "max_bbox_delta": round(image_delta, 8),
                          "workstation_boxes": len(workstation_preds),
                          "max_workstation_delta": round(ws_delta, 8)})

    passed = bool(max_bbox_delta <= float(tolerance) and unmatched_onnx == 0 and unmatched_torch == 0)
    reason = None
    if not passed:
        reason = (f"最大坐标误差 {max_bbox_delta:.6f} > 容差 {tolerance}"
                  if max_bbox_delta > float(tolerance) else
                  f"存在单侧独有框（.pt 独有 {unmatched_torch} / ONNX 独有 {unmatched_onnx}）")
    return {
        "passed": passed, "tolerance": float(tolerance), "images": len(per_image),
        "imgsz": package_imgsz, "conf": package_conf, "iou": package_iou,
        "boxes": total_boxes, "matched": sum(item["matched"] for item in per_image),
        "unmatched_pt": unmatched_torch, "unmatched_onnx": unmatched_onnx,
        "max_bbox_delta": round(max_bbox_delta, 8), "max_raw_delta": round(max_raw_delta, 8),
        "max_workstation_delta": round(max_workstation_delta, 8),
        "unmatched_workstation": unmatched_workstation,
        "onnx_providers": list(getattr(onnx_detector, "providers", [])),
        "labels": labels, "per_image": per_image, "checked_at": _now(), "reason": reason,
        "method": "同 letterbox 张量：torch 前向 vs onnxruntime 前向（原始输出 + 同套解码/NMS）",
    }


