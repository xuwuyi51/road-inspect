"""预标注编排：任务选择 → 切片推理 → 类别映射 → 候选落库 →（可选）SAM 掩膜 → 运行记录。

降级策略（docs/04-annotation-workflow.md §5.3）：
  * 未安装 ML 依赖 / 模型加载失败 → 抛 DetectorUnavailable，API 转 501，人工标注不受影响；
  * 单张图失败只记入 errors 并继续，任务状态与人工标注流程保持可用；
  * 同一 run_key 重复提交直接返回既有 run（幂等，避免重复烧算力）。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from PIL import Image

from ..config import Config
from ..storage.repo import Repo
from .detector import (
    Detection,
    Detector,
    DetectorUnavailable,
    UltralyticsDetector,
    detector_summary,
    map_class,
    ml_available,
    model_version_of,
    to_normalized_bbox,
)
from .sahi import predict_sliced
from .sam import SamMasker, mask_metrics, save_mask

#: 默认权重：先用通用 YOLO11n 打通链路，生产质量需换成道路病害训练权重（见 docs/05-model-plan.md）
DEFAULT_WEIGHTS = "yolo11n.pt"
CRACK_CLASSES = ("transverse_crack", "longitudinal_crack", "alligator_crack")


@dataclass
class PrelabelReport:
    run_id: int | None
    requested: int
    processed: int = 0
    candidates: int = 0
    masks: int = 0
    tiles: int = 0
    unmapped: int = 0
    unmapped_classes: dict[str, int] = field(default_factory=dict)
    skipped: int = 0
    errors: list[dict[str, str]] = field(default_factory=list)
    model: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "requested": self.requested,
            "processed": self.processed,
            "candidates": self.candidates,
            "masks": self.masks,
            "tiles": self.tiles,
            "unmapped": self.unmapped,
            "unmapped_classes": self.unmapped_classes,
            "skipped": self.skipped,
            "errors": self.errors[:20],
            "model": self.model,
        }


def _localize_weights(config: Config, weights: str) -> str:
    """裸文件名（如 yolo11n.pt）统一落到 data_dir/weights/，避免下载到进程 CWD。"""
    if not weights or "/" in weights or weights.startswith("~"):
        return weights
    existing = Path(weights)
    if existing.exists():
        return str(existing)
    return str(config.weights_dir / weights)


def resolve_weights(config: Config, repo: Repo, model: str | None = None) -> str:
    """权重解析顺序：显式参数 → 配置 → 现役 production 模型 → 默认 yolo11n（全部落到 weights_dir）。"""
    if model:
        return _localize_weights(config, model)
    if config.prelabel.model:
        return _localize_weights(config, config.prelabel.model)
    production = repo.production_model(task="detection")
    if production is not None and production.get("weights_path"):
        return str(production["weights_path"])
    return _localize_weights(config, DEFAULT_WEIGHTS)


def build_detector(config: Config, weights: str, *, device: str | None = None,
                   imgsz: int | None = None, conf: float | None = None, iou: float | None = None) -> Detector:
    """构造真实检测器（供 CLI/服务使用；测试注入假实现）。"""
    return UltralyticsDetector(
        weights,
        device=device or config.prelabel.device,
        imgsz=imgsz or config.prelabel.imgsz,
        conf=conf if conf is not None else config.prelabel.conf,
        iou=iou if iou is not None else config.prelabel.iou,
        max_detections=config.prelabel.max_detections,
    )


def _run_key(*, weights: str, conf: float, iou: float, status: str, task_ids: Sequence[int] | None,
             limit: int, sam: bool, sam_limit: int) -> str:
    """幂等键：SAM 开关与限额也参与，否则「同样的检测参数 + 打开 SAM」会被误判为重复运行。"""
    payload = json.dumps({"weights": weights, "conf": round(conf, 4), "iou": round(iou, 4),
                          "status": status, "task_ids": sorted(task_ids) if task_ids else None,
                          "limit": limit, "sam": bool(sam), "sam_limit": int(sam_limit)},
                         sort_keys=True)
    return "prelabel:" + hashlib.sha256(payload.encode()).hexdigest()[:24]


def _select_tasks(repo: Repo, *, status: str, limit: int,
                  task_ids: Sequence[int] | None) -> list[dict[str, Any]]:
    if task_ids:
        tasks = [task for task in (repo.get_task(task_id) for task_id in task_ids) if task is not None]
        return tasks[:limit] if limit else tasks
    return repo.list_tasks(status=status, limit=limit)


def prelabel_tasks(config: Config, repo: Repo, *, limit: int = 50, status: str = "pending",
                   task_ids: Sequence[int] | None = None, model: str | None = None,
                   conf: float | None = None, iou: float | None = None,
                   sam: bool | None = None, sam_limit: int = 2, actor: str = "cli",
                   detector: Detector | None = None, masker: SamMasker | None = None,
                   run_key: str | None = None) -> PrelabelReport:
    """批量预标注：把候选框写入任务，供标注台一键载入/逐条采纳或忽略。"""
    if not config.prelabel.enabled and detector is None:
        raise DetectorUnavailable("预标注已在配置中关闭（configs/prelabel.yaml: enabled=false）")

    weights = str(getattr(detector, "weights", None) or resolve_weights(config, repo, model))
    effective_conf = conf if conf is not None else float(getattr(detector, "conf", config.prelabel.conf))
    effective_iou = iou if iou is not None else float(getattr(detector, "iou", config.prelabel.iou))
    tasks = _select_tasks(repo, status=status, limit=limit, task_ids=task_ids)
    report = PrelabelReport(run_id=None, requested=len(tasks))

    if detector is None:
        detector = build_detector(config, weights, conf=effective_conf, iou=effective_iou)
    report.model = detector_summary(detector)

    known_codes = [cls["code"] for cls in repo.list_classes(active_only=True)]
    name, version = model_version_of(weights)
    model_row = repo.upsert_model_version(
        name, version, task="detection", status="candidate", weights_path=weights,
        labels_json={"names": list(getattr(detector, "names", {}).values()), "class_aliases": config.prelabel.class_aliases},
        actor=actor,
    )
    model_version_id = int(model_row["id"])

    use_sam_requested = bool(sam if sam is not None else config.prelabel.sam.enabled)
    effective_run_key = run_key or _run_key(weights=weights, conf=effective_conf, iou=effective_iou,
                                            status=status, task_ids=task_ids, limit=limit,
                                            sam=use_sam_requested, sam_limit=sam_limit)
    existing = repo.conn.execute("SELECT * FROM runs WHERE run_key = ?", (effective_run_key,)).fetchone()
    if existing is not None and existing["status"] == "succeeded":
        # 幂等：同一参数组合已成功跑过 → 返回既有结果，绝不重复烧算力
        stored = json.loads(existing["metrics_json"] or "{}")
        return PrelabelReport(
            run_id=int(existing["id"]), requested=len(tasks),
            processed=int(stored.get("processed", 0)), candidates=int(stored.get("candidates", 0)),
            masks=int(stored.get("masks", 0)), tiles=int(stored.get("tiles", 0)),
            unmapped=int(stored.get("unmapped", 0)),
            unmapped_classes=dict(stored.get("unmapped_classes", {})),
            skipped=int(stored.get("skipped", 0)),
            model=json.loads(existing["config_json"] or "{}").get("model", {}),
        )

    run = repo.create_run(
        "prelabel",
        run_key=effective_run_key,
        config_json={"weights": weights, "conf": effective_conf, "iou": effective_iou, "status": status,
                     "limit": limit, "sam": bool(sam if sam is not None else config.prelabel.sam.enabled),
                     "model_version_id": model_version_id, "model": report.model},
        model_id=model_version_id,
        log_path=None,
    )
    report.run_id = int(run["id"])

    use_sam = bool(sam if sam is not None else config.prelabel.sam.enabled)
    if use_sam and masker is None and ml_available():
        try:
            masker = SamMasker(_localize_weights(config, config.prelabel.sam.model),
                               device=config.prelabel.device)
        except DetectorUnavailable as exc:
            report.errors.append({"stage": "sam-init", "error": str(exc)[:200]})
            use_sam = False

    try:
        for task in tasks:
            task_id = int(task["id"])
            image_row = repo.get_image(int(task["image_id"]))
            if image_row is None:
                report.skipped += 1
                report.errors.append({"task": str(task_id), "error": "影像记录缺失"})
                continue
            image_path = config.abs_data_path(image_row["path"])
            if not image_path.exists():
                report.skipped += 1
                repo.set_prelabel_state(task_id, "failed")
                report.errors.append({"task": str(task_id), "error": "影像文件丢失"})
                continue
            try:
                with Image.open(image_path) as image:
                    image = image.convert("RGB")
                    detections, tiles = predict_sliced(image, detector, tile_config=config.prelabel.tile)
                    report.tiles += tiles
                    width, height = image.size
                    items: list[dict[str, Any]] = []
                    for detection in detections:
                        code = map_class(detection.class_name, config.prelabel.class_aliases, known_codes)
                        if code is None:
                            report.unmapped += 1
                            report.unmapped_classes[detection.class_name] = (
                                report.unmapped_classes.get(detection.class_name, 0) + 1)
                            continue
                        bbox = to_normalized_bbox(detection.bbox, width, height)
                        if bbox is None:
                            continue
                        items.append({"class_code": code, "bbox": bbox, "score": round(detection.score, 4),
                                      "detection": detection})
                    inserted = repo.add_model_candidates(
                        task_id, [{k: v for k, v in item.items() if k != "detection"} for item in items],
                        model_version_id=model_version_id)
                    report.candidates += inserted
                    report.processed += 1
                    if use_sam and masker is not None:
                        report.masks += _attach_masks(
                            config, repo, task_id=task_id, image=image, image_id=int(image_row["id"]),
                            items=items, masker=masker, limit=sam_limit, model_version_id=model_version_id)
            except DetectorUnavailable:
                raise
            except Exception as exc:  # 单张失败不影响整批
                report.skipped += 1
                repo.set_prelabel_state(task_id, "failed")
                report.errors.append({"task": str(task_id), "error": f"{type(exc).__name__}: {exc}"[:200]})
    except DetectorUnavailable:
        repo.finish_run(report.run_id, status="failed", error="ml-unavailable")
        raise
    except Exception as exc:
        repo.finish_run(report.run_id, status="failed", error=f"{type(exc).__name__}: {exc}"[:500])
        raise

    repo.finish_run(report.run_id, status="succeeded", metrics={
        "processed": report.processed, "candidates": report.candidates, "masks": report.masks,
        "tiles": report.tiles, "unmapped": report.unmapped,
        "unmapped_classes": report.unmapped_classes, "skipped": report.skipped,
        "errors": len(report.errors),
    })
    return report


def _attach_masks(config: Config, repo: Repo, *, task_id: int, image: Image.Image, image_id: int,
                  items: list[dict[str, Any]], masker: SamMasker, limit: int,
                  model_version_id: int) -> int:
    """为裂缝类候选生成掩膜（每张图最多 limit 个，避免 CPU 推理过慢）。"""
    made = 0
    for item in items:
        if made >= limit:
            break
        if item["class_code"] not in CRACK_CLASSES:
            continue
        width, height = image.size
        px = (item["bbox"]["x1"] * width, item["bbox"]["y1"] * height,
              item["bbox"]["x2"] * width, item["bbox"]["y2"] * height)
        if min(px[2] - px[0], px[3] - px[1]) < config.prelabel.sam.min_box_side_px:
            continue
        try:
            mask = masker.mask_for_box(image, px)
        except Exception:
            continue
        if not mask.any():
            continue
        rel_path = save_mask(config, image_id, mask)
        metrics = mask_metrics(mask)
        repo.add_mask_candidate(task_id=task_id, class_code=item["class_code"], bbox=item["bbox"],
                                mask_path=rel_path, metrics=metrics, model_version_id=model_version_id)
        made += 1
    return made


def attach_mask_for_annotation(config: Config, repo: Repo, annotation_id: int, *,
                               masker: SamMasker | None = None) -> dict[str, Any]:
    """标注台交互用：对某条框标注做 SAM 提示式掩膜（人工确认后才入库，source='human'）。"""
    row = repo.conn.execute("SELECT * FROM annotations WHERE id = ?", (annotation_id,)).fetchone()
    if row is None or row["deleted_at"] is not None:
        raise KeyError(f"标注 {annotation_id} 不存在")
    annotation = dict(row)
    if annotation["kind"] != "bbox" or annotation["bbox_x1"] is None:
        raise ValueError("只有框标注可以做掩膜")
    image_row = repo.get_image(int(annotation["image_id"]))
    if image_row is None:
        raise KeyError(f"影像 {annotation['image_id']} 不存在")
    image_path = config.abs_data_path(image_row["path"])
    if masker is None:
        masker = SamMasker(config.prelabel.sam.model, device=config.prelabel.device)
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        width, height = image.size
        px = (annotation["bbox_x1"] * width, annotation["bbox_y1"] * height,
              annotation["bbox_x2"] * width, annotation["bbox_y2"] * height)
        mask = masker.mask_for_box(image, px)
    if not mask.any():
        raise ValueError("SAM 未生成有效掩膜（可尝试扩大框或提高图像质量）")
    rel_path = save_mask(config, int(image_row["id"]), mask)
    metrics = mask_metrics(mask)
    mask_annotation_id = repo.add_mask_candidate(
        task_id=int(annotation["task_id"]), class_code=annotation["class_code"],
        bbox={"x1": annotation["bbox_x1"], "y1": annotation["bbox_y1"],
              "x2": annotation["bbox_x2"], "y2": annotation["bbox_y2"]},
        mask_path=rel_path, metrics=metrics, source="human",
        model_version_id=annotation["model_version_id"])
    return {"annotation_id": annotation_id, "mask_annotation_id": mask_annotation_id,
            "mask_path": rel_path, "metrics": metrics}


def mask_file_path(config: Config, rel_path: str) -> Path:
    """掩膜的绝对路径（供 API 流式返回）。"""
    return config.abs_data_path(rel_path)
