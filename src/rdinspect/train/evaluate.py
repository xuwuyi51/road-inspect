"""评估：冻结数据集上的分类别指标、混淆矩阵、大小桶召回、SAHI 开关对比与失败样例导出（M3）。

两种口径并存，且都会写进 `metrics.json`：

* ``ultralytics`` —— ``YOLO.val()`` 的官方口径（mAP50 / mAP50-95 / 分类别 P·R），用于门禁主判据；
* ``internal``    —— 本项目自己的贪心匹配口径（:mod:`rdinspect.train.matching`），
  用于混淆矩阵、大小桶召回、固定阈值 P/R，以及**切片推理开关对比**（官方 val 不支持逐图切片）。

失败样例（漏检 / 误检 / 类别混淆）会导出带标注叠加图的 PNG，便于回看标注质量而不是只看一个数字。
"""

from __future__ import annotations

import json
import shutil
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Sequence

from PIL import Image, ImageDraw

from ..config import Config
from ..errors import NotFoundError
from ..prelabel.detector import Detector, UltralyticsDetector, map_class, to_normalized_bbox
from ..prelabel.sahi import predict_sliced
from ..storage.files import sha256_file
from ..storage.repo import Repo
from . import matching
from .runner import absolute_data_yaml, dataset_data_yaml, resolve_train_device, slugify

#: 失败样例叠加图的最长边（保持轻量，原图另存硬链接）
FAILURE_OVERLAY_LONG_EDGE = 1024
GT_COLOR = (46, 204, 113)
PRED_COLOR = (231, 76, 60)

ProgressCallback = Callable[[int, int, str], None]


# ─────────────────────────── 预测与真值 ───────────────────────────
def predict_normalized(detector: Detector, image: Image.Image, *, known_codes: Sequence[str],
                       aliases: dict[str, str], tile_config: Any = None,
                       use_tiles: bool = False, conf: float = 0.0) -> dict[str, Any]:
    """一次（可选切片的）推理 → 归一化预测框 + 未映射类别统计。"""
    if use_tiles and tile_config is not None:
        detections, tiles = predict_sliced(image, detector, tile_config=tile_config)
    else:
        detections, tiles = detector.predict(image), 1
    width, height = image.size
    preds: list[dict[str, Any]] = []
    unmapped: dict[str, int] = {}
    for detection in detections:
        if float(detection.score) < conf:
            continue
        code = map_class(detection.class_name, aliases, known_codes)
        if code is None:
            unmapped[detection.class_name] = unmapped.get(detection.class_name, 0) + 1
            continue
        bbox = to_normalized_bbox(detection.bbox, width, height)
        if bbox is None:
            continue
        preds.append({"class_code": code, "bbox": bbox, "score": float(detection.score)})
    return {"preds": preds, "tiles": tiles, "unmapped": unmapped, "raw_detections": len(detections)}


def _match_payload(preds: Sequence[dict[str, Any]], gts: Sequence[dict[str, Any]],
                   iou_thr: float) -> dict[str, Any]:
    return matching.match_detections(list(preds), list(gts), iou_thr=iou_thr)


# ─────────────────────────── 叠加图与失败样例 ───────────────────────────
def draw_overlay(image: Image.Image, gts: Sequence[dict[str, Any]], preds: Sequence[dict[str, Any]],
                 *, matched_gt: set[int], matched_pred: set[int]) -> Image.Image:
    """画真值（绿）与预测（红）：实线=已匹配，虚线框感=未匹配（用较粗/较淡区分）。"""
    canvas = image.convert("RGB").copy()
    scale = 1.0
    if max(canvas.size) > FAILURE_OVERLAY_LONG_EDGE:
        scale = FAILURE_OVERLAY_LONG_EDGE / max(canvas.size)
        canvas = canvas.resize((max(1, int(canvas.width * scale)), max(1, int(canvas.height * scale))))
    draw = ImageDraw.Draw(canvas)
    width, height = canvas.size

    def to_px(bbox: dict[str, float]) -> tuple[float, float, float, float]:
        return (bbox["x1"] * width, bbox["y1"] * height, bbox["x2"] * width, bbox["y2"] * height)

    for index, gt in enumerate(gts):
        box = to_px(gt["bbox"])
        draw.rectangle(box, outline=GT_COLOR, width=3 if index in matched_gt else 2)
        if index not in matched_gt:
            draw.text((box[0] + 2, max(0, box[1] - 11)), f"miss {gt['class_code']}", fill=GT_COLOR)
    for index, pred in enumerate(preds):
        box = to_px(pred["bbox"])
        draw.rectangle(box, outline=PRED_COLOR, width=2)
        if index not in matched_pred:
            draw.text((box[0] + 2, box[3] + 1), f"fp {pred['class_code']} {pred['score']:.2f}", fill=PRED_COLOR)
    return canvas


def export_failures(work_dir: Path, items: Sequence[dict[str, Any]], *, limit: int,
                    conf: float, iou_thr: float) -> dict[str, Any]:
    """把失败样例写成 `failures/`：叠加图 + 原始图硬链接 + failures.json 索引。"""
    out_dir = work_dir / "failures"
    entries: list[dict[str, Any]] = []
    if limit <= 0:
        return {"dir": str(out_dir), "total": 0, "exported": 0, "items": []}
    ranked: list[tuple[int, dict[str, Any]]] = []
    for item in items:
        match = item["match"]
        miss = len(match["missed"])
        fp = len(match["extra"])
        confusion = len(match["class_confusions"])
        if miss or fp or confusion:
            ranked.append((miss * 2 + fp + confusion, item))
    ranked.sort(key=lambda pair: (-pair[0], pair[1]["image_id"]))
    out_dir.mkdir(parents=True, exist_ok=True)
    for score, item in ranked[:limit]:
        match = item["match"]
        stem = f"img{item['image_id']}-miss{len(match['missed'])}-fp{len(match['extra'])}"
        overlay_path = out_dir / f"{stem}.png"
        try:
            with Image.open(item["abs_path"]) as handle:
                overlay = draw_overlay(handle, item["gts"], item["preds"],
                                       matched_gt=set(item["match"]["matched_gt_indices"]),
                                       matched_pred=set(item["match"]["matched_pred_indices"]))
            overlay.save(overlay_path)
            source_link = out_dir / f"{stem}{Path(item['abs_path']).suffix}"
            if not source_link.exists():
                try:
                    source_link.hardlink_to(Path(item["abs_path"]))
                except Exception:  # noqa: BLE001 - 跨分区时退化为复制
                    shutil.copy2(item["abs_path"], source_link)
        except Exception as exc:  # noqa: BLE001 - 单张失败不影响整体评估
            entries.append({"image_id": item["image_id"], "error": str(exc)})
            continue
        entries.append({
            "image_id": item["image_id"], "split": item["split"], "severity": score,
            "overlay": str(overlay_path.relative_to(work_dir)),
            "missed": [{"class_code": item["gts"][i]["class_code"]} for i in match["missed"]],
            "extra": [{"class_code": item["preds"][i]["class_code"], "score": round(float(item["preds"][i]["score"]), 4)}
                      for i in match["extra"]],
            "class_confusions": [{"gt_class": pair["gt_class"], "pred_class": pair["pred_class"],
                                  "iou": round(float(pair["iou"]), 4)} for pair in match["class_confusions"]],
        })
    (out_dir / "failures.json").write_text(
        json.dumps({"conf": conf, "iou_thr": iou_thr, "items": entries}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    return {"dir": str(out_dir), "total": len(ranked), "exported": len(entries), "items": entries[:limit]}


# ─────────────────────────── 核心评估 ───────────────────────────
def evaluate_predictions(config: Config, repo: Repo, *, dataset: dict[str, Any], split: str,
                         detector: Detector, classes: Sequence[dict[str, Any]],
                         conf: float, iou: float, tile_config: Any = None,
                         sahi_ablation: bool = False, export_failure_limit: int = 0,
                         work_dir: Path | None = None, val_metrics: dict[str, Any] | None = None,
                         progress: ProgressCallback | None = None) -> dict[str, Any]:
    """在冻结数据集的某个 split 上评估（检测器可注入 → 无 ML 依赖也能单测）。"""
    from ..core import datasets as datasets_mod  # 局部导入避免循环

    items = repo.dataset_items(int(dataset["id"]), split=split)
    if not items:
        raise NotFoundError(f"数据集 {dataset['name']} 的 {split} 划分为空，无法评估")
    rows = datasets_mod.fetch_dataset_rows(repo, items)
    known_codes = [cls["code"] for cls in classes]
    aliases = dict(config.prelabel.class_aliases)

    per_image: list[dict[str, Any]] = []
    plain_preds: list[dict[str, Any]] = []
    tiled_preds: list[dict[str, Any]] = []
    failures_source: list[dict[str, Any]] = []
    unmapped: dict[str, int] = {}
    tiles_used = 0

    for index, row in enumerate(rows, start=1):
        image_id = int(row["image"]["id"])
        abs_path = config.abs_data_path(row["image"]["path"])
        gts = [{"class_code": ann["class_code"], "bbox": ann["bbox"]} for ann in row["annotations"]]
        with Image.open(abs_path) as handle:
            image = handle.convert("RGB")
            plain = predict_normalized(detector, image, known_codes=known_codes, aliases=aliases,
                                       tile_config=tile_config, use_tiles=False, conf=conf)
            tiled = None
            if sahi_ablation:
                tiled = predict_normalized(detector, image, known_codes=known_codes, aliases=aliases,
                                           tile_config=tile_config, use_tiles=True, conf=conf)
        for key, value in plain["unmapped"].items():
            unmapped[key] = unmapped.get(key, 0) + value
        tiles_used += int(tiled["tiles"]) if tiled else int(plain["tiles"])
        entry = {"image_id": image_id, "split": split, "abs_path": str(abs_path),
                 "width": int(row["image"]["width"]), "height": int(row["image"]["height"]),
                 "gts": gts, "preds": plain["preds"]}
        plain_preds.append({"image_id": image_id, "gts": gts, "preds": plain["preds"]})
        if tiled is not None:
            tiled_preds.append({"image_id": image_id, "gts": gts, "preds": tiled["preds"]})
        per_image.append(entry)
        if progress is not None:
            progress(index, len(rows), f"image {image_id}")

    # 匹配与指标
    for entry in per_image:
        match = _match_payload(entry["preds"], entry["gts"], iou)
        matched_gt = {int(pair["gt_index"]) for pair in match["pairs"]}
        matched_pred = {int(pair["pred_index"]) for pair in match["pairs"]}
        failures_source.append({**entry, "match": {**match,
                                                   "matched_gt_indices": sorted(matched_gt),
                                                   "matched_pred_indices": sorted(matched_pred)}})

    ap_payload = matching.average_precision(per_image, known_codes, iou_thr=iou)
    # 官方 val 的 mAP 只在「有真值的类别」上取平均；内部口径按类别表全量平均（缺真值的类记 0）。
    # 两个数都留下，避免「表里有 5 类、本次只评了 1 类」时把 0.2 误读成模型很差。
    present_classes = [code for code, entry in ap_payload.get("per_class", {}).items()
                       if not entry.get("no_gt")]
    ap_payload["classes_with_gt"] = present_classes
    ap_payload["map50_classes_with_gt"] = (
        round(sum(float(ap_payload["per_class"][code]["ap"]) for code in present_classes)
              / len(present_classes), 6) if present_classes else None)

    internal = {
        "map50": ap_payload,
        "precision_recall": matching.precision_recall_at(per_image, known_codes, conf=conf, iou_thr=iou),
        "confusion_matrix": matching.confusion_matrix(per_image, known_codes, iou_thr=iou, score_thr=conf),
        "size_bucket_recall": matching.size_bucket_recall(per_image, iou_thr=iou),
        "images": len(per_image),
        "gt_boxes": sum(len(entry["gts"]) for entry in per_image),
        "pred_boxes": sum(len(entry["preds"]) for entry in per_image),
        "unmapped_classes": unmapped,
    }

    payload: dict[str, Any] = {
        "dataset": dataset["name"], "dataset_id": int(dataset["id"]),
        "manifest_hash": dataset.get("manifest_hash"), "split": split,
        "conf": conf, "iou": iou,
        "ultralytics": val_metrics or {},
        "internal": internal,
        "sahi_ablation": None,
    }
    if sahi_ablation and tiled_preds:
        payload["sahi_ablation"] = {
            "off": matching.precision_recall_at(plain_preds, known_codes, conf=conf, iou_thr=iou),
            "on": matching.precision_recall_at(tiled_preds, known_codes, conf=conf, iou_thr=iou),
            "map50_off": matching.average_precision(plain_preds, known_codes, iou_thr=iou)["map50"],
            "map50_on": matching.average_precision(tiled_preds, known_codes, iou_thr=iou)["map50"],
            "boxes_off": sum(len(entry["preds"]) for entry in plain_preds),
            "boxes_on": sum(len(entry["preds"]) for entry in tiled_preds),
            "tiles_used": tiles_used,
            "note": "切片只对细裂缝/航拍小目标有利；大目标被切碎后可能整体漏检（ADR-0006）",
        }
        ab = payload["sahi_ablation"]
        off_r = (ab["off"]["overall"]["recall"] if ab["off"]["overall"]["recall"] is not None else 0.0)
        on_r = (ab["on"]["overall"]["recall"] if ab["on"]["overall"]["recall"] is not None else 0.0)
        ab["recall_delta"] = round(on_r - off_r, 6)

    if work_dir is not None:
        failures = export_failures(work_dir, failures_source, limit=export_failure_limit,
                                   conf=conf, iou_thr=iou)
        payload["failures"] = {key: value for key, value in failures.items() if key != "items"}
        payload["failures"]["worst"] = failures["items"]
    return payload


def _as_list(values: Any) -> list[Any]:
    """numpy 数组安全转 list（注意：不能写 ``values or []``，数组的真值判断会抛异常）。"""
    if values is None:
        return []
    try:
        return list(values)
    except TypeError:
        return []


def _pick(values: Any, position: int, class_index: int) -> float | None:
    """按「ap_class_index 位置」或「类别下标」取值（不同 ultralytics 版本长度语义不同）。"""
    seq = _as_list(values)
    for index in (position, class_index):
        if 0 <= index < len(seq):
            try:
                number = float(seq[index])
            except (TypeError, ValueError):
                continue
            if number == number:                      # 排除 NaN
                return number
    return None


def extract_val_metrics(results: Any, classes: Sequence[dict[str, Any]],
                        aliases: dict[str, str] | None = None) -> dict[str, Any]:
    """把 ``YOLO.val()`` 的 DetMetrics 解析成 {mean, per_class}（分类别指标走数组，不解析文本表）。

    分类别数据源：``box.ap50`` / ``box.maps`` / ``box.p`` / ``box.r`` + ``box.ap_class_index``，
    类别名经别名表映射回本项目 code（与预标注同一套映射，避免两处口径不一致）。
    """
    box = getattr(results, "box", None)
    names = {int(key): str(value) for key, value in dict(getattr(results, "names", {}) or {}).items()}
    payload: dict[str, Any] = {
        "map50": float(getattr(box, "map50", float("nan"))),
        "map50_95": float(getattr(box, "map", float("nan"))),
        "precision": float(getattr(box, "mp", float("nan"))),
        "recall": float(getattr(box, "mr", float("nan"))),
        "per_class": {},
    }
    if box is None:
        return payload
    known_codes = [cls["code"] for cls in classes]
    index_to_code = {int(cls.get("order_index", position)): cls["code"]
                     for position, cls in enumerate(sorted(classes, key=lambda c: (c["order_index"], c["code"])))}
    resolved_aliases = aliases or {}
    nt_per_class = _as_list(getattr(results, "nt_per_class", None))
    for position, class_index in enumerate(_as_list(getattr(box, "ap_class_index", None))):
        name = names.get(int(class_index), str(class_index))
        code = map_class(name, resolved_aliases, known_codes) if resolved_aliases else None
        if code is None:
            code = next((cls["code"] for cls in classes
                         if str(cls["code"]).lower() == name.lower()
                         or str(cls.get("name_en", "")).lower() == name.lower()), None)
        if code is None:
            code = index_to_code.get(int(class_index), name)
        payload["per_class"][code] = {
            "map50": _pick(getattr(box, "ap50", None), position, int(class_index)),
            "map50_95": _pick(getattr(box, "maps", None), position, int(class_index)),
            "precision": _pick(getattr(box, "p", None), position, int(class_index)),
            "recall": _pick(getattr(box, "r", None), position, int(class_index)),
            "instances": (int(nt_per_class[int(class_index)])
                          if int(class_index) < len(nt_per_class) else None),
        }
    return payload


def parse_val_summary(summary: Sequence[dict[str, Any]], classes: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """兼容路径（保留给"只有文本表"的旧版本）：把 ``summary()`` 行解析成 {per_class, mean}。

    主路径是 :func:`extract_val_metrics`（直接读数组，不受表格列名变动影响）。
    """
    per_class: dict[str, Any] = {}
    code_by_name = {str(cls["name_en"]).lower(): cls["code"] for cls in classes}
    code_by_name.update({str(cls["code"]).lower(): cls["code"] for cls in classes})
    mean: dict[str, float] = {}
    for row in summary:
        if not isinstance(row, dict):
            continue
        lowered = {str(key).lower(): value for key, value in row.items()}
        label = str(lowered.get("class", "")).strip()
        if label.lower() in ("all", "total"):
            for key, value in lowered.items():
                if isinstance(value, (int, float)):
                    mean[key] = float(value)
            continue
        code = code_by_name.get(label.lower())
        if code is None:
            continue
        per_class[code] = {key: (float(value) if isinstance(value, (int, float)) else value)
                           for key, value in lowered.items()}
    return {"per_class": per_class, "mean": mean}


def _val_metrics(config: Config, repo: Repo, *, weights: str, data_yaml: Path, split: str,
                 imgsz: int, device: str, conf: float, iou: float,
                 classes: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """ultralytics 官方 val 口径（失败时返回 {}，门禁会自动退回 internal 口径）。"""
    from ..prelabel.detector import ensure_runtime_env, ml_available  # noqa: PLC0415

    if not ml_available():
        return {}
    ensure_runtime_env()
    from .compat import apply_compat_patches, preload_ultralytics  # noqa: PLC0415

    preload_ultralytics()
    apply_compat_patches()
    from ultralytics import YOLO  # noqa: PLC0415

    model = YOLO(weights)
    results = model.val(data=str(data_yaml), split=split, imgsz=int(imgsz), device=device,
                        conf=conf, iou=iou, plots=False, verbose=False,
                        project=str(config.runs_dir), name=f"val-{slugify(split)}", exist_ok=True)
    if getattr(results, "box", None) is None:
        return {}
    try:
        return extract_val_metrics(results, classes, config.prelabel.class_aliases)
    except Exception as exc:  # noqa: BLE001 - 指标结构变动不应让评估整体失败
        return {"error": f"解析官方 val 指标失败: {type(exc).__name__}: {exc}", "per_class": {}}


def evaluate_weights(config: Config, repo: Repo, *, weights: str, dataset: dict[str, Any],
                     split: str = "val", imgsz: int | None = None, device: str | None = None,
                     conf: float | None = None, iou: float | None = None,
                     detector_factory: Callable[[Config, str], Detector] | None = None,
                     run_id: int | None = None, work_dir: Path | None = None,
                     sahi_ablation: bool | None = None,
                     export_failure_limit: int | None = None,
                     progress: ProgressCallback | None = None) -> dict[str, Any]:
    """完整评估：官方 val + 内部匹配口径 + 失败样例。"""
    eval_cfg = config.train.evaluate
    resolved_conf = float(conf if conf is not None else eval_cfg.conf)
    resolved_iou = float(iou if iou is not None else eval_cfg.iou)
    resolved_imgsz = int(imgsz or config.train.imgsz)
    if imgsz is None and detector_factory is None:
        resolved_imgsz = int(_model_train_imgsz(repo, weights) or resolved_imgsz)
    classes = repo.list_classes()
    data_yaml = absolute_data_yaml(config, dataset_data_yaml(config, repo, dataset))
    work_root = work_dir or (config.runs_dir / f"eval-{run_id if run_id is not None else 'adhoc'}-"
                                              f"{slugify(str(dataset['name']))}-{slugify(split)}")
    work_root.mkdir(parents=True, exist_ok=True)

    if detector_factory is not None:
        detector = detector_factory(config, weights)
    else:
        # 与官方 val 保持同一 imgsz/conf/iou：否则「官方 0.91 / 内部 0.27」这种差异只是分辨率不同造成的
        detector = UltralyticsDetector(weights, device=resolve_train_device(device or config.train.device),
                                       imgsz=resolved_imgsz, conf=resolved_conf, iou=resolved_iou,
                                       max_detections=config.prelabel.max_detections)
    # tiles > 0 → 强制开启切片并把窗口缩到指定边长（小图也能对照出差异）；否则沿用 prelabel.tile 的 auto 策略
    tile_config = (replace(config.prelabel.tile, enabled=True, size=int(eval_cfg.tiles))
                   if eval_cfg.tiles else config.prelabel.tile)
    ablate = eval_cfg.sahi_ablation if sahi_ablation is None else sahi_ablation
    failure_limit = eval_cfg.export_failures if export_failure_limit is None else export_failure_limit

    val_metrics: dict[str, Any] = {}
    if detector_factory is None:
        try:
            val_metrics = _val_metrics(config, repo, weights=weights, data_yaml=data_yaml, split=split,
                                       imgsz=resolved_imgsz,
                                       device=resolve_train_device(device or config.train.device),
                                       conf=resolved_conf, iou=resolved_iou, classes=classes)
        except Exception as exc:  # noqa: BLE001 - 官方 val 失败不阻塞内部口径评估
            val_metrics = {"error": f"{type(exc).__name__}: {exc}"}

    payload = evaluate_predictions(
        config, repo, dataset=dataset, split=split, detector=detector, classes=classes,
        conf=resolved_conf, iou=resolved_iou, tile_config=tile_config, sahi_ablation=ablate,
        export_failure_limit=failure_limit, work_dir=work_root, val_metrics=val_metrics,
        progress=progress,
    )
    payload.update({
        "weights": weights,
        "weights_sha256": _sha256_or_none(weights),
        "imgsz": resolved_imgsz,
        "generated_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "work_dir": str(work_root),
    })
    metrics_path = work_root / f"metrics-{slugify(split)}.json"
    metrics_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path = work_root / f"report-{slugify(split)}.md"
    report_path.write_text(render_report(payload), encoding="utf-8")
    payload["artifacts"] = {"metrics": str(metrics_path), "report": str(report_path),
                            "work_dir": str(work_root)}
    return payload


def _sha256_or_none(path: str | Path | None) -> str | None:
    if not path:
        return None
    candidate = Path(path)
    return sha256_file(candidate) if candidate.exists() and candidate.is_file() else None


# ─────────────────────────── 报告 ───────────────────────────
def _fmt(value: Any, digits: int = 4) -> str:
    if isinstance(value, float):
        return "—" if value != value else f"{value:.{digits}f}"   # NaN → —
    return "—" if value is None else str(value)


def primary_map50(metrics: dict[str, Any]) -> float | None:
    """门禁主判据：优先官方 val 的 mAP50，缺失时用内部口径。"""
    official = (metrics.get("ultralytics") or {}).get("map50")
    if isinstance(official, (int, float)) and official == official:
        return float(official)
    internal = ((metrics.get("internal") or {}).get("map50") or {})
    for key in ("map50_classes_with_gt", "map50"):
        value = internal.get(key)
        if isinstance(value, (int, float)) and value == value:
            return float(value)
    return None


def primary_per_class_map50(metrics: dict[str, Any]) -> dict[str, float]:
    """分类别 mAP50（优先官方口径；官方缺失时用内部 AP）。"""
    result: dict[str, float] = {}
    official = ((metrics.get("ultralytics") or {}).get("per_class") or {})
    for code, entry in official.items():
        value = None
        for key, candidate in entry.items():
            lowered = str(key).lower()
            if "map50" in lowered and "95" not in lowered and isinstance(candidate, (int, float)):
                value = float(candidate)
                break
        if value is not None:
            result[code] = value
    if result:
        return result
    internal = ((metrics.get("internal") or {}).get("map50") or {}).get("per_class") or {}
    return {code: float(entry.get("ap") or 0.0) for code, entry in internal.items()}


def render_report(metrics: dict[str, Any]) -> str:
    """人类可读报告（Markdown）：整体 → 分类别 → 大小桶 → 切片对比 → 失败样例。"""
    official = metrics.get("ultralytics") or {}
    internal = metrics.get("internal") or {}
    lines = [
        f"# 评估报告 · {metrics.get('dataset')} / {metrics.get('split')}",
        "",
        f"- 权重：`{metrics.get('weights')}`（sha256 `{(metrics.get('weights_sha256') or '')[:12]}`）",
        f"- 数据集清单：`{metrics.get('manifest_hash')}`",
        f"- imgsz={metrics.get('imgsz')} · conf={metrics.get('conf')} · iou={metrics.get('iou')}",
        f"- 图像 {internal.get('images')} 张 · 真值框 {internal.get('gt_boxes')} · 预测框 {internal.get('pred_boxes')}",
        "",
        "## 1. 整体指标",
        "",
        "| 口径 | mAP50 | mAP50-95 | 精确率 | 召回率 |",
        "|---|---|---|---|---|",
        f"| ultralytics val | {_fmt(official.get('map50'))} | {_fmt(official.get('map50_95'))} | "
        f"{_fmt(official.get('precision'))} | {_fmt(official.get('recall'))} |",
        f"| 内部匹配（AP/mAP50，仅有真值的类） | "
        f"{_fmt((internal.get('map50') or {}).get('map50_classes_with_gt'))} | — | "
        f"{_fmt(((internal.get('precision_recall') or {}).get('overall') or {}).get('precision'))} | "
        f"{_fmt(((internal.get('precision_recall') or {}).get('overall') or {}).get('recall'))} |",
        "",
        "## 2. 分类别指标",
        "",
        "| 类别 | 官方 mAP50 | mAP50-95 | 实例数 | 内部 AP | 内部 TP/FP/FN |",
        "|---|---|---|---|---|---|",
    ]
    official_per_class = official.get("per_class") or {}
    internal_per_class = ((internal.get("map50") or {}).get("per_class") or {})
    pr_per_class = ((internal.get("precision_recall") or {}).get("per_class") or {})
    codes = sorted(set(official_per_class) | set(internal_per_class))
    for code in codes:
        entry = official_per_class.get(code) or {}
        map50 = next((value for key, value in entry.items()
                      if "map50" in str(key) and "95" not in str(key)), None)
        map5095 = next((value for key, value in entry.items()
                        if "map50-95" in str(key).lower() or "map50_95" in str(key).lower()), None)
        instances = entry.get("instances")
        internal_entry = internal_per_class.get(code) or {}
        pr = pr_per_class.get(code) or {}
        lines.append(
            f"| {code} | {_fmt(map50)} | {_fmt(map5095)} | {_fmt(instances, 0)} | "
            f"{_fmt(internal_entry.get('ap'))} | "
            f"{pr.get('tp', '—')}/{pr.get('fp', '—')}/{pr.get('fn', '—')} |")
    buckets = (internal.get("size_bucket_recall") or {}).get("buckets") or {}
    lines += ["", "## 3. 大小桶召回（COCO 口径，IoU 0.5）", "",
              "| 桶 | 真值数 | 命中 | 召回 |", "|---|---|---|---|"]
    for name in ("small", "medium", "large"):
        entry = buckets.get(name) or {}
        lines.append(f"| {name} | {entry.get('gt', '—')} | {entry.get('matched', '—')} | {_fmt(entry.get('recall'))} |")
    all_class_map = (internal.get("map50") or {}).get("map50")
    if all_class_map is not None:
        lines.append(f"| 内部匹配（按类别表全量平均，缺真值的类记 0） | {_fmt(all_class_map)} | — | — | — |")
    ablation = metrics.get("sahi_ablation")
    if ablation:
        lines += ["", "## 4. 切片推理开关对比（ADR-0006）", "",
                  f"- 关闭切片：mAP50 {_fmt(ablation.get('map50_off'))} · "
                  f"召回 {_fmt((ablation['off']['overall'] or {}).get('recall'))}",
                  f"- 开启切片：mAP50 {_fmt(ablation.get('map50_on'))} · "
                  f"召回 {_fmt((ablation['on']['overall'] or {}).get('recall'))} · "
                  f"切片数 {ablation.get('tiles_used')} · 框数 {ablation.get('boxes_off')}→{ablation.get('boxes_on')}",
                  f"- 召回增量：{_fmt(ablation.get('recall_delta'))}"]
    cm = internal.get("confusion_matrix") or {}
    if cm.get("matrix"):
        lines += ["", "## 5. 混淆矩阵（行=真值，列=预测，最后一列/行=背景）", "",
                  "| 真值\\预测 | " + " | ".join(cm["labels"]) + " |",
                  "|" + "---|" * (len(cm["labels"]) + 1)]
        for label, row in zip(cm["labels"], cm["matrix"]):
            lines.append(f"| {label} | " + " | ".join(str(value) for value in row) + " |")
    failures = metrics.get("failures") or {}
    if failures:
        lines += ["", "## 6. 失败样例", "",
                  f"- 含失败样例图像 {failures.get('total')} 张，已导出 {failures.get('exported')} 张 → `{failures.get('dir')}`"]
        for item in (failures.get("worst") or [])[:10]:
            if "error" in item:
                continue
            lines.append(f"  - image {item.get('image_id')}（严重度 {item.get('severity')}）："
                         f"漏检 {len(item.get('missed') or [])} · 误检 {len(item.get('extra') or [])} · "
                         f"类别混淆 {len(item.get('class_confusions') or [])} → `{item.get('overlay')}`")
    if internal.get("unmapped_classes"):
        lines += ["", "## 7. 未映射类别（被丢弃的检测）", "",
                  ", ".join(f"{key}×{value}" for key, value in internal["unmapped_classes"].items())]
    lines.append("")
    return "\n".join(lines)


def _model_train_imgsz(repo: Repo, weights: str) -> int | None:
    """模型训练时的 imgsz（按权重路径匹配），用于「评估分辨率 = 训练分辨率」。"""
    try:
        row = repo.conn.execute(
            "SELECT metrics_json FROM model_versions WHERE weights_path = ? ORDER BY id DESC LIMIT 1",
            (str(weights),)).fetchone()
        if row is None:
            return None
        payload = json.loads(row["metrics_json"] or "{}")
        value = (payload.get("train_params") or {}).get("imgsz")
        return int(value) if value else None
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


