"""数据集：选样 → 划分（按 GPS 网格分组，防泄漏）→ 冻结（清单哈希）→ 导出。

冻结语义（ADR-0005）：冻结后不可变；`manifest_hash` 覆盖
「图片内容哈希 + 标注（类别与归一化坐标）+ 类别顺序」，用于可复现训练与审计。

数据集只包含「已被人确认」的标注：`source != 'model'`（模型候选必须先在标注台采纳）。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

from ..config import Config
from ..errors import ConflictError, NotFoundError
from ..storage.repo import Repo
from .formats import SOURCE_KEYS, export_coco, export_labelme, export_yolo

ANNOTATION_CHUNK = 400


def _chunks(values: Sequence[int], size: int) -> Iterable[Sequence[int]]:
    for start in range(0, len(values), size):
        yield values[start:start + size]


def fetch_dataset_rows(repo: Repo, items: Sequence[dict[str, Any]], *,
                       include_model: bool = False) -> list[dict[str, Any]]:
    """为一份 dataset_items（含 split）补齐影像字段与人工标注。"""
    if not items:
        return []
    image_ids = [int(item["id"]) for item in items]
    annotations: dict[int, list[dict[str, Any]]] = {image_id: [] for image_id in image_ids}
    for chunk in _chunks(image_ids, ANNOTATION_CHUNK):
        placeholders = ",".join("?" * len(chunk))
        sql = (f"SELECT image_id, class_code, bbox_x1, bbox_y1, bbox_x2, bbox_y2, source, score, difficult "
               f"FROM annotations WHERE deleted_at IS NULL AND image_id IN ({placeholders})")
        if not include_model:
            sql += " AND source <> 'model'"
        for row in repo.conn.execute(sql, list(chunk)):
            annotations[row["image_id"]].append({
                "class_code": row["class_code"],
                "bbox": {"x1": row["bbox_x1"], "y1": row["bbox_y1"],
                         "x2": row["bbox_x2"], "y2": row["bbox_y2"]},
                "source": row["source"], "score": row["score"], "difficult": row["difficult"],
            })
    rows: list[dict[str, Any]] = []
    for item in items:
        image_id = int(item["id"])
        rows.append({
            "split": item["split"],
            "image": {key: item[key] for key in
                      ("id", "path", "width", "height", "sha256", "captured_at", "gps_lat", "gps_lon")},
            "annotations": annotations[image_id],
        })
    return rows


def select_images(repo: Repo, filters: dict[str, Any]) -> list[dict[str, Any]]:
    """按条件选样（返回 images 行，含 task 状态）。

    filters: class_codes[] / batch_ids[] / captured_from / captured_to / review_status
    """
    clauses = ["i.duplicate_of IS NULL", "t.status IN ('annotated','reviewing','approved')"]
    params: list[Any] = []
    review_status = filters.get("review_status", "approved")
    if review_status == "approved":
        clauses.append("t.status = 'approved'")
    elif review_status == "annotated":
        clauses.append("t.status IN ('annotated','reviewing','approved')")
    elif review_status == "any":
        clauses = ["i.duplicate_of IS NULL"]
    if filters.get("batch_ids"):
        placeholders = ",".join("?" * len(filters["batch_ids"]))
        clauses.append(f"i.batch_id IN ({placeholders})")
        params.extend(filters["batch_ids"])
    if filters.get("captured_from"):
        clauses.append("i.captured_at >= ?")
        params.append(filters["captured_from"])
    if filters.get("captured_to"):
        clauses.append("i.captured_at <= ?")
        params.append(filters["captured_to"])
    if filters.get("class_codes"):
        placeholders = ",".join("?" * len(filters["class_codes"]))
        clauses.append(
            "EXISTS (SELECT 1 FROM annotations a WHERE a.image_id = i.id AND a.deleted_at IS NULL "
            f"AND a.source <> 'model' AND a.class_code IN ({placeholders}))"
        )
        params.extend(filters["class_codes"])
    sql = (f"SELECT i.*, t.status AS task_status, t.id AS task_id "
           f"FROM images i JOIN tasks t ON t.image_id = i.id "
           f"WHERE {' AND '.join(clauses)} ORDER BY i.id")
    return [dict(row) for row in repo.conn.execute(sql, params)]


def group_key(row: dict[str, Any]) -> str:
    """划分分组键：优先 GPS 网格（0.01°≈1km），无 GPS 时退化为「每图独立」。"""
    lat, lon = row.get("gps_lat"), row.get("gps_lon")
    if lat is None or lon is None:
        return f"img:{row['id']}"
    return f"gps:{round(float(lat), 2):.2f},{round(float(lon), 2):.2f}"


def assign_splits(rows: Sequence[dict[str, Any]], split_cfg: dict[str, Any]) -> list[tuple[dict[str, Any], str]]:
    """按分组键确定性划分（同一路段只落一个 split，避免训练/测试泄漏）。"""
    ratios = {
        "train": float(split_cfg.get("train", 0.7)),
        "val": float(split_cfg.get("val", 0.15)),
        "test": float(split_cfg.get("test", 0.15)),
    }
    total = sum(ratios.values()) or 1.0
    ratios = {key: value / total for key, value in ratios.items()}
    seed = str(split_cfg.get("seed", 42))
    group_by = split_cfg.get("group_by", "gps_grid")

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        key = group_key(row) if group_by != "none" else f"img:{row['id']}"
        grouped.setdefault(key, []).append(row)

    # 组内先按行 id 稳定排序，组间按 (seed, key) 哈希排序 → 与输入顺序无关
    ordered_groups = sorted(grouped.items(),
                            key=lambda kv: hashlib.sha256(f"{seed}:{kv[0]}".encode()).hexdigest())
    assignments: list[tuple[dict[str, Any], str]] = []
    thresholds = {"train": ratios["train"], "val": ratios["train"] + ratios["val"]}
    for index, (_, group_rows) in enumerate(ordered_groups):
        position = (index + 0.5) / max(1, len(ordered_groups))
        split = "train" if position <= thresholds["train"] else "val" if position <= thresholds["val"] else "test"
        for row in sorted(group_rows, key=lambda r: r["id"]):
            assignments.append((row, split))
    return assignments


def create_draft(config: Config, repo: Repo, name: str, filters: dict[str, Any],
                 split_cfg: dict[str, Any]) -> dict[str, Any]:
    """建立数据集草稿：选样 + 划分 + 落库（不导出、不可训练）。"""
    rows = select_images(repo, filters)
    assignments = assign_splits(rows, split_cfg)
    dataset_id = repo.create_dataset(name, filter_json=filters, split_json=split_cfg)
    repo.add_dataset_items(dataset_id, [(row["id"], split) for row, split in assignments])
    stats = {"images": len(assignments),
             "splits": {split: sum(1 for _, s in assignments if s == split) for split in ("train", "val", "test")},
             "classes": _class_histogram([row["id"] for row, _ in assignments], repo.conn)}
    return {**repo.get_dataset(dataset_id=dataset_id), "stats": stats}  # type: ignore[dict-item]


def _class_histogram(image_ids: Sequence[int], conn) -> dict[str, int]:
    histogram: dict[str, int] = {}
    for chunk in _chunks(list(image_ids), ANNOTATION_CHUNK):
        placeholders = ",".join("?" * len(chunk))
        for row in conn.execute(
            f"SELECT class_code, COUNT(*) AS n FROM annotations "
            f"WHERE deleted_at IS NULL AND source <> 'model' AND image_id IN ({placeholders}) "
            f"GROUP BY class_code", list(chunk),
        ):
            histogram[row["class_code"]] = histogram.get(row["class_code"], 0) + row["n"]
    return histogram


def compute_manifest_hash(rows: Sequence[dict[str, Any]], class_order: Sequence[str]) -> str:
    """清单哈希：图片内容哈希 + 标注（类别/坐标）+ 类别顺序。"""
    payload = []
    for row in sorted(rows, key=lambda r: r["image"]["sha256"]):
        annotations = sorted(
            [[ann["class_code"], round(ann["bbox"]["x1"], 6), round(ann["bbox"]["y1"], 6),
              round(ann["bbox"]["x2"], 6), round(ann["bbox"]["y2"], 6)] for ann in row["annotations"]],
            key=lambda item: (item[0], item[1], item[2]),
        )
        payload.append({"sha256": row["image"]["sha256"], "split": row["split"], "annotations": annotations})
    document = {"class_order": list(class_order), "items": payload}
    return hashlib.sha256(json.dumps(document, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def freeze_dataset(config: Config, repo: Repo, *, name: str | None = None, dataset_id: int | None = None,
                   export_formats: Sequence[str] = ("yolo",), copy_images: bool = True) -> dict[str, Any]:
    """冻结数据集：导出 + 清单哈希 + 状态置 frozen（之后不可变）。"""
    dataset = repo.get_dataset(dataset_id=dataset_id, name=name)
    if dataset is None:
        raise NotFoundError(f"数据集不存在: {name or dataset_id}")
    if dataset["status"] != "draft":
        raise ConflictError(f"数据集 {dataset['name']} 状态为 {dataset['status']}，只有 draft 可以冻结")
    items = repo.dataset_items(dataset["id"])
    if not items:
        raise ValueError(f"数据集 {dataset['name']} 没有任何样本，无法冻结")
    with_annotations = [item for item in items if item.get("id")]
    rows = fetch_dataset_rows(repo, with_annotations)
    classes = repo.list_classes()
    class_order = [cls["code"] for cls in sorted(classes, key=lambda c: (c["order_index"], c["code"]))]
    manifest = compute_manifest_hash(rows, class_order)

    root = config.datasets_dir / dataset["name"]
    root.mkdir(parents=True, exist_ok=True)
    abs_rows = []
    for row in rows:
        enriched = dict(row)
        enriched["abs_path"] = str(config.abs_data_path(row["image"]["path"]))
        abs_rows.append(enriched)

    exports: dict[str, Any] = {}
    for fmt in export_formats:
        if fmt == "yolo":
            exports["yolo"] = export_yolo(abs_rows, classes, root, copy_images=copy_images)
        elif fmt == "coco":
            exports["coco"] = export_coco(abs_rows, classes, root / "coco.json", copy_images=False)
        elif fmt == "labelme":
            exports["labelme"] = export_labelme(abs_rows, classes, root / "labelme")
        else:
            raise ValueError(f"未知导出格式: {fmt}")

    stats = {
        "images": len(rows),
        "splits": {split: sum(1 for row in rows if row["split"] == split) for split in ("train", "val", "test")},
        "labels": sum(len(row["annotations"]) for row in rows),
        "classes": _class_histogram([row["image"]["id"] for row in rows], repo.conn),
        "exports": {key: value.get("images", value.get("file")) for key, value in exports.items()},
    }
    (root / "manifest.sha256").write_text(manifest + "\n", encoding="utf-8")
    (root / "dataset.json").write_text(
        json.dumps({"name": dataset["name"], "manifest_hash": manifest, "class_order": class_order,
                    "stats": stats, "exports": exports}, ensure_ascii=False, indent=2), encoding="utf-8")

    frozen = repo.freeze_dataset(dataset["id"], class_order=class_order, manifest_hash=manifest,
                                 stats=stats, root_path=str(root), split_json=dataset["split_json"] or {})
    frozen["stats"] = stats
    frozen["exports"] = exports
    return frozen


def dataset_summary(config: Config, repo: Repo, dataset: dict[str, Any]) -> dict[str, Any]:
    """给 CLI/API 的紧凑摘要（不含逐图明细）。"""
    return {
        "id": dataset["id"], "name": dataset["name"], "status": dataset["status"],
        "manifest_hash": dataset.get("manifest_hash"), "root_path": dataset.get("root_path"),
        "frozen_at": dataset.get("frozen_at"),
        "stats": json.loads(dataset["stats_json"]) if dataset.get("stats_json") else None,
    }
