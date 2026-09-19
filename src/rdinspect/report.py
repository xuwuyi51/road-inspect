"""M5 统计报表：时间趋势、批次对比、类别覆盖与 GIS 字段导出。

设计要点
  * **只读**：本模块不写任何表，也不产生 audit_log 行，可安全重放（CLI/API/定时报表共用）；
  * **口径统一**：近似重复影像（``images.duplicate_of IS NOT NULL``）与软删标注
    （``annotations.deleted_at IS NOT NULL``）一律排除；"已确认标注" = ``source <> 'model'``
    （模型候选在人工确认前不计入任何产量统计，``model_edited`` 已采纳故计入）；
  * **时间**：库内时间列均为 UTC ISO8601 文本 ``YYYY-MM-DDTHH:MM:SSZ``，窗口按 UTC 自然日切分，
    因此可以直接用 ``substr(col, 1, 10)`` 做字典序日期比较（列宽固定，见 db/schema.sql §设计要点）；
  * **诚实报告**：``gis_features`` 同时给出"能画到地图上的点数"与"缺少 GPS 的点数"，
    便于报表如实说明覆盖率，而不是静默丢弃。

本模块与 db/schema.sql 的三处偏差（已在 docstring 中就地对齐，未修改 schema）：
  1. ``annotations.source`` 的 CHECK 取值为 ``human|model|model_edited|external``，**没有** ``import``；
     文档把导入标注写作 ``import``，故此处把 ``import`` 别名展开为 ``{'import','external'}``；
  2. ``annotations`` 表**没有** ``area_ratio`` / ``length_px`` / ``width_px`` 列（这些指标目前写在
     ``geometry_json`` 里，见 ``prelabel/sam.py``）；导出时按「列（若未来补列）→ geometry_json →
     bbox 面积」的优先级取值，``length_px``/``width_px`` 取不到时为 ``None``；
  3. ``reviews`` 表没有 ``kappa`` 列（``coverage()["review"]`` 只用 decision 汇总）。
"""

from __future__ import annotations

import csv
import io
import json
import sqlite3
from datetime import date, datetime, timedelta, timezone
from typing import Any, Sequence

from .storage.files import utc_now
from .storage.repo import Repo

#: ``trends()`` 支持的分桶粒度。
BUCKETS: tuple[str, ...] = ("day", "week", "month")

#: ``gis_features()`` 缺省导出的标注来源（``model`` 候选需显式传入才导出）。
DEFAULT_GIS_SOURCES: tuple[str, ...] = ("human", "model_edited", "import")

#: 来源别名：文档里的 ``import`` 在本库实际落在 ``external``（schema CHECK 无 import）。
_SOURCE_ALIASES: dict[str, tuple[str, ...]] = {"import": ("import", "external")}

#: GIS 导出中可选的派生指标（列不存在时回退到 geometry_json）。
_METRIC_KEYS: tuple[str, ...] = ("area_ratio", "length_px", "width_px")

#: Markdown 报表里的缺失值占位符。
_MISSING = "—"

_GEOJSON_CSV_HEADER: tuple[str, ...] = (
    "image_id", "path", "captured_at", "lat", "lon", "class", "source", "score",
    "x1", "y1", "x2", "y2", "length_px", "width_px", "area_ratio",
)


# ────────────────────────────── 通用小工具 ──────────────────────────────
def _bucket_label(day: date, bucket: str) -> str:
    """日期 → 分桶键（day: ``YYYY-MM-DD``；week: 该 ISO 周的周一；month: ``YYYY-MM``）。"""
    if bucket == "day":
        return day.isoformat()
    if bucket == "week":
        return (day - timedelta(days=day.weekday())).isoformat()
    return f"{day.year:04d}-{day.month:02d}"


def _bucket_labels(start: date, end: date, bucket: str) -> list[str]:
    """窗口内全部分桶键（时间升序，含空桶）。"""
    labels: list[str] = []
    if bucket == "day":
        cursor = start
        while cursor <= end:
            labels.append(_bucket_label(cursor, bucket))
            cursor += timedelta(days=1)
        return labels
    if bucket == "week":
        cursor = start - timedelta(days=start.weekday())
        last = end - timedelta(days=end.weekday())
        while cursor <= last:
            labels.append(_bucket_label(cursor, bucket))
            cursor += timedelta(days=7)
        return labels
    cursor = date(start.year, start.month, 1)
    last = date(end.year, end.month, 1)
    while cursor <= last:
        labels.append(_bucket_label(cursor, bucket))
        cursor = date(cursor.year + cursor.month // 12, cursor.month % 12 + 1, 1)
    return labels


def _parse_date_arg(name: str, value: str | None) -> date | None:
    """解析 ``YYYY-MM-DD`` 过滤器；非法值抛 ``ValueError``。"""
    if value is None:
        return None
    try:
        return date.fromisoformat(str(value).strip()[:10])
    except ValueError as exc:
        raise ValueError(f"{name} 必须是 YYYY-MM-DD 格式，收到 {value!r}") from exc


def _parse_json_object(raw: Any) -> dict[str, Any]:
    """解析 JSON 对象文本；空值/坏 JSON/非对象一律返回 ``{}``（报表不得因此失败）。"""
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _expand_sources(sources: Sequence[str] | None) -> tuple[str, ...]:
    """展开来源列表（``import`` → ``import``/``external`` 别名），保持首次出现顺序。"""
    requested = list(DEFAULT_GIS_SOURCES if sources is None else sources)
    expanded: list[str] = []
    for name in requested:
        for alias in _SOURCE_ALIASES.get(name, (name,)):
            if alias not in expanded:
                expanded.append(alias)
    return tuple(expanded)


def _source_clause(sources: Sequence[str]) -> tuple[str, list[Any]]:
    """构造 ``a.source IN (...)`` 片段与参数；空列表表示"不限制来源"。"""
    if not sources:
        return "1 = 1", []
    placeholders = ", ".join("?" * len(sources))
    return f"a.source IN ({placeholders})", list(sources)


# ────────────────────────────── 时间趋势 ──────────────────────────────
def trends(repo: Repo, *, days: int = 30, bucket: str = "day") -> dict[str, Any]:
    """按 UTC 自然日/周/月分桶统计产量趋势。

    窗口 = 今天（UTC）往前 ``days`` 天，起点为 ``今天 - (days - 1)`` 的 00:00 UTC；
    ``bucket`` 只能是 ``day``/``week``/``month``，非法值抛 ``ValueError``（``days < 1`` 同样抛错）。
    空桶以 0 值出现，保证前端可直接画折线。

    计数口径：
      * ``images``：``images.created_at`` 分桶，``duplicate_of IS NULL``；
      * ``boxes``：``annotations.created_at`` 分桶，``deleted_at IS NULL`` 且 ``source <> 'model'``；
      * ``annotated_tasks``：``tasks.submitted_at`` 分桶；
      * ``approved_tasks``/``rejected_tasks``：``tasks.reviewed_at`` 分桶且状态分别为
        ``approved``/``rejected``（注意：``Repo.add_review()`` 打回后把任务置回 ``annotating``，
        因此除非显式 ``set_task_status(..., 'rejected')``，打回数通常为 0，见本模块 docstring 说明）。
    """
    if bucket not in BUCKETS:
        raise ValueError(f"bucket 必须是 {'/'.join(BUCKETS)} 之一，收到 {bucket!r}")
    if days < 1:
        raise ValueError(f"days 必须为正整数，收到 {days!r}")

    today = datetime.now(timezone.utc).date()
    start = today - timedelta(days=days - 1)
    first, last = start.isoformat(), today.isoformat()

    buckets: dict[str, dict[str, Any]] = {
        label: {"key": label, "images": 0, "boxes": 0, "annotated_tasks": 0,
                "approved_tasks": 0, "rejected_tasks": 0, "by_class": {}}
        for label in _bucket_labels(start, today, bucket)
    }

    def bucket_at(text: Any) -> dict[str, Any] | None:
        if not text:
            return None
        try:
            day = date.fromisoformat(str(text)[:10])
        except ValueError:  # 非 ISO 文本（历史脏数据）不得让报表崩溃
            return None
        return buckets.get(_bucket_label(day, bucket))

    window = "substr({column}, 1, 10) BETWEEN ? AND ?"
    queries: list[tuple[str, tuple[Any, ...], str, bool]] = [
        (f"SELECT substr(created_at, 1, 10) AS d, COUNT(*) AS n FROM images "
         f"WHERE duplicate_of IS NULL AND {window.format(column='created_at')} GROUP BY d",
         (first, last), "images", False),
        (f"SELECT substr(a.created_at, 1, 10) AS d, a.class_code AS cls, COUNT(*) AS n "
         f"FROM annotations a WHERE a.deleted_at IS NULL AND a.source <> 'model' "
         f"AND {window.format(column='a.created_at')} GROUP BY d, cls",
         (first, last), "boxes", True),
        (f"SELECT substr(submitted_at, 1, 10) AS d, COUNT(*) AS n FROM tasks "
         f"WHERE submitted_at IS NOT NULL AND {window.format(column='submitted_at')} GROUP BY d",
         (first, last), "annotated_tasks", False),
        # 复核：通过按 tasks（状态确实是 approved），打回按 reviews（打回会把任务放回 annotating，
        # 只按 tasks.status='rejected' 统计会永远是 0 —— 那是"打回后又被改好"的中间态，不是事实来源）
        (f"SELECT substr(reviewed_at, 1, 10) AS d, status, COUNT(*) AS n FROM tasks "
         f"WHERE status = 'approved' AND reviewed_at IS NOT NULL "
         f"AND {window.format(column='reviewed_at')} GROUP BY d, status",
         (first, last), "review", False),
        (f"SELECT substr(created_at, 1, 10) AS d, decision AS status, COUNT(*) AS n FROM reviews "
         f"WHERE decision = 'reject' AND {window.format(column='created_at')} GROUP BY d, status",
         (first, last), "review", False),
    ]
    for sql, params, field, grouped_by_class in queries:
        for row in repo.conn.execute(sql, params):
            item = bucket_at(row["d"])
            if item is None:
                continue
            if field == "review":
                if row["status"] == "approved":
                    item["approved_tasks"] += int(row["n"])
                else:
                    item["rejected_tasks"] += int(row["n"])
            elif grouped_by_class:
                code = row["cls"]
                item["by_class"][code] = item["by_class"].get(code, 0) + int(row["n"])
                item["boxes"] += int(row["n"])
            else:
                item[field] += int(row["n"])

    ordered = [buckets[label] for label in _bucket_labels(start, today, bucket)]
    totals: dict[str, Any] = {"images": 0, "boxes": 0, "annotated_tasks": 0,
                              "approved_tasks": 0, "rejected_tasks": 0, "by_class": {}}
    for item in ordered:
        for field in ("images", "boxes", "annotated_tasks", "approved_tasks", "rejected_tasks"):
            totals[field] += item[field]
        for code, count in item["by_class"].items():
            totals["by_class"][code] = totals["by_class"].get(code, 0) + count
    totals["by_class"] = dict(sorted(totals["by_class"].items(), key=lambda kv: (-kv[1], kv[0])))
    return {"bucket": bucket, "days": days, "start": first, "end": last,
            "buckets": ordered, "totals": totals}


# ────────────────────────────── 批次对比 ──────────────────────────────
def batch_comparison(repo: Repo, *, limit: int = 20) -> list[dict[str, Any]]:
    """按导入批次横向对比，最新批次在前（``batches.id DESC``）。

    ``images``/``duplicates`` 按该批次图像数（``duplicate_of`` 是否为空）拆分；
    ``boxes``/``labeled_images``/``by_class`` 统计该批次图像上"非模型候选且未软删"的标注，
    口径为**该批次全部图像**（含近似重复行——重复影像同样会被导入流程建任务并可能被标注，
    便于如实反映该批次产出）；因此 ``avg_boxes_per_image`` 可能大于"非重复图像上的框数/影像数"；
    ``import_stats`` 直接取 ``batches.stats_json``（解析失败或非对象给 ``{}``）；
    ``avg_boxes_per_image`` = ``boxes / images``（``images = 0`` 时 0.0，保留 4 位小数）。
    """
    if limit < 1:
        raise ValueError(f"limit 必须为正整数，收到 {limit!r}")
    rows = repo.conn.execute(
        "SELECT id, kind, source, note, stats_json, created_at FROM batches ORDER BY id DESC LIMIT ?",
        (limit,),
    ).fetchall()
    items: list[dict[str, Any]] = []
    for row in rows:
        batch_id = int(row["id"])
        counts = repo.conn.execute(
            "SELECT COUNT(*) AS total, SUM(CASE WHEN duplicate_of IS NOT NULL THEN 1 ELSE 0 END) AS dup "
            "FROM images WHERE batch_id = ?", (batch_id,)).fetchone()
        total_images = int(counts["total"])
        duplicates = int(counts["dup"] or 0)
        images = total_images - duplicates
        labeled = repo.conn.execute(
            "SELECT COUNT(*) AS boxes, COUNT(DISTINCT a.image_id) AS labeled FROM annotations a "
            "JOIN images i ON i.id = a.image_id "
            "WHERE i.batch_id = ? AND i.duplicate_of IS NULL "
            "AND a.deleted_at IS NULL AND a.source <> 'model'",
            (batch_id,)).fetchone()
        boxes = int(labeled["boxes"])
        by_class = {
            str(item["class_code"]): int(item["n"]) for item in repo.conn.execute(
                "SELECT a.class_code, COUNT(*) AS n FROM annotations a JOIN images i ON i.id = a.image_id "
                "WHERE i.batch_id = ? AND i.duplicate_of IS NULL "
                "AND a.deleted_at IS NULL AND a.source <> 'model' "
                "GROUP BY a.class_code", (batch_id,))
        }
        approved_tasks = int(repo.conn.execute(
            "SELECT COUNT(*) AS n FROM tasks t JOIN images i ON i.id = t.image_id "
            "WHERE i.batch_id = ? AND t.status = 'approved'", (batch_id,)).fetchone()["n"])
        items.append({
            "batch_id": batch_id,
            "kind": str(row["kind"]),
            "source": row["source"],
            "note": row["note"],
            "created_at": str(row["created_at"]),
            "images": images,
            "duplicates": duplicates,
            "boxes": boxes,
            "labeled_images": int(labeled["labeled"]),
            "approved_tasks": approved_tasks,
            "by_class": dict(sorted(by_class.items(), key=lambda kv: (-kv[1], kv[0]))),
            "avg_boxes_per_image": round(boxes / images, 4) if images else 0.0,
            "import_stats": _parse_json_object(row["stats_json"]),
        })
    return items


# ────────────────────────── 类别覆盖与进度 ──────────────────────────
def coverage(repo: Repo) -> dict[str, Any]:
    """类别覆盖、标注来源分布与复核进度。

    说明：
      * ``classes`` 覆盖 ``classes`` 表的**全部**类别（含 0 标注的预留类别），按 ``order_index, code`` 排序；
      * ``approved_images`` = 有该类"任务已通过（``tasks.status='approved'``）且非 model 候选"标注的图像数；
      * ``totals.images`` 为 ``images`` 全表行数（含近似重复行，便于与批次表对照）；
      * ``totals.tasks`` 与 :meth:`Repo.task_progress` 一致，只列出库中**实际存在**的状态键（缺失即 0）；
      * ``review`` 按 ``reviews.decision`` 汇总（``approve_ratio = approved / (approved + rejected)``，
        无复核记录时为 ``None``，不编造 0/1）；
      * ``sources`` 是固定键集合，库内 ``external`` 计入文档口径的 ``import`` 桶（见模块 docstring）。
    """
    classes = [dict(row) for row in repo.conn.execute(
        "SELECT code, name_zh, name_en, active, is_crack FROM classes ORDER BY order_index, code")]

    def grouped(sql: str, params: tuple[Any, ...] = ()) -> dict[str, int]:
        return {str(row["class_code"]): int(row["n"]) for row in repo.conn.execute(sql, params)}

    boxes_by_class = grouped(
        "SELECT class_code, COUNT(*) AS n FROM annotations "
        "WHERE deleted_at IS NULL AND source <> 'model' GROUP BY class_code")
    candidates_by_class = grouped(
        "SELECT class_code, COUNT(*) AS n FROM annotations "
        "WHERE deleted_at IS NULL AND source = 'model' GROUP BY class_code")
    adopted_by_class = grouped(
        "SELECT class_code, COUNT(*) AS n FROM annotations "
        "WHERE deleted_at IS NULL AND source = 'model_edited' GROUP BY class_code")
    approved_by_class = grouped(
        "SELECT a.class_code AS class_code, COUNT(DISTINCT a.image_id) AS n FROM annotations a "
        "JOIN tasks t ON t.id = a.task_id WHERE a.deleted_at IS NULL AND a.source <> 'model' "
        "AND t.status = 'approved' GROUP BY a.class_code")

    class_items = [{
        "code": str(row["code"]),
        "name_zh": str(row["name_zh"]),
        "name_en": str(row["name_en"]),
        "active": bool(row["active"]),
        "is_crack": bool(row["is_crack"]),
        "approved_images": approved_by_class.get(str(row["code"]), 0),
        "boxes": boxes_by_class.get(str(row["code"]), 0),
        "model_candidates": candidates_by_class.get(str(row["code"]), 0),
        "adopted": adopted_by_class.get(str(row["code"]), 0),
    } for row in classes]

    boxes_total = int(repo.conn.execute(
        "SELECT COUNT(*) AS n FROM annotations WHERE deleted_at IS NULL AND source <> 'model'"
    ).fetchone()["n"])
    tasks = repo.task_progress()
    sources = {name: 0 for name in ("human", "model", "model_edited", "import")}
    for row in repo.conn.execute(
            "SELECT source, COUNT(*) AS n FROM annotations WHERE deleted_at IS NULL GROUP BY source"):
        name = str(row["source"])
        key = "import" if name in _SOURCE_ALIASES["import"] else name
        sources[key] = sources.get(key, 0) + int(row["n"])

    decisions = {str(row["decision"]): int(row["n"]) for row in repo.conn.execute(
        "SELECT decision, COUNT(*) AS n FROM reviews GROUP BY decision")}
    approved = decisions.get("approve", 0)
    rejected = decisions.get("reject", 0)
    reviewed = approved + rejected
    return {
        "classes": class_items,
        "totals": {
            "images": repo.count_images(),
            "tasks": tasks,
            "boxes": boxes_total,
            "pending_tasks": int(tasks.get("pending", 0)),
            "approved_tasks": int(tasks.get("approved", 0)),
        },
        "review": {"approved": approved, "rejected": rejected,
                   "approve_ratio": round(approved / reviewed, 4) if reviewed else None},
        "sources": sources,
    }


# ───────────────────────────── GIS 字段导出 ─────────────────────────────
def _annotation_metric_columns(conn: sqlite3.Connection) -> tuple[str, ...]:
    """探测 ``annotations`` 表是否已有派生指标列（当前 schema 没有，见模块 docstring）。"""
    try:
        present = {str(row["name"]) for row in conn.execute("PRAGMA table_info(annotations)")}
    except sqlite3.Error:
        return ()
    return tuple(name for name in _METRIC_KEYS if name in present)


def _metrics_of(row: sqlite3.Row, columns: tuple[str, ...], bbox: list[float] | None,
                kind: str) -> tuple[float | None, float | None, float | None]:
    """取 ``area_ratio``/``length_px``/``width_px``：列 → geometry_json → bbox 面积。"""
    metrics = _parse_json_object(row["geometry_json"])
    values: dict[str, float] = {}
    for name in _METRIC_KEYS:
        if name in columns and row[name] is not None:
            values[name] = float(row[name])
        elif isinstance(metrics.get(name), (int, float)) and not isinstance(metrics[name], bool):
            values[name] = float(metrics[name])
    if "area_ratio" not in values and kind == "bbox" and bbox is not None:
        values["area_ratio"] = round(max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1]), 6)
    return values.get("area_ratio"), values.get("length_px"), values.get("width_px")


def gis_features(repo: Repo, *, classes: Sequence[str] | None = None,
                 sources: Sequence[str] | None = None, since: str | None = None,
                 until: str | None = None, limit: int = 20000,
                 bbox: tuple[float, float, float, float] | None = None) -> dict[str, Any]:
    """导出带 GPS 的检测点（GeoJSON/CSV 用）。

    过滤条件：图像有 GPS（``gps_lat``/``gps_lon`` 均非空）、``duplicate_of IS NULL``、
    标注 ``deleted_at IS NULL``；``sources`` 缺省为 :data:`DEFAULT_GIS_SOURCES`（即排除 ``model`` 候选，
    传 ``sources=("model",)`` 可显式包含）；``classes`` 按 ``class_code`` 过滤；
    ``since``/``until``（``YYYY-MM-DD``，闭区间）按 ``COALESCE(images.captured_at, images.created_at)`` 过滤；
    ``bbox`` = ``(min_lon, min_lat, max_lon, max_lat)`` 空间过滤。

    返回 ``count`` = 实际导出的要素数（= ``len(features)``）；被 ``limit`` 截断时 ``truncated=True``；
    ``without_gps`` = 命中类别/来源/时间过滤但**没有 GPS** 的标注数（bbox 只作用于有 GPS 的点，
    故该计数不受 ``bbox`` 影响）。缺 bbox 坐标的标注（schema 只对 ``kind='bbox'`` 强制坐标齐全）
    跳过导出，且不计入 ``without_gps``。排序稳定：``image_id, annotation id``。
    """
    if limit < 0:
        raise ValueError(f"limit 不能为负数，收到 {limit!r}")
    since_date = _parse_date_arg("since", since)
    until_date = _parse_date_arg("until", until)
    if since_date is not None and until_date is not None and since_date > until_date:
        raise ValueError(f"since({since_date.isoformat()}) 不能晚于 until({until_date.isoformat()})")
    if bbox is not None:
        box_values = tuple(float(value) for value in bbox)
        if len(box_values) != 4:
            raise ValueError(f"bbox 必须是 (min_lon, min_lat, max_lon, max_lat)，收到 {bbox!r}")
        min_lon, min_lat, max_lon, max_lat = box_values
        if min_lon > max_lon or min_lat > max_lat:
            raise ValueError(f"bbox 的最小值不能大于最大值：{bbox!r}")

    source_sql, source_params = _source_clause(_expand_sources(sources))
    base_where = ["a.deleted_at IS NULL", "i.duplicate_of IS NULL", source_sql]
    base_params: list[Any] = list(source_params)
    if classes:
        base_where.append(f"a.class_code IN ({', '.join('?' * len(classes))})")
        base_params.extend(classes)
    if since_date is not None:
        base_where.append("substr(COALESCE(i.captured_at, i.created_at), 1, 10) >= ?")
        base_params.append(since_date.isoformat())
    if until_date is not None:
        base_where.append("substr(COALESCE(i.captured_at, i.created_at), 1, 10) <= ?")
        base_params.append(until_date.isoformat())
    spatial_where = [*base_where, "i.gps_lat IS NOT NULL AND i.gps_lon IS NOT NULL"]
    spatial_params = list(base_params)
    if bbox is not None:
        spatial_where.append("i.gps_lon >= ? AND i.gps_lon <= ? AND i.gps_lat >= ? AND i.gps_lat <= ?")
        spatial_params.extend([min_lon, max_lon, min_lat, max_lat])

    metric_columns = _annotation_metric_columns(repo.conn)
    extra = "".join(f", a.{name}" for name in metric_columns)
    sql = (f"SELECT a.id AS annotation_id, a.image_id, a.task_id, a.class_code, a.kind, a.source, "
           f"a.score, a.difficult, a.bbox_x1, a.bbox_y1, a.bbox_x2, a.bbox_y2, a.geometry_json, "
           f"i.path, i.captured_at, i.gps_lat, i.gps_lon, i.batch_id{extra} "
           f"FROM annotations a JOIN images i ON i.id = a.image_id "
           f"WHERE {' AND '.join(spatial_where)} ORDER BY a.image_id, a.id LIMIT ?")
    rows = repo.conn.execute(sql, [*spatial_params, limit + 1]).fetchall()
    truncated = len(rows) > limit
    features: list[dict[str, Any]] = []
    for row in rows[:limit]:
        coords = [row["bbox_x1"], row["bbox_y1"], row["bbox_x2"], row["bbox_y2"]]
        if any(value is None for value in coords):
            continue
        box = [float(value) for value in coords]
        area_ratio, length_px, width_px = _metrics_of(row, metric_columns, box, str(row["kind"]))
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [float(row["gps_lon"]), float(row["gps_lat"])]},
            "properties": {
                "image_id": int(row["image_id"]),
                "task_id": int(row["task_id"]),
                "path": str(row["path"]),
                "captured_at": row["captured_at"],
                "class_code": str(row["class_code"]),
                "source": str(row["source"]),
                "score": None if row["score"] is None else float(row["score"]),
                "difficult": bool(row["difficult"]),
                "bbox": box,
                "length_px": length_px,
                "width_px": width_px,
                "area_ratio": area_ratio,
                "batch_id": None if row["batch_id"] is None else int(row["batch_id"]),
            },
        })
    missing_sql = (f"SELECT COUNT(*) AS n FROM annotations a JOIN images i ON i.id = a.image_id "
                   f"WHERE {' AND '.join(base_where)} "
                   f"AND (i.gps_lat IS NULL OR i.gps_lon IS NULL)")
    without_gps = int(repo.conn.execute(missing_sql, base_params).fetchone()["n"])
    return {"features": features, "count": len(features),
            "without_gps": without_gps, "truncated": truncated}


def to_geojson(payload: dict[str, Any], *, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    """把 :func:`gis_features` 的结果转成 GeoJSON ``FeatureCollection``。

    ``properties`` = 传入的 ``metadata`` + ``count`` + ``without_gps``（后两者以 payload 为准）。
    """
    features = [dict(feature) for feature in payload.get("features") or []]
    properties: dict[str, Any] = dict(metadata or {})
    properties["count"] = int(payload.get("count", len(features)))
    properties["without_gps"] = int(payload.get("without_gps") or 0)
    return {"type": "FeatureCollection", "features": features, "properties": properties}


def to_gis_csv(payload: dict[str, Any]) -> str:
    """把 :func:`gis_features` 的结果转成 CSV 文本（表头固定、``\\n`` 结尾、空值输出空串）。"""
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(_GEOJSON_CSV_HEADER)
    for feature in payload.get("features") or []:
        properties = feature.get("properties") or {}
        coordinates = (feature.get("geometry") or {}).get("coordinates") or [None, None]
        box = list(properties.get("bbox") or [None, None, None, None])
        writer.writerow([
            _csv_cell(properties.get("image_id")), _csv_cell(properties.get("path")),
            _csv_cell(properties.get("captured_at")), _csv_cell(coordinates[1]),
            _csv_cell(coordinates[0]), _csv_cell(properties.get("class_code")),
            _csv_cell(properties.get("source")), _csv_cell(properties.get("score")),
            _csv_cell(box[0]), _csv_cell(box[1]), _csv_cell(box[2]), _csv_cell(box[3]),
            _csv_cell(properties.get("length_px")), _csv_cell(properties.get("width_px")),
            _csv_cell(properties.get("area_ratio")),
        ])
    return buffer.getvalue()


def _csv_cell(value: Any) -> Any:
    """CSV 单元格：数值保留原精度，缺失值输出空串。"""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (int, float)):
        return value if isinstance(value, int) else repr(float(value))
    return str(value)


# ────────────────────────────── 汇总报表 ──────────────────────────────
def _gps_gap(repo: Repo) -> dict[str, int]:
    """全库（缺省导出过滤条件下）"可导出 GPS 点"与"缺少 GPS 点"的数量。"""
    source_sql, source_params = _source_clause(_expand_sources(None))
    condition = (f"a.deleted_at IS NULL AND i.duplicate_of IS NULL AND {source_sql}")
    row = repo.conn.execute(
        f"SELECT COUNT(*) AS total, "
        f"SUM(CASE WHEN i.gps_lat IS NOT NULL AND i.gps_lon IS NOT NULL THEN 1 ELSE 0 END) AS located "
        f"FROM annotations a JOIN images i ON i.id = a.image_id WHERE {condition}",
        source_params).fetchone()
    total = int(row["total"])
    located = int(row["located"] or 0)
    return {"exportable": located, "without_gps": total - located}


def summary(repo: Repo, *, days: int = 30, bucket: str = "day", batch_limit: int = 20) -> dict[str, Any]:
    """一次性汇总趋势、批次对比、类别覆盖与 GPS 覆盖（报表/接口的单一入口）。

    除文档约定的 ``generated_at``/``trends``/``batches``/``coverage`` 外，
    额外给出 ``gis`` = ``{"exportable": 可导出 GPS 点数, "without_gps": 缺少 GPS 点数}``，
    供 :func:`render_markdown` 输出"缺少 GPS 的点"这一行（GDPR/诚实报告的硬要求）。
    """
    return {
        "generated_at": utc_now(),
        "trends": trends(repo, days=days, bucket=bucket),
        "batches": batch_comparison(repo, limit=batch_limit),
        "coverage": coverage(repo),
        "gis": _gps_gap(repo),
    }


def _md_int(value: Any) -> str:
    """整数单元格；缺失值写 ``—``。"""
    if value is None:
        return _MISSING
    try:
        return str(int(value))
    except (TypeError, ValueError):
        return _MISSING


def _md_ratio(value: Any, digits: int = 3) -> str:
    """比例单元格（默认 3 位小数）；缺失值写 ``—``。"""
    if value is None:
        return _MISSING
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return _MISSING


def _md_percent(value: Any) -> str:
    """百分比单元格（1 位小数）；缺失值写 ``—``。"""
    if value is None:
        return _MISSING
    try:
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return _MISSING


def _md_text(value: Any) -> str:
    """文本单元格：空值写 ``—``，并转义 Markdown 表格分隔符。"""
    if value is None or value == "":
        return _MISSING
    return str(value).replace("|", "\\|").replace("\n", " ")


def render_markdown(summary: dict[str, Any]) -> str:
    """把 :func:`summary` 的结果渲染为中文 Markdown 报表。

    约定：趋势表只列**数值非零的桶**并在末尾给出总计；比例保留 3 位小数、比率用百分比 1 位小数；
    缺失值统一写 ``—``；最后一行固定为"缺少 GPS 的点：N"。
    """
    trends_payload = summary.get("trends") or {}
    batches = summary.get("batches") or []
    coverage_payload = summary.get("coverage") or {}
    gis_payload = summary.get("gis") or {}
    totals = trends_payload.get("totals") or {}

    names = {str(item.get("code")): str(item.get("name_zh") or item.get("code"))
             for item in coverage_payload.get("classes") or []}

    lines: list[str] = ["# 路检统计报表", ""]
    lines.append(f"- 生成时间：{_md_text(summary.get('generated_at'))}")
    lines.append(f"- 趋势窗口：{_md_text(trends_payload.get('start'))} ~ "
                 f"{_md_text(trends_payload.get('end'))}"
                 f"（{_md_int(trends_payload.get('days'))} 天，按 {_md_text(trends_payload.get('bucket'))} 分桶）")
    lines.append("")

    lines.append("## 1. 时间趋势")
    lines.append("")
    lines.append("| 桶 | 影像 | 标注框 | 已提交任务 | 已通过任务 | 已打回任务 |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    fields = ("images", "boxes", "annotated_tasks", "approved_tasks", "rejected_tasks")
    nonzero = [item for item in trends_payload.get("buckets") or []
               if any(int(item.get(field) or 0) for field in fields)]
    for item in nonzero:
        cells = " | ".join(_md_int(item.get(field)) for field in fields)
        lines.append(f"| {_md_text(item.get('key'))} | {cells} |")
    total_cells = " | ".join(_md_int(totals.get(field)) for field in fields)
    lines.append(f"| **总计** | {total_cells} |")
    if not nonzero:
        lines.append("")
        lines.append("（窗口内没有非零桶）")
    by_class = totals.get("by_class") or {}
    if by_class:
        detail = "、".join(f"{names.get(code, code)} {count}" for code, count in by_class.items())
    else:
        detail = _MISSING
    lines.append("")
    lines.append(f"窗口内分类别框数：{detail}")
    lines.append("")

    lines.append("## 2. 批次对比（最新在前）")
    lines.append("")
    if batches:
        lines.append("| 批次 | 类型 | 来源 | 备注 | 创建时间 | 影像 | 重复 | 标注框 | 已标注影像 "
                     "| 已通过任务 | 平均框/图 | 导入统计 |")
        lines.append("|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---|")
        for item in batches:
            stats = item.get("import_stats") or {}
            stats_text = "、".join(f"{key}={value}" for key, value in sorted(stats.items()))
            lines.append(
                f"| #{_md_int(item.get('batch_id'))} | {_md_text(item.get('kind'))} "
                f"| {_md_text(item.get('source'))} | {_md_text(item.get('note'))} "
                f"| {_md_text(item.get('created_at'))} | {_md_int(item.get('images'))} "
                f"| {_md_int(item.get('duplicates'))} | {_md_int(item.get('boxes'))} "
                f"| {_md_int(item.get('labeled_images'))} | {_md_int(item.get('approved_tasks'))} "
                f"| {_md_ratio(item.get('avg_boxes_per_image'))} | {_md_text(stats_text)} |")
    else:
        lines.append("（无批次数据）")
    lines.append("")

    lines.append("## 3. 类别覆盖与标注进度")
    lines.append("")
    lines.append("| 类别 | 中文名 | 已复核图像数 | 框数 | 候选数 | 采纳数 |")
    lines.append("|---|---|---:|---:|---:|---:|")
    class_items = coverage_payload.get("classes") or []
    for item in class_items:
        lines.append(
            f"| {_md_text(item.get('code'))} | {_md_text(item.get('name_zh'))} "
            f"| {_md_int(item.get('approved_images'))} | {_md_int(item.get('boxes'))} "
            f"| {_md_int(item.get('model_candidates'))} | {_md_int(item.get('adopted'))} |")
    coverage_totals = coverage_payload.get("totals") or {}
    # 已复核图像数跨类别不可相加（同一图像可含多类），故总计行按约定写 "—"
    lines.append(f"| **总计** | {_MISSING} | {_MISSING} | {_md_int(coverage_totals.get('boxes'))} "
                 f"| {_md_int(sum(int(item.get('model_candidates') or 0) for item in class_items))} "
                 f"| {_md_int(sum(int(item.get('adopted') or 0) for item in class_items))} |")
    lines.append("")

    lines.append("## 4. 复核与导出")
    lines.append("")
    tasks = coverage_totals.get("tasks") or {}
    task_text = "、".join(f"{status} {count}" for status, count in sorted(tasks.items())) or _MISSING
    review = coverage_payload.get("review") or {}
    sources = coverage_payload.get("sources") or {}
    source_text = "、".join(f"{name} {count}" for name, count in sources.items()) or _MISSING
    lines.append(f"- 影像总数：{_md_int(coverage_totals.get('images'))}｜任务状态：{task_text}")
    lines.append(f"- 待标注任务：{_md_int(coverage_totals.get('pending_tasks'))}"
                 f"｜已通过任务：{_md_int(coverage_totals.get('approved_tasks'))}")
    lines.append(f"- 标注来源分布：{source_text}")
    lines.append(f"- 复核通过率：{_md_ratio(review.get('approve_ratio'))}"
                 f"（{_md_percent(review.get('approve_ratio'))}）"
                 f"｜通过 {_md_int(review.get('approved'))}｜打回 {_md_int(review.get('rejected'))}")
    lines.append(f"- 可导出 GPS 点：{_md_int(gis_payload.get('exportable'))}")
    lines.append("")
    lines.append(f"缺少 GPS 的点：{_md_int(gis_payload.get('without_gps'))}")
    return "\n".join(lines) + "\n"
