"""仓储层：所有 SQL 访问的唯一入口（事务、幂等、审计）。

约定：
  * 入参/出参使用普通 dict（sqlite3.Row → dict），不向调用方泄露连接细节；
  * 写操作在 :func:`transaction` 内完成；
  * 标注采用「软删 + 差异审计」，模型候选（source='model'）不被人工提交覆盖。
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any, Iterable, Sequence

from .db import transaction
from .files import utc_now


def _dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    return dict(row) if row is not None else None


def _dicts(rows: Iterable[sqlite3.Row]) -> list[dict[str, Any]]:
    return [dict(row) for row in rows]


class Repo:
    """薄封装：一个连接 + 一组方法。线程内使用（FastAPI 依赖注入时每请求一个连接）。"""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn

    # ────────────────────────────── 类别 ──────────────────────────────
    def list_classes(self, *, active_only: bool = False) -> list[dict[str, Any]]:
        sql = "SELECT * FROM classes"
        if active_only:
            sql += " WHERE active = 1"
        sql += " ORDER BY order_index, code"
        return _dicts(self.conn.execute(sql))

    def get_class(self, code: str) -> dict[str, Any] | None:
        return _dict(self.conn.execute("SELECT * FROM classes WHERE code = ?", (code,)).fetchone())

    def add_class(self, code: str, name_zh: str, name_en: str, *, color: str = "#e6194b",
                  parent_code: str | None = None, is_crack: bool = False,
                  order_index: int = 100, actor: str = "system") -> dict[str, Any]:
        with transaction(self.conn):
            self.conn.execute(
                "INSERT INTO classes(code, name_zh, name_en, color, parent_code, is_crack, order_index) "
                "VALUES(?,?,?,?,?,?,?)",
                (code, name_zh, name_en, color, parent_code, int(is_crack), order_index),
            )
            self._audit("class", code, "create", actor, None, {
                "code": code, "name_zh": name_zh, "order_index": order_index})
        created = self.get_class(code)
        assert created is not None
        return created

    def set_class_active(self, code: str, active: bool, *, actor: str = "system") -> bool:
        before = self.get_class(code)
        if before is None:
            return False
        with transaction(self.conn):
            self.conn.execute("UPDATE classes SET active = ? WHERE code = ?", (int(active), code))
            self._audit("class", code, "update", actor, {"active": before["active"]}, {"active": int(active)})
        return True

    # ────────────────────────── 批次与导入项 ──────────────────────────
    def create_batch(self, kind: str, source: str | None = None, note: str | None = None,
                     created_by: str | None = None) -> int:
        with transaction(self.conn):
            cur = self.conn.execute(
                "INSERT INTO batches(kind, source, note, created_by) VALUES(?,?,?,?)",
                (kind, source, note, created_by),
            )
            batch_id = int(cur.lastrowid)
        return batch_id

    def add_ingest_item(self, batch_id: int, source_path: str, outcome: str,
                        reason: str | None = None, image_id: int | None = None) -> None:
        self.conn.execute(
            "INSERT INTO ingest_items(batch_id, source_path, outcome, reason, image_id) VALUES(?,?,?,?,?)",
            (batch_id, source_path, outcome, reason, image_id),
        )

    def audit_ingest_items(self, batch_id: int, source_path: str, outcome: str,
                           reason: str | None = None, image_id: int | None = None) -> None:
        """批量导入时在一个事务里累积（调用方负责事务边界）。"""
        self.conn.execute(
            "INSERT INTO ingest_items(batch_id, source_path, outcome, reason, image_id) VALUES(?,?,?,?,?)",
            (batch_id, source_path, outcome, reason, image_id),
        )

    def update_batch_stats(self, batch_id: int, stats: dict[str, int]) -> None:
        self.conn.execute(
            "UPDATE batches SET stats_json = ? WHERE id = ?",
            (json.dumps(stats, ensure_ascii=False, sort_keys=True), batch_id),
        )

    def get_batch(self, batch_id: int) -> dict[str, Any] | None:
        return _dict(self.conn.execute("SELECT * FROM batches WHERE id = ?", (batch_id,)).fetchone())

    def batch_outcome_counts(self, batch_id: int) -> dict[str, int]:
        rows = self.conn.execute(
            "SELECT outcome, COUNT(*) AS n FROM ingest_items WHERE batch_id = ? GROUP BY outcome", (batch_id,)
        ).fetchall()
        return {row["outcome"]: row["n"] for row in rows}

    def batch_failures(self, batch_id: int, limit: int = 50) -> list[dict[str, Any]]:
        return _dicts(self.conn.execute(
            "SELECT source_path, outcome, reason FROM ingest_items "
            "WHERE batch_id = ? AND outcome IN ('error','skipped_type','skipped_size') LIMIT ?",
            (batch_id, limit),
        ))

    def list_batches(self, limit: int = 50, cursor: int | None = None) -> list[dict[str, Any]]:
        if cursor:
            rows = self.conn.execute("SELECT * FROM batches WHERE id < ? ORDER BY id DESC LIMIT ?", (cursor, limit))
        else:
            rows = self.conn.execute("SELECT * FROM batches ORDER BY id DESC LIMIT ?", (limit,))
        return _dicts(rows)

    # ────────────────────────────── 影像 ──────────────────────────────
    def find_image_by_sha(self, sha256: str) -> dict[str, Any] | None:
        return _dict(self.conn.execute("SELECT * FROM images WHERE sha256 = ?", (sha256,)).fetchone())

    def insert_image(self, *, batch_id: int | None, path: str, sha256: str, phash: str | None,
                     width: int, height: int, bytes_: int, source_kind: str,
                     parent_image_id: int | None = None, tile_json: str | None = None,
                     captured_at: str | None = None, gps_lat: float | None = None,
                     gps_lon: float | None = None, gps_source: str = "none",
                     device: str | None = None, quality_json: str | None = None) -> int:
        cur = self.conn.execute(
            """INSERT INTO images(batch_id, path, sha256, phash, width, height, bytes, source_kind,
                                  parent_image_id, tile_json, captured_at, gps_lat, gps_lon, gps_source,
                                  device, quality_json)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (batch_id, path, sha256, phash, width, height, bytes_, source_kind, parent_image_id,
             tile_json, captured_at, gps_lat, gps_lon, gps_source, device, quality_json),
        )
        return int(cur.lastrowid)

    def get_image(self, image_id: int) -> dict[str, Any] | None:
        return _dict(self.conn.execute("SELECT * FROM images WHERE id = ?", (image_id,)).fetchone())

    def list_phashes(self, *, same_batch_only: int | None = None, limit: int = 50_000) -> list[tuple[int, str]]:
        if same_batch_only is not None:
            rows = self.conn.execute(
                "SELECT id, phash FROM images WHERE phash IS NOT NULL AND batch_id = ? LIMIT ?",
                (same_batch_only, limit),
            )
        else:
            rows = self.conn.execute(
                "SELECT id, phash FROM images WHERE phash IS NOT NULL ORDER BY id DESC LIMIT ?", (limit,))
        return [(row["id"], row["phash"]) for row in rows]

    def list_images(self, *, batch_id: int | None = None, source_kind: str | None = None,
                    limit: int = 50, cursor: int | None = None) -> list[dict[str, Any]]:
        clauses, params = [], []
        if batch_id:
            clauses.append("batch_id = ?")
            params.append(batch_id)
        if source_kind:
            clauses.append("source_kind = ?")
            params.append(source_kind)
        if cursor:
            clauses.append("id < ?")
            params.append(cursor)
        sql = "SELECT * FROM images"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        return _dicts(self.conn.execute(sql, params))

    def count_images(self) -> int:
        return int(self.conn.execute("SELECT COUNT(*) AS n FROM images").fetchone()["n"])

    # ────────────────────────────── 任务 ──────────────────────────────
    def create_task(self, image_id: int, priority: int = 100) -> int:
        existing = self.conn.execute(
            "SELECT id FROM tasks WHERE image_id = ? ORDER BY id LIMIT 1", (image_id,)
        ).fetchone()
        if existing:
            return int(existing["id"])
        cur = self.conn.execute("INSERT INTO tasks(image_id, priority) VALUES(?,?)", (image_id, priority))
        return int(cur.lastrowid)

    def lease_tasks(self, *, count: int, status: str = "pending", assignee: str = "web",
                    lease_seconds: int = 1800) -> list[dict[str, Any]]:
        """原子领取任务：只领未被租约占用的行（并发安全）。"""
        now = utc_now()
        with transaction(self.conn):
            rows = self.conn.execute(
                """SELECT id FROM tasks
                   WHERE status = ? AND (lease_until IS NULL OR lease_until < ?)
                   ORDER BY priority ASC, id ASC LIMIT ?""",
                (status, now, count),
            ).fetchall()
            ids = [row["id"] for row in rows]
            for task_id in ids:
                self.conn.execute(
                    """UPDATE tasks
                       SET status = 'annotating', assignee = ?, attempts = attempts + 1,
                           lease_until = strftime('%Y-%m-%dT%H:%M:%SZ', 'now', ?)
                       WHERE id = ?""",
                    (assignee, f"+{int(lease_seconds)} seconds", task_id),
                )
            if not ids:
                return []
            placeholders = ",".join("?" * len(ids))
            leased = self.conn.execute(
                f"SELECT * FROM tasks WHERE id IN ({placeholders}) ORDER BY priority, id", ids
            ).fetchall()
        return _dicts(leased)

    def get_task(self, task_id: int) -> dict[str, Any] | None:
        return _dict(self.conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone())

    def get_task_detail(self, task_id: int) -> dict[str, Any] | None:
        task = self.get_task(task_id)
        if task is None:
            return None
        image = self.get_image(task["image_id"])
        task["image"] = image
        task["annotations"] = self.list_annotations(task_id=task_id)
        return task

    def list_tasks(self, *, status: str | None = None, batch_id: int | None = None,
                   class_code: str | None = None, strategy: str = "fifo",
                   limit: int = 50, cursor: int | None = None) -> list[dict[str, Any]]:
        clauses, params = [], []
        if status:
            clauses.append("t.status = ?")
            params.append(status)
        if batch_id:
            clauses.append("i.batch_id = ?")
            params.append(batch_id)
        if class_code:
            clauses.append("EXISTS (SELECT 1 FROM annotations a WHERE a.task_id = t.id AND a.class_code = ? AND a.deleted_at IS NULL)")
            params.append(class_code)
        if cursor:
            clauses.append("t.id > ?")
            params.append(cursor)
        order = {
            "priority": "t.priority ASC, t.id ASC",
            "low_conf": "t.priority ASC, t.id ASC",  # 预标注置信度排序在 M2 引入
            "random": "RANDOM()",
            "fifo": "t.id ASC",
        }.get(strategy, "t.id ASC")
        sql = """SELECT t.*, i.path AS image_path, i.source_kind
                 FROM tasks t JOIN images i ON i.id = t.image_id"""
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += f" ORDER BY {order} LIMIT ?"
        params.append(limit)
        return _dicts(self.conn.execute(sql, params))

    def set_task_status(self, task_id: int, status: str, *, actor: str = "system",
                        assignee: str | None = None, touch: str | None = None) -> bool:
        task = self.get_task(task_id)
        if task is None:
            return False
        fields, params = ["status = ?"], [status]
        if assignee is not None:
            fields.append("assignee = ?")
            params.append(assignee)
        if status == "annotated":
            fields.append("submitted_at = strftime('%Y-%m-%dT%H:%M:%SZ','now')")
        if status in ("approved", "rejected"):
            fields.append("reviewed_at = strftime('%Y-%m-%dT%H:%M:%SZ','now')")
        if touch:
            fields.append(touch)
        params.append(task_id)
        with transaction(self.conn):
            self.conn.execute(f"UPDATE tasks SET {', '.join(fields)} WHERE id = ?", params)
            self._audit("task", str(task_id), "update", actor, {"status": task["status"]}, {"status": status})
        return True

    def submit_task(self, task_id: int, *, actor: str = "web") -> dict[str, Any] | None:
        ok = self.set_task_status(task_id, "annotated", actor=actor)
        return self.get_task(task_id) if ok else None

    # ────────────────────────────── 标注 ──────────────────────────────
    def list_annotations(self, *, task_id: int | None = None, image_id: int | None = None,
                         include_deleted: bool = False, include_model: bool = True) -> list[dict[str, Any]]:
        clauses, params = [], []
        if task_id is not None:
            clauses.append("task_id = ?")
            params.append(task_id)
        if image_id is not None:
            clauses.append("image_id = ?")
            params.append(image_id)
        if not include_deleted:
            clauses.append("deleted_at IS NULL")
        if not include_model:
            clauses.append("source <> 'model'")
        sql = "SELECT * FROM annotations"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY id"
        rows = _dicts(self.conn.execute(sql, params))
        for row in rows:
            row["bbox"] = {
                "x1": row["bbox_x1"], "y1": row["bbox_y1"],
                "x2": row["bbox_x2"], "y2": row["bbox_y2"],
            }
        return rows

    def replace_annotations(self, task_id: int, items: Sequence[dict[str, Any]], *,
                            actor: str = "web", default_source: str = "human") -> dict[str, Any]:
        """全量提交人工标注：模型候选保持不动，人工标注按 (类别, 框) 差异更新。

        @returns {"annotations": [...], "changed": n, "added": n, "updated": n, "deleted": n}
        """
        task = self.get_task(task_id)
        if task is None:
            raise KeyError(f"task {task_id} 不存在")
        image_id = task["image_id"]
        current = [row for row in self.list_annotations(task_id=task_id, include_model=False)]
        key = lambda item: (item["class_code"], item["kind"], round(item["bbox_x1"], 6),  # noqa: E731
                            round(item["bbox_y1"], 6), round(item["bbox_x2"], 6), round(item["bbox_y2"], 6))
        current_by_key = {key(row): row for row in current}
        incoming_keys: set[tuple] = set()
        added = updated = deleted = 0
        now = utc_now()
        before_snapshot = [self._annotation_snapshot(row) for row in current]

        with transaction(self.conn):
            for item in items:
                bbox = item.get("bbox") or {}
                x1, y1, x2, y2 = (float(bbox.get(k, 0.0)) for k in ("x1", "y1", "x2", "y2"))
                if not (0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0):
                    raise ValueError(f"非法 bbox: {bbox}（要求 0<=x1<x2<=1 且 0<=y1<y2<=1）")
                cls = self.get_class(item["class_code"])
                if cls is None:
                    raise ValueError(f"未知类别: {item['class_code']}")
                row_key = (item["class_code"], item.get("kind", "bbox"),
                           round(x1, 6), round(y1, 6), round(x2, 6), round(y2, 6))
                incoming_keys.add(row_key)
                existing = current_by_key.get(row_key)
                source = item.get("source") or default_source
                if existing is None:
                    self.conn.execute(
                        """INSERT INTO annotations(task_id, image_id, class_code, kind,
                                                   bbox_x1, bbox_y1, bbox_x2, bbox_y2,
                                                   difficult, score, source, model_version_id)
                           VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
                        (task_id, image_id, item["class_code"], item.get("kind", "bbox"),
                         x1, y1, x2, y2, int(bool(item.get("difficult"))), item.get("score"),
                         source, item.get("model_version_id")),
                    )
                    added += 1
                else:
                    self.conn.execute(
                        """UPDATE annotations SET difficult = ?, deleted_at = NULL, updated_at = ?
                           WHERE id = ?""",
                        (int(bool(item.get("difficult"))), now, existing["id"]),
                    )
                    updated += 1
            for row_key, row in current_by_key.items():
                if row_key not in incoming_keys:
                    self.conn.execute("UPDATE annotations SET deleted_at = ? WHERE id = ?", (now, row["id"]))
                    deleted += 1
            changed = added + updated + deleted
            if changed:
                after_snapshot = [
                    {"class_code": item["class_code"], "bbox": item.get("bbox")} for item in items
                ]
                self._audit("annotation", str(task_id), "replace", actor,
                            {"count": len(before_snapshot), "items": before_snapshot[:50]},
                            {"count": len(after_snapshot), "items": after_snapshot[:50]})
            if task["status"] == "pending":
                self.conn.execute("UPDATE tasks SET status = 'annotating' WHERE id = ?", (task_id,))
        return {
            "annotations": self.list_annotations(task_id=task_id),
            "changed": changed, "added": added, "updated": updated, "deleted": deleted,
        }

    def adopt_model_candidates(self, task_id: int, *, ids: Sequence[int] | None = None,
                               actor: str = "web") -> int:
        """采纳模型候选（Z 键）：source model → model_edited，纳入人工管理。"""
        with transaction(self.conn):
            if ids:
                placeholders = ",".join("?" * len(ids))
                cur = self.conn.execute(
                    f"UPDATE annotations SET source = 'model_edited', updated_at = ? "
                    f"WHERE task_id = ? AND source = 'model' AND id IN ({placeholders})",
                    (utc_now(), task_id, *ids),
                )
            else:
                cur = self.conn.execute(
                    "UPDATE annotations SET source = 'model_edited', updated_at = ? "
                    "WHERE task_id = ? AND source = 'model'",
                    (utc_now(), task_id),
                )
            count = cur.rowcount or 0
            if count:
                self._audit("annotation", str(task_id), "adopt-model", actor, None, {"count": count})
        return count

    def add_model_candidates(self, task_id: int, items: Sequence[dict[str, Any]], *,
                             model_version_id: int | None = None) -> int:
        """写入预标注候选（source='model'）；已存在相同框则跳过。"""
        task = self.get_task(task_id)
        if task is None:
            raise KeyError(f"task {task_id}")
        existing = {
            (row["class_code"], round(row["bbox_x1"], 6), round(row["bbox_y1"], 6),
             round(row["bbox_x2"], 6), round(row["bbox_y2"], 6))
            for row in self.list_annotations(task_id=task_id, include_model=True)
        }
        inserted = 0
        with transaction(self.conn):
            for item in items:
                bbox = item["bbox"]
                key = (item["class_code"], round(bbox["x1"], 6), round(bbox["y1"], 6),
                       round(bbox["x2"], 6), round(bbox["y2"], 6))
                if key in existing:
                    continue
                self.conn.execute(
                    """INSERT INTO annotations(task_id, image_id, class_code, kind,
                                               bbox_x1, bbox_y1, bbox_x2, bbox_y2,
                                               score, source, model_version_id)
                       VALUES(?,?,?,'bbox',?,?,?,?,?,'model',?)""",
                    (task_id, task["image_id"], item["class_code"], bbox["x1"], bbox["y1"],
                     bbox["x2"], bbox["y2"], item.get("score"), model_version_id),
                )
                inserted += 1
            if inserted:
                self.conn.execute(
                    "UPDATE tasks SET prelabel_state = 'done', status = CASE WHEN status = 'pending' THEN 'prelabeled' ELSE status END WHERE id = ?",
                    (task_id,),
                )
        return inserted

    def add_mask_candidate(self, *, task_id: int, class_code: str, bbox: dict[str, float],
                           mask_path: str, metrics: dict[str, Any] | None = None,
                           source: str = "model", model_version_id: int | None = None) -> int:
        """写入一条掩膜标注（kind='mask'）；派生指标存 geometry_json，便于后续统计裂缝宽度。"""
        task = self.get_task(task_id)
        if task is None:
            raise KeyError(f"task {task_id}")
        with transaction(self.conn):
            cur = self.conn.execute(
                """INSERT INTO annotations(task_id, image_id, class_code, kind,
                                           bbox_x1, bbox_y1, bbox_x2, bbox_y2,
                                           geometry_json, mask_path, source, model_version_id)
                   VALUES(?,?,?, 'mask', ?,?,?,?,?,?,?,?)""",
                (task_id, task["image_id"], class_code, bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"],
                 json.dumps(metrics or {}, ensure_ascii=False, sort_keys=True), mask_path, source,
                 model_version_id),
            )
            annotation_id = int(cur.lastrowid)
            self._audit("annotation", str(task_id), "create-mask", source, None,
                        {"id": annotation_id, "class_code": class_code, "mask": mask_path,
                         "metrics": metrics or {}})
        return annotation_id

    def mask_annotations(self, *, task_id: int | None = None, image_id: int | None = None,
                         limit: int = 200) -> list[dict[str, Any]]:
        """待复核的掩膜列表（含派生指标）。"""
        clauses = ["deleted_at IS NULL", "kind = 'mask'"]
        params: list[Any] = []
        if task_id is not None:
            clauses.append("task_id = ?")
            params.append(task_id)
        if image_id is not None:
            clauses.append("image_id = ?")
            params.append(image_id)
        params.append(limit)
        rows = _dicts(self.conn.execute(
            f"SELECT * FROM annotations WHERE {' AND '.join(clauses)} ORDER BY id LIMIT ?", params))
        for row in rows:
            try:
                row["mask_metrics"] = json.loads(row["geometry_json"] or "{}")
            except json.JSONDecodeError:
                row["mask_metrics"] = {}
        return rows

    def set_prelabel_state(self, task_id: int, state: str) -> None:
        self.conn.execute("UPDATE tasks SET prelabel_state = ? WHERE id = ?", (state, task_id))

    def class_counts(self) -> list[dict[str, Any]]:
        return _dicts(self.conn.execute(
            """SELECT a.class_code, c.name_zh, COUNT(*) AS n
               FROM annotations a JOIN classes c ON c.code = a.class_code
               WHERE a.deleted_at IS NULL GROUP BY a.class_code, c.name_zh ORDER BY n DESC"""
        ))

    # ────────────────────────────── 复核 ──────────────────────────────
    def add_review(self, task_id: int, *, decision: str, reason_code: str | None = None,
                   note: str | None = None, reviewer: str | None = None) -> dict[str, Any]:
        if decision not in ("approve", "reject"):
            raise ValueError("decision 必须是 approve 或 reject")
        if decision == "reject" and reason_code is None:
            raise ValueError("打回必须提供 reason_code")
        with transaction(self.conn):
            cur = self.conn.execute(
                "INSERT INTO reviews(task_id, reviewer, decision, reason_code, note) VALUES(?,?,?,?,?)",
                (task_id, reviewer, decision, reason_code, note),
            )
            review_id = int(cur.lastrowid)
            next_status = "approved" if decision == "approve" else "annotating"
            self.conn.execute(
                """UPDATE tasks SET status = ?, reviewed_at = strftime('%Y-%m-%dT%H:%M:%SZ','now')
                   WHERE id = ?""",
                (next_status, task_id),
            )
            self._audit("review", str(task_id), decision, reviewer or "system", None,
                        {"review_id": review_id, "reason_code": reason_code, "status": next_status})
        return _dict(self.conn.execute("SELECT * FROM reviews WHERE id = ?", (review_id,)).fetchone())  # type: ignore[return-value]

    def review_stats(self) -> dict[str, Any]:
        rows = _dicts(self.conn.execute("SELECT decision, COUNT(*) AS n FROM reviews GROUP BY decision"))
        total = sum(row["n"] for row in rows) or 0
        approved = next((row["n"] for row in rows if row["decision"] == "approve"), 0)
        return {"total": total, "approved": approved,
                "rejected": total - approved,
                "approve_rate": round(approved / total, 4) if total else None}

    # ────────────────────────────── 数据集 ──────────────────────────────
    def create_dataset(self, name: str, *, filter_json: dict[str, Any] | None = None,
                       split_json: dict[str, Any] | None = None) -> int:
        with transaction(self.conn):
            cur = self.conn.execute(
                "INSERT INTO dataset_versions(name, status, filter_json, split_json) VALUES(?,?,?,?)",
                (name, "draft",
                 json.dumps(filter_json or {}, ensure_ascii=False, sort_keys=True),
                 json.dumps(split_json or {}, ensure_ascii=False, sort_keys=True)),
            )
            dataset_id = int(cur.lastrowid)
        return dataset_id

    def add_dataset_items(self, dataset_id: int, rows: Sequence[tuple[int, str]]) -> int:
        with transaction(self.conn):
            self.conn.executemany(
                "INSERT OR IGNORE INTO dataset_items(dataset_id, image_id, split) VALUES(?,?,?)",
                [(dataset_id, image_id, split) for image_id, split in rows],
            )
        return len(rows)

    def get_dataset(self, *, dataset_id: int | None = None, name: str | None = None) -> dict[str, Any] | None:
        if dataset_id is not None:
            return _dict(self.conn.execute("SELECT * FROM dataset_versions WHERE id = ?", (dataset_id,)).fetchone())
        return _dict(self.conn.execute("SELECT * FROM dataset_versions WHERE name = ?", (name,)).fetchone())

    def list_datasets(self) -> list[dict[str, Any]]:
        return _dicts(self.conn.execute("SELECT * FROM dataset_versions ORDER BY id DESC"))

    def dataset_items(self, dataset_id: int, split: str | None = None) -> list[dict[str, Any]]:
        sql = """SELECT di.split, i.* FROM dataset_items di JOIN images i ON i.id = di.image_id
                 WHERE di.dataset_id = ?"""
        params: list[Any] = [dataset_id]
        if split:
            sql += " AND di.split = ?"
            params.append(split)
        return _dicts(self.conn.execute(sql + " ORDER BY i.id", params))

    def freeze_dataset(self, dataset_id: int, *, class_order: Sequence[str], manifest_hash: str,
                       stats: dict[str, Any], root_path: str, split_json: dict[str, Any]) -> dict[str, Any]:
        with transaction(self.conn):
            self.conn.execute(
                """UPDATE dataset_versions
                   SET status='frozen', class_order_json=?, split_json=?, stats_json=?,
                       manifest_hash=?, root_path=?, frozen_at=strftime('%Y-%m-%dT%H:%M:%SZ','now')
                   WHERE id=?""",
                (json.dumps(list(class_order), ensure_ascii=False),
                 json.dumps(split_json, ensure_ascii=False, sort_keys=True),
                 json.dumps(stats, ensure_ascii=False, sort_keys=True),
                 manifest_hash, root_path, dataset_id),
            )
            self._audit("dataset", str(dataset_id), "freeze", "cli", None,
                        {"manifest_hash": manifest_hash, "stats": stats})
        result = self.get_dataset(dataset_id=dataset_id)
        assert result is not None
        return result

    # ────────────────────────────── 运行 ──────────────────────────────
    def create_run(self, kind: str, *, run_key: str | None = None,
                   config_json: dict[str, Any] | None = None,
                   dataset_id: int | None = None, model_id: int | None = None,
                   log_path: str | None = None) -> dict[str, Any]:
        if run_key:
            existing = _dict(self.conn.execute("SELECT * FROM runs WHERE run_key = ?", (run_key,)).fetchone())
            if existing:
                return existing
        with transaction(self.conn):
            cur = self.conn.execute(
                """INSERT INTO runs(kind, status, run_key, config_json, dataset_id, model_id, log_path, started_at)
                   VALUES(?, 'running', ?, ?, ?, ?, ?, strftime('%Y-%m-%dT%H:%M:%SZ','now'))""",
                (kind, run_key, json.dumps(config_json or {}, ensure_ascii=False, sort_keys=True),
                 dataset_id, model_id, log_path),
            )
            run_id = int(cur.lastrowid)
        run = self.get_run(run_id)
        assert run is not None
        return run

    def finish_run(self, run_id: int, *, status: str, metrics: dict[str, Any] | None = None,
                   error: str | None = None) -> None:
        with transaction(self.conn):
            self.conn.execute(
                """UPDATE runs SET status = ?, metrics_json = ?, error = ?,
                   finished_at = strftime('%Y-%m-%dT%H:%M:%SZ','now') WHERE id = ?""",
                (status, json.dumps(metrics or {}, ensure_ascii=False, sort_keys=True), error, run_id),
            )

    def get_run(self, run_id: int) -> dict[str, Any] | None:
        return _dict(self.conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone())

    def update_run(self, run_id: int, *, metrics: dict[str, Any] | None = None,
                   log_path: str | None = None, merge_metrics: bool = True) -> dict[str, Any] | None:
        """增量更新运行（训练每轮写进度用；merge_metrics=False 则整体替换）。"""
        row = self.get_run(run_id)
        if row is None:
            return None
        sets, params = [], []
        if metrics is not None:
            payload = metrics
            if merge_metrics:
                try:
                    current = json.loads(row["metrics_json"] or "{}")
                except json.JSONDecodeError:
                    current = {}
                if not isinstance(current, dict):
                    current = {}
                payload = {**current, **metrics}
            sets.append("metrics_json = ?")
            params.append(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        if log_path is not None:
            sets.append("log_path = ?")
            params.append(log_path)
        if sets:
            params.append(run_id)
            with transaction(self.conn):
                self.conn.execute(f"UPDATE runs SET {', '.join(sets)} WHERE id = ?", params)
        return self.get_run(run_id)

    def latest_run(self, kind: str, *, dataset_id: int | None = None) -> dict[str, Any] | None:
        sql = "SELECT * FROM runs WHERE kind = ? AND status = 'succeeded'"
        params: list[Any] = [kind]
        if dataset_id is not None:
            sql += " AND dataset_id = ?"
            params.append(dataset_id)
        sql += " ORDER BY id DESC LIMIT 1"
        return _dict(self.conn.execute(sql, params).fetchone())

    def list_runs(self, *, kind: str | None = None, status: str | None = None,
                  limit: int = 50) -> list[dict[str, Any]]:
        clauses, params = [], []
        if kind:
            clauses.append("kind = ?")
            params.append(kind)
        if status:
            clauses.append("status = ?")
            params.append(status)
        sql = "SELECT * FROM runs"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        return _dicts(self.conn.execute(sql, params))

    # ────────────────────────────── 统计 ──────────────────────────────
    def task_progress(self) -> dict[str, int]:
        rows = self.conn.execute("SELECT status, COUNT(*) AS n FROM tasks GROUP BY status").fetchall()
        return {row["status"]: row["n"] for row in rows}

    def overview(self) -> dict[str, Any]:
        total_images = self.count_images()
        annotated = int(self.conn.execute(
            "SELECT COUNT(*) AS n FROM tasks WHERE status IN ('annotated','reviewing','approved','rejected')"
        ).fetchone()["n"])
        return {
            "images": total_images,
            "tasks": self.task_progress(),
            "annotated_images": annotated,
            "class_counts": self.class_counts(),
            "reviews": self.review_stats(),
            "batches": int(self.conn.execute("SELECT COUNT(*) AS n FROM batches").fetchone()["n"]),
            "datasets": int(self.conn.execute("SELECT COUNT(*) AS n FROM dataset_versions").fetchone()["n"]),
        }

    # ─────────────────────── 模型注册（M2 预标注用） ───────────────────────
    def upsert_model_version(self, name: str, version: str, *, task: str = "detection",
                             status: str = "candidate", weights_path: str | None = None,
                             labels_json: dict | None = None, metrics_json: dict | None = None,
                             run_id: int | None = None, dataset_id: int | None = None,
                             actor: str = "system") -> dict[str, Any]:
        """按 (name, version) 幂等登记模型权重。"""
        existing = self.conn.execute(
            "SELECT * FROM model_versions WHERE name = ? AND version = ?", (name, version)).fetchone()
        if existing is not None:
            return dict(existing)
        with transaction(self.conn):
            cur = self.conn.execute(
                """INSERT INTO model_versions(name, version, task, status, run_id, dataset_id,
                                              weights_path, labels_json, metrics_json)
                   VALUES(?,?,?,?,?,?,?,?,?)""",
                (name, version, task, status, run_id, dataset_id, weights_path,
                 json.dumps(labels_json or {}, ensure_ascii=False, sort_keys=True),
                 json.dumps(metrics_json or {}, ensure_ascii=False, sort_keys=True)),
            )
            model_id = int(cur.lastrowid)
            self._audit("model", f"{name}:{version}", "create", actor, None,
                        {"id": model_id, "task": task, "status": status, "weights": weights_path})
        row = self.conn.execute("SELECT * FROM model_versions WHERE id = ?", (model_id,)).fetchone()
        return dict(row) if row is not None else {}

    def list_model_versions(self, *, task: str | None = None,
                            status: str | None = None) -> list[dict[str, Any]]:
        clauses, params = [], []
        if task:
            clauses.append("task = ?")
            params.append(task)
        if status:
            clauses.append("status = ?")
            params.append(status)
        sql = "SELECT * FROM model_versions"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY status, id DESC"
        return _dicts(self.conn.execute(sql, params))

    def production_model(self, *, task: str = "detection") -> dict[str, Any] | None:
        return _dict(self.conn.execute(
            "SELECT * FROM model_versions WHERE task = ? AND status = 'production' ORDER BY id DESC LIMIT 1",
            (task,)).fetchone())

    def set_model_status(self, model_id: int, status: str, *, actor: str = "system") -> bool:
        before = _dict(self.conn.execute("SELECT * FROM model_versions WHERE id = ?", (model_id,)).fetchone())
        if before is None:
            return False
        with transaction(self.conn):
            self.conn.execute("UPDATE model_versions SET status = ? WHERE id = ?", (status, model_id))
            self._audit("model", f"{before['name']}:{before['version']}", "status", actor,
                        {"status": before["status"]}, {"status": status})
        return True

    def get_model_version(self, model_id: int) -> dict[str, Any] | None:
        return _dict(self.conn.execute("SELECT * FROM model_versions WHERE id = ?", (model_id,)).fetchone())

    def resolve_model_version(self, ref: str | int) -> dict[str, Any] | None:
        """按 id / `name:version` / `version` 解析模型；`production` 取当前生产模型。"""
        if isinstance(ref, int) or (isinstance(ref, str) and ref.isdigit()):
            return self.get_model_version(int(ref))
        if not isinstance(ref, str):
            return None
        if ref in ("production", "latest"):
            return self.production_model()
        if ":" in ref:
            name, version = ref.split(":", 1)
            return _dict(self.conn.execute(
                "SELECT * FROM model_versions WHERE name = ? AND version = ?", (name, version)).fetchone())
        return _dict(self.conn.execute(
            "SELECT * FROM model_versions WHERE version = ? ORDER BY id DESC LIMIT 1", (ref,)).fetchone())

    def update_model_version(self, model_id: int, *, weights_path: str | None = None,
                             onnx_path: str | None = None, labels_json: dict | None = None,
                             metrics_json: dict | None = None, gate_json: dict | None = None,
                             sha256: str | None = None, run_id: int | None = None,
                             dataset_id: int | None = None, actor: str = "system") -> dict[str, Any] | None:
        """更新模型登记信息（只覆盖显式传入的字段）。"""
        before = self.get_model_version(model_id)
        if before is None:
            return None
        fields: dict[str, Any] = {}
        if weights_path is not None:
            fields["weights_path"] = weights_path
        if onnx_path is not None:
            fields["onnx_path"] = onnx_path
        if labels_json is not None:
            fields["labels_json"] = json.dumps(labels_json, ensure_ascii=False, sort_keys=True)
        if metrics_json is not None:
            fields["metrics_json"] = json.dumps(metrics_json, ensure_ascii=False, sort_keys=True)
        if gate_json is not None:
            fields["gate_json"] = json.dumps(gate_json, ensure_ascii=False, sort_keys=True)
        if sha256 is not None:
            fields["sha256"] = sha256
        if run_id is not None:
            fields["run_id"] = run_id
        if dataset_id is not None:
            fields["dataset_id"] = dataset_id
        if not fields:
            return before
        sets = ", ".join(f"{key} = ?" for key in fields)
        with transaction(self.conn):
            self.conn.execute(f"UPDATE model_versions SET {sets} WHERE id = ?",
                              [*fields.values(), model_id])
            self._audit("model", f"{before['name']}:{before['version']}", "update", actor,
                        None, {key: str(value)[:200] for key, value in fields.items()})
        return self.get_model_version(model_id)

    def archive_other_production(self, keep_id: int, *, task: str = "detection",
                                 actor: str = "system") -> list[int]:
        """把同任务下其它 production 模型归档（保证同任务只有一个生产模型）。"""
        rows = self.conn.execute(
            "SELECT id, name, version FROM model_versions WHERE task = ? AND status = 'production' AND id != ?",
            (task, keep_id)).fetchall()
        archived: list[int] = []
        for row in rows:
            with transaction(self.conn):
                self.conn.execute("UPDATE model_versions SET status = 'archived' WHERE id = ?", (row["id"],))
                self._audit("model", f"{row['name']}:{row['version']}", "archive", actor,
                            {"status": "production"}, {"status": "archived", "replaced_by": keep_id})
            archived.append(int(row["id"]))
        return archived

    # ─────────────────────── 忽略/删除标注（M2） ───────────────────────
    def delete_annotation(self, annotation_id: int, *, actor: str = "api") -> bool:
        """软删一条标注（用于「忽略候选」）。保留审计痕迹，可复查。"""
        row = _dict(self.conn.execute(
            "SELECT * FROM annotations WHERE id = ?", (annotation_id,)).fetchone())
        if row is None or row["deleted_at"] is not None:
            return False
        with transaction(self.conn):
            self.conn.execute("UPDATE annotations SET deleted_at = ? WHERE id = ?", (utc_now(), annotation_id))
            self._audit("annotation", str(row["task_id"]), "delete", actor,
                        self._annotation_snapshot(row), {"deleted": True, "id": annotation_id})
        return True

    # ─────────────────────── 预标注质量统计（M2） ───────────────────────
    def annotation_source_counts(self, *, include_deleted: bool = True) -> dict[str, int]:
        sql = "SELECT source, COUNT(*) AS n FROM annotations"
        if not include_deleted:
            sql += " WHERE deleted_at IS NULL"
        sql += " GROUP BY source"
        return {row["source"]: row["n"] for row in self.conn.execute(sql)}

    def tasks_by_prelabel_state(self) -> dict[str, int]:
        rows = self.conn.execute(
            "SELECT prelabel_state, COUNT(*) AS n FROM tasks GROUP BY prelabel_state").fetchall()
        return {row["prelabel_state"]: row["n"] for row in rows}

    def model_versions_seen(self) -> list[int]:
        rows = self.conn.execute(
            "SELECT DISTINCT model_version_id FROM annotations WHERE model_version_id IS NOT NULL").fetchall()
        return [row["model_version_id"] for row in rows]

    # ────────────────────────────── 审计 ──────────────────────────────
    def _audit(self, entity: str, entity_id: str, action: str, actor: str | None,
               before: Any, after: Any) -> None:
        self.conn.execute(
            "INSERT INTO audit_log(entity, entity_id, action, actor, before_json, after_json) VALUES(?,?,?,?,?,?)",
            (entity, entity_id, action, actor,
             json.dumps(before, ensure_ascii=False, sort_keys=True) if before is not None else None,
             json.dumps(after, ensure_ascii=False, sort_keys=True) if after is not None else None),
        )

    def audit_tail(self, limit: int = 20) -> list[dict[str, Any]]:
        return _dicts(self.conn.execute("SELECT * FROM audit_log ORDER BY id DESC LIMIT ?", (limit,)))

    @staticmethod
    def _annotation_snapshot(row: dict[str, Any]) -> dict[str, Any]:
        return {"id": row["id"], "class_code": row["class_code"],
                "bbox": [row["bbox_x1"], row["bbox_y1"], row["bbox_x2"], row["bbox_y2"]],
                "source": row["source"]}
