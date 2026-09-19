"""M5 统计报表测试：时间趋势、批次对比、类别覆盖、GIS 导出与 Markdown 报表。

夹具策略：``helpers`` 提供真实导入链路（``synth_images`` + ``import_path`` + ``annotate_all``），
需要精细口径（窗口外的 ``created_at``、GPS、软删、模型候选）时直接用 ``repo.conn.execute`` 补数据。
"""

from __future__ import annotations

import csv
import io
import re
import tempfile
import unittest
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from rdinspect import report
from rdinspect.core.ingest import import_path

from helpers import annotate_all, make_config, open_repo, synth_images

BOX_A = {"x1": 0.10, "y1": 0.20, "x2": 0.40, "y2": 0.35}
BOX_B = {"x1": 0.55, "y1": 0.50, "x2": 0.80, "y2": 0.70}
CSV_HEADER = ("image_id,path,captured_at,lat,lon,class,source,score,x1,y1,x2,y2,"
              "length_px,width_px,area_ratio")
ISO_DAY = re.compile(r"^\d{4}-\d{2}-\d{2}$")
ISO_MONTH = re.compile(r"^\d{4}-\d{2}$")
ISO_STAMP = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")


class ReportTestBase(unittest.TestCase):
    """公共夹具：临时数据目录 + 临时库 + 造数工具。"""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.config = make_config(Path(self._tmp.name))
        self.conn, self.repo = open_repo(self.config)
        self.addCleanup(self.conn.close)
        self.today = datetime.now(timezone.utc).date()
        self._seq = 0

    # ────────────────────────── 造数工具 ──────────────────────────
    def day(self, offset: int) -> str:
        """相对今天的 UTC 日期文本（offset 为负表示过去）。"""
        return (self.today + timedelta(days=offset)).isoformat()

    def stamp(self, offset: int, hour: int = 10) -> str:
        """相对今天的 UTC 时间戳文本。"""
        return f"{self.day(offset)}T{hour:02d}:00:00Z"

    def seed_photos(self, count: int = 3, seed: int = 1) -> Any:
        """走真实导入链路造一个批次（返回 IngestReport）。"""
        source = self.config.data_dir / "inbox" / f"photos-{seed}"
        synth_images(source, count=count, seed=seed)
        return import_path(self.config, self.repo, source, kind="photo")

    def add_image(self, batch_id: int | None = None, *, gps: tuple[float, float] | None = None,
                  captured_at: str | None = None, created_at: str | None = None,
                  duplicate_of: int | None = None) -> int:
        """直接插一张影像（无需真实文件），可指定 GPS / 时间 / 重复指向。"""
        self._seq += 1
        image_id = self.repo.insert_image(
            batch_id=batch_id, path=f"raw/fake-{self._seq:03d}.jpg", sha256=f"sha-{self._seq:03d}",
            phash=None, width=96, height=64, bytes_=128, source_kind="photo",
            captured_at=captured_at,
            gps_lat=None if gps is None else gps[0], gps_lon=None if gps is None else gps[1],
            gps_source="none" if gps is None else "exif",
        )
        updates: dict[str, Any] = {}
        if created_at is not None:
            updates["created_at"] = created_at
        if duplicate_of is not None:
            updates["duplicate_of"] = duplicate_of
        if updates:
            self.update_image(image_id, **updates)
        return image_id

    def update_image(self, image_id: int, **fields: Any) -> None:
        """直接更新影像字段（created_at / captured_at / gps_* 等）。"""
        keys = ", ".join(f"{name} = ?" for name in fields)
        self.repo.conn.execute(f"UPDATE images SET {keys} WHERE id = ?", [*fields.values(), image_id])

    def set_task(self, task_id: int, **fields: Any) -> None:
        """直接改任务状态/时间戳（绕过状态机，用于构造窗口内外数据）。"""
        keys = ", ".join(f"{name} = ?" for name in fields)
        self.repo.conn.execute(f"UPDATE tasks SET {keys} WHERE id = ?", [*fields.values(), task_id])

    def annotate(self, task_id: int, *, class_code: str = "transverse_crack",
                 bbox: dict[str, float] | None = None, approve: bool = False) -> int:
        """给任务写一条人工框并提交（可选复核通过），返回标注 id。"""
        self.repo.replace_annotations(
            task_id, [{"class_code": class_code, "kind": "bbox", "bbox": bbox or BOX_A}], actor="test")
        self.repo.submit_task(task_id, actor="test")
        if approve:
            self.repo.add_review(task_id, decision="approve", reviewer="test")
        return int(self.repo.conn.execute(
            "SELECT id FROM annotations WHERE task_id = ? AND deleted_at IS NULL ORDER BY id DESC LIMIT 1",
            (task_id,)).fetchone()["id"])

    def task_ids_of(self, image_ids: list[int]) -> list[int]:
        """按影像顺序取任务 id（导入时每张图都会建任务）。"""
        return [int(self.repo.conn.execute(
            "SELECT id FROM tasks WHERE image_id = ? ORDER BY id LIMIT 1", (image_id,)).fetchone()["id"])
            for image_id in image_ids]


class TrendsTests(ReportTestBase):
    """时间趋势：窗口、分桶、空桶、口径与非法参数。"""

    def test_empty_library_returns_zero_buckets(self) -> None:
        data = report.trends(self.repo, days=5)
        self.assertEqual(data["bucket"], "day")
        self.assertEqual(data["days"], 5)
        self.assertEqual(data["start"], self.day(-4))
        self.assertEqual(data["end"], self.day(0))
        self.assertEqual([item["key"] for item in data["buckets"]],
                         [self.day(-4), self.day(-3), self.day(-2), self.day(-1), self.day(0)])
        for item in data["buckets"]:
            self.assertEqual(
                (item["images"], item["boxes"], item["annotated_tasks"],
                 item["approved_tasks"], item["rejected_tasks"]), (0, 0, 0, 0, 0))
            self.assertEqual(item["by_class"], {})
        self.assertEqual(data["totals"], {"images": 0, "boxes": 0, "annotated_tasks": 0,
                                          "approved_tasks": 0, "rejected_tasks": 0, "by_class": {}})

    def test_day_buckets_are_contiguous_and_ascending(self) -> None:
        data = report.trends(self.repo, days=7)
        keys = [item["key"] for item in data["buckets"]]
        self.assertEqual(len(keys), 7)
        self.assertEqual(keys, sorted(keys))
        self.assertEqual(len(set(keys)), 7)
        for key in keys:
            self.assertRegex(key, ISO_DAY)
        self.assertEqual([date.fromisoformat(key) for key in keys],
                         [self.today - timedelta(days=offset) for offset in range(6, -1, -1)])

    def test_week_bucket_keys_are_mondays(self) -> None:
        data = report.trends(self.repo, days=14, bucket="week")
        keys = [item["key"] for item in data["buckets"]]
        self.assertGreaterEqual(len(keys), 2)
        self.assertEqual(keys, sorted(keys))
        for key in keys:
            self.assertRegex(key, ISO_DAY)
            self.assertEqual(date.fromisoformat(key).weekday(), 0, "周桶键应为该 ISO 周的周一")
        self.assertEqual(keys[-1], (self.today - timedelta(days=self.today.weekday())).isoformat())

    def test_month_bucket_keys_format_and_contiguity(self) -> None:
        data = report.trends(self.repo, days=70, bucket="month")
        keys = [item["key"] for item in data["buckets"]]
        self.assertGreaterEqual(len(keys), 3)
        self.assertEqual(keys, sorted(keys))
        for key in keys:
            self.assertRegex(key, ISO_MONTH)
        self.assertEqual(keys[-1], f"{self.today.year:04d}-{self.today.month:02d}")
        first = date.fromisoformat(f"{keys[0]}-01")
        last = date.fromisoformat(f"{keys[-1]}-01")
        self.assertEqual(len(keys), (last.year - first.year) * 12 + last.month - first.month + 1)

    def test_invalid_bucket_and_days_raise(self) -> None:
        with self.assertRaises(ValueError):
            report.trends(self.repo, days=7, bucket="hour")
        with self.assertRaises(ValueError):
            report.trends(self.repo, days=0)
        with self.assertRaises(ValueError):
            report.batch_comparison(self.repo, limit=0)

    def test_window_excludes_outside_rows(self) -> None:
        ingest = self.seed_photos(count=3)
        annotate_all(self.repo, class_code="transverse_crack", approve=True)
        inside = report.trends(self.repo, days=7)
        self.assertEqual(inside["totals"]["images"], 3)
        self.assertEqual(inside["buckets"][-1]["images"], 3)

        self.update_image(ingest.image_ids[0], created_at=self.stamp(-30))
        outside = report.trends(self.repo, days=7)
        self.assertEqual(outside["totals"]["images"], 2, "窗口外影像不计入")
        self.assertEqual(outside["buckets"][-1]["images"], 2)
        self.assertNotIn(self.day(-30), [item["key"] for item in outside["buckets"]])

    def test_boxes_exclude_model_candidates_and_deleted(self) -> None:
        ingest = self.seed_photos(count=2)
        first, second = self.task_ids_of(ingest.image_ids)
        self.annotate(first, class_code="transverse_crack")
        deleted_id = self.annotate(second, class_code="garbage", bbox=BOX_B)
        self.assertTrue(self.repo.delete_annotation(deleted_id))
        self.repo.add_model_candidates(second, [{"class_code": "pothole", "bbox": BOX_A, "score": 0.9}])

        data = report.trends(self.repo, days=7)
        today = data["buckets"][-1]
        self.assertEqual(today["boxes"], 1, "模型候选与被软删的框都不计入")
        self.assertEqual(today["by_class"], {"transverse_crack": 1})
        self.assertEqual(data["totals"]["boxes"], 1)
        self.assertNotIn("pothole", data["totals"]["by_class"])
        self.assertNotIn("garbage", data["totals"]["by_class"])

    def test_by_class_counts_within_window(self) -> None:
        ingest = self.seed_photos(count=2)
        first, second = self.task_ids_of(ingest.image_ids)
        self.repo.replace_annotations(first, [
            {"class_code": "transverse_crack", "kind": "bbox", "bbox": BOX_A},
            {"class_code": "pothole", "kind": "bbox", "bbox": BOX_B},
        ], actor="test")
        self.repo.submit_task(first, actor="test")
        self.annotate(second, class_code="pothole", bbox=BOX_A)

        data = report.trends(self.repo, days=3)
        self.assertEqual(data["buckets"][-1]["by_class"], {"pothole": 2, "transverse_crack": 1})
        self.assertEqual(data["totals"]["by_class"], {"pothole": 2, "transverse_crack": 1})
        self.assertEqual(data["totals"]["boxes"], 3)
        self.assertEqual(sum(data["totals"]["by_class"].values()), data["totals"]["boxes"])

    def test_tasks_counted_by_submitted_and_reviewed_dates(self) -> None:
        ingest = self.seed_photos(count=3)
        first, second, third = self.task_ids_of(ingest.image_ids)
        for task_id in (first, second, third):
            self.repo.replace_annotations(
                task_id, [{"class_code": "transverse_crack", "kind": "bbox", "bbox": BOX_A}], actor="test")
        self.set_task(first, status="approved", submitted_at=self.stamp(-2), reviewed_at=self.stamp(-1))
        self.set_task(second, status="annotating", submitted_at=self.stamp(-1))
        # 打回是"把任务放回 annotating"（见 Repo.add_review），事实来源是 reviews 表
        self.repo.conn.execute(
            "INSERT INTO reviews(task_id, reviewer, decision, reason_code, created_at) "
            "VALUES(?, 'test', 'reject', 'missing', ?)", (second, self.stamp(-1)))
        self.repo.conn.commit()
        self.set_task(third, status="approved", submitted_at=self.stamp(-20), reviewed_at=self.stamp(-20))

        data = report.trends(self.repo, days=7)
        buckets = {item["key"]: item for item in data["buckets"]}
        self.assertEqual(buckets[self.day(-2)]["annotated_tasks"], 1)
        self.assertEqual(buckets[self.day(-1)]["annotated_tasks"], 1)
        self.assertEqual(buckets[self.day(-1)]["approved_tasks"], 1)
        self.assertEqual(buckets[self.day(-1)]["rejected_tasks"], 1)
        self.assertEqual(data["totals"]["annotated_tasks"], 2, "窗口外的提交不计入")
        self.assertEqual(data["totals"]["approved_tasks"], 1)
        self.assertEqual(data["totals"]["rejected_tasks"], 1)

    def test_week_bucket_aggregates_days_of_same_week(self) -> None:
        ingest = self.seed_photos(count=2)
        self.update_image(ingest.image_ids[0], created_at=self.stamp(-1))
        self.update_image(ingest.image_ids[1], created_at=self.stamp(0))
        data = report.trends(self.repo, days=14, bucket="week")
        buckets = {item["key"]: item for item in data["buckets"]}
        yesterday = self.today - timedelta(days=1)
        monday_today = self.today - timedelta(days=self.today.weekday())
        monday_yesterday = yesterday - timedelta(days=yesterday.weekday())
        if monday_yesterday == monday_today:  # 今天不是周一：两天同周，应并入同一周桶
            self.assertEqual(buckets[monday_today.isoformat()]["images"], 2)
        else:  # 今天是周一：昨天属于上一周
            self.assertEqual(buckets[monday_yesterday.isoformat()]["images"], 1)
            self.assertEqual(buckets[monday_today.isoformat()]["images"], 1)
        self.assertEqual(data["totals"]["images"], 2)


class BatchComparisonTests(ReportTestBase):
    """批次对比：排序、计数口径、导入统计解析。"""

    def test_empty_library_returns_empty_list(self) -> None:
        self.assertEqual(report.batch_comparison(self.repo), [])

    def test_latest_batch_first_and_limit(self) -> None:
        first = self.seed_photos(count=2, seed=1)
        second = self.seed_photos(count=2, seed=2)
        items = report.batch_comparison(self.repo)
        self.assertEqual([item["batch_id"] for item in items], [second.batch_id, first.batch_id])
        limited = report.batch_comparison(self.repo, limit=1)
        self.assertEqual([item["batch_id"] for item in limited], [second.batch_id])

    def test_counts_boxes_labeled_images_and_average(self) -> None:
        batch_id = self.repo.create_batch("photo", source="fixture", note="对比用")
        first = self.add_image(batch_id)
        second = self.add_image(batch_id)
        duplicate = self.add_image(batch_id, duplicate_of=first)
        first_task = self.repo.create_task(first)
        second_task = self.repo.create_task(second)
        duplicate_task = self.repo.create_task(duplicate)
        self.repo.replace_annotations(first_task, [
            {"class_code": "transverse_crack", "kind": "bbox", "bbox": BOX_A},
            {"class_code": "pothole", "kind": "bbox", "bbox": BOX_B},
        ], actor="test")
        self.repo.submit_task(first_task, actor="test")
        self.repo.add_review(first_task, decision="approve", reviewer="test")
        # 近似重复影像上的标注不计入 boxes/labeled_images（与 images 口径保持一致）
        self.annotate(duplicate_task, class_code="transverse_crack", bbox=BOX_B)
        self.repo.add_model_candidates(second_task, [{"class_code": "garbage", "bbox": BOX_A, "score": 0.5}])

        item = report.batch_comparison(self.repo)[0]
        self.assertEqual(item["batch_id"], batch_id)
        self.assertEqual(item["kind"], "photo")
        self.assertEqual(item["source"], "fixture")
        self.assertEqual(item["note"], "对比用")
        self.assertRegex(item["created_at"], ISO_STAMP)
        self.assertEqual(item["images"], 2)
        self.assertEqual(item["duplicates"], 1)
        self.assertEqual(item["boxes"], 2, "模型候选与重复影像上的框都不计入")
        self.assertEqual(item["labeled_images"], 1, "只数非重复且有非模型标注的影像")
        self.assertEqual(item["approved_tasks"], 1)
        self.assertEqual(item["by_class"], {"transverse_crack": 1, "pothole": 1},
                         "重复影像上的 transverse_crack 不计入")
        self.assertAlmostEqual(item["avg_boxes_per_image"], 1.0, places=6,
                               msg="2 个非重复影像 / 2 个框")

    def test_import_stats_parsed_and_bad_json(self) -> None:
        ingest = self.seed_photos(count=3)
        good = report.batch_comparison(self.repo)[0]
        self.assertEqual(good["batch_id"], ingest.batch_id)
        self.assertEqual(good["import_stats"]["added"], ingest.added)
        self.assertIn("dup_sha", good["import_stats"])

        broken = self.repo.create_batch("photo", source="broken")
        self.repo.conn.execute("UPDATE batches SET stats_json = ? WHERE id = ?", ("{not json", broken))
        items = {item["batch_id"]: item for item in report.batch_comparison(self.repo)}
        self.assertEqual(items[broken]["import_stats"], {}, "坏 JSON 应回退为空字典")
        self.assertEqual(items[broken]["images"], 0)
        self.assertEqual(items[broken]["avg_boxes_per_image"], 0.0, "无影像时平均框数为 0.0")

        array_stats = self.repo.create_batch("photo", source="array")
        self.repo.conn.execute("UPDATE batches SET stats_json = ? WHERE id = ?", ("[1, 2]", array_stats))
        items = {item["batch_id"]: item for item in report.batch_comparison(self.repo)}
        self.assertEqual(items[array_stats]["import_stats"], {}, "非对象 JSON 也应回退为空字典")


class CoverageTests(ReportTestBase):
    """类别覆盖、来源分布与复核进度。"""

    def test_classes_sorted_by_order_index(self) -> None:
        self.repo.add_class("water_puddle", "路面积水", "Water Puddle", order_index=-1)
        data = report.coverage(self.repo)
        codes = [item["code"] for item in data["classes"]]
        self.assertEqual(codes[0], "water_puddle", "order_index 更小的类别应排在最前")
        order_index = [int(row["order_index"]) for row in self.repo.conn.execute(
            "SELECT order_index FROM classes ORDER BY order_index, code")]
        indexes = [int(self.repo.get_class(code)["order_index"]) for code in codes]
        self.assertEqual(indexes, order_index)
        self.assertEqual(set(data["classes"][0]),
                         {"code", "name_zh", "name_en", "active", "is_crack", "approved_images",
                          "boxes", "model_candidates", "adopted"})
        self.assertEqual(data["classes"][0]["name_zh"], "路面积水")

    def test_approve_ratio_is_none_without_reviews(self) -> None:
        data = report.coverage(self.repo)
        self.assertEqual(data["review"], {"approved": 0, "rejected": 0, "approve_ratio": None})
        self.assertEqual(data["totals"]["boxes"], 0)
        self.assertEqual(data["totals"]["pending_tasks"], 0)
        self.assertEqual(data["sources"], {"human": 0, "model": 0, "model_edited": 0, "import": 0})

    def test_review_counts_and_ratio(self) -> None:
        ingest = self.seed_photos(count=2)
        first, second = self.task_ids_of(ingest.image_ids)
        self.annotate(first, approve=True)
        self.annotate(second)
        self.repo.add_review(second, decision="reject", reason_code="missing", reviewer="test")

        data = report.coverage(self.repo)
        self.assertEqual(data["review"]["approved"], 1)
        self.assertEqual(data["review"]["rejected"], 1)
        self.assertAlmostEqual(data["review"]["approve_ratio"], 0.5, places=4)
        # 注意：Repo.add_review(reject) 会把任务置回 annotating（不写 status='rejected'），
        # 因此 totals.approved_tasks 只反映状态为 approved 的任务数。
        self.assertEqual(data["totals"]["approved_tasks"], 1)
        self.assertEqual(data["totals"]["tasks"].get("annotating"), 1)

    def test_class_metrics_adoption_and_source_distribution(self) -> None:
        batch_id = self.repo.create_batch("photo", source="coverage")
        image_a = self.add_image(batch_id)
        image_b = self.add_image(batch_id)
        image_c = self.add_image(batch_id)
        task_a = self.repo.create_task(image_a)
        task_b = self.repo.create_task(image_b)
        task_c = self.repo.create_task(image_c)

        self.annotate(task_a, class_code="transverse_crack", approve=True)
        self.repo.add_model_candidates(task_b, [{"class_code": "pothole", "bbox": BOX_A, "score": 0.8}])
        self.assertEqual(self.repo.adopt_model_candidates(task_b), 1)
        self.repo.add_model_candidates(task_c, [{"class_code": "garbage", "bbox": BOX_B, "score": 0.4}])
        deleted_id = self.annotate(task_c, class_code="garbage", bbox=BOX_A)
        self.assertTrue(self.repo.delete_annotation(deleted_id))

        data = report.coverage(self.repo)
        by_code = {item["code"]: item for item in data["classes"]}
        self.assertEqual(by_code["transverse_crack"]["boxes"], 1)
        self.assertEqual(by_code["transverse_crack"]["approved_images"], 1)
        self.assertEqual(by_code["pothole"]["boxes"], 1, "model_edited 属于已确认标注")
        self.assertEqual(by_code["pothole"]["adopted"], 1)
        self.assertEqual(by_code["pothole"]["model_candidates"], 0)
        self.assertEqual(by_code["garbage"]["model_candidates"], 1)
        self.assertEqual(by_code["garbage"]["boxes"], 0, "被软删的人工框不计入")
        self.assertEqual(by_code["garbage"]["adopted"], 0)
        self.assertEqual(data["sources"], {"human": 1, "model": 1, "model_edited": 1, "import": 0})
        self.assertEqual(data["totals"]["boxes"], 2)

    def test_pending_tasks_and_task_progress(self) -> None:
        self.assertEqual(report.coverage(self.repo)["totals"]["pending_tasks"], 0)
        ingest = self.seed_photos(count=3)
        self.assertEqual(report.coverage(self.repo)["totals"]["pending_tasks"], 3)
        self.annotate(self.task_ids_of(ingest.image_ids)[0], approve=True)
        data = report.coverage(self.repo)
        self.assertEqual(data["totals"]["pending_tasks"], 2)
        self.assertEqual(data["totals"]["tasks"].get("approved"), 1)
        self.assertEqual(data["totals"]["images"], 3)

    def test_approved_images_requires_approved_task(self) -> None:
        batch_id = self.repo.create_batch("photo", source="approve")
        task_id = self.repo.create_task(self.add_image(batch_id))
        self.annotate(task_id, class_code="pothole")
        self.assertEqual({item["code"]: item["approved_images"]
                          for item in report.coverage(self.repo)["classes"]}["pothole"], 0)
        self.repo.add_review(task_id, decision="approve", reviewer="test")
        self.assertEqual({item["code"]: item["approved_images"]
                          for item in report.coverage(self.repo)["classes"]}["pothole"], 1)


class GisFeatureTests(ReportTestBase):
    """GIS 字段导出：GPS 过滤、坐标顺序、bbox/时间过滤、截断与候选来源。"""

    def setUp(self) -> None:
        super().setUp()
        self.batch_id = self.repo.create_batch("photo", source="gis")
        self.image_a = self.add_image(self.batch_id, gps=(30.0, 120.0),
                                      captured_at=self.stamp(-3), created_at=self.stamp(-3))
        self.image_b = self.add_image(self.batch_id, gps=(31.5, 121.5),
                                      captured_at=self.stamp(-1), created_at=self.stamp(-1))
        self.image_no_gps = self.add_image(self.batch_id, captured_at=self.stamp(-1),
                                           created_at=self.stamp(-1))
        self.image_old = self.add_image(self.batch_id, gps=(-33.9, 151.2), created_at=self.stamp(-40))
        self.image_dup = self.add_image(self.batch_id, gps=(10.0, 10.0),
                                        captured_at=self.stamp(-1), created_at=self.stamp(-1),
                                        duplicate_of=self.image_a)
        self.image_model = self.add_image(self.batch_id, gps=(20.0, 20.0),
                                          captured_at=self.stamp(-2), created_at=self.stamp(-2))
        self.image_deleted = self.add_image(self.batch_id, gps=(21.0, 21.0),
                                            captured_at=self.stamp(-2), created_at=self.stamp(-2))
        self.task_a = self.repo.create_task(self.image_a)
        self.task_b = self.repo.create_task(self.image_b)
        self.task_no_gps = self.repo.create_task(self.image_no_gps)
        self.task_old = self.repo.create_task(self.image_old)
        self.task_dup = self.repo.create_task(self.image_dup)
        self.task_model = self.repo.create_task(self.image_model)
        self.task_deleted = self.repo.create_task(self.image_deleted)

        self.annotate(self.task_a, class_code="transverse_crack", bbox=BOX_A)
        self.annotate(self.task_b, class_code="pothole", bbox=BOX_B)
        self.annotate(self.task_no_gps, class_code="transverse_crack", bbox=BOX_A)
        self.annotate(self.task_old, class_code="transverse_crack", bbox=BOX_A)
        self.annotate(self.task_dup, class_code="transverse_crack", bbox=BOX_A)
        self.repo.add_model_candidates(self.task_model, [
            {"class_code": "pothole", "bbox": BOX_B, "score": 0.77}])
        deleted_id = self.annotate(self.task_deleted, class_code="garbage", bbox=BOX_B)
        self.assertTrue(self.repo.delete_annotation(deleted_id))

    def test_only_gps_points_and_lon_lat_order(self) -> None:
        data = report.gis_features(self.repo)
        self.assertEqual(data["count"], 3)
        self.assertEqual(len(data["features"]), 3)
        self.assertFalse(data["truncated"])
        by_image = {feature["properties"]["image_id"]: feature for feature in data["features"]}
        self.assertEqual(sorted(by_image), [self.image_a, self.image_b, self.image_old],
                         "无 GPS、重复影像、模型候选与被软删的标注都不导出")
        self.assertEqual(by_image[self.image_a]["geometry"],
                         {"type": "Point", "coordinates": [120.0, 30.0]}, "GeoJSON 是 lon,lat")
        self.assertEqual(by_image[self.image_old]["geometry"]["coordinates"], [151.2, -33.9])
        self.assertEqual(by_image[self.image_a]["type"], "Feature")

    def test_feature_properties_shape(self) -> None:
        feature = report.gis_features(self.repo, classes=("pothole",))["features"][0]
        self.assertEqual(set(feature["properties"]),
                         {"image_id", "task_id", "path", "captured_at", "class_code", "source", "score",
                          "difficult", "bbox", "length_px", "width_px", "area_ratio", "batch_id"})
        properties = feature["properties"]
        self.assertEqual(properties["task_id"], self.task_b)
        self.assertEqual(properties["class_code"], "pothole")
        self.assertEqual(properties["source"], "human")
        self.assertIsNone(properties["score"])
        self.assertIs(properties["difficult"], False)
        self.assertEqual(properties["bbox"], [BOX_B["x1"], BOX_B["y1"], BOX_B["x2"], BOX_B["y2"]])
        self.assertEqual(properties["captured_at"], self.stamp(-1))
        self.assertEqual(properties["batch_id"], self.batch_id)
        self.assertAlmostEqual(properties["area_ratio"], (0.80 - 0.55) * (0.70 - 0.50), places=6)

    def test_bbox_filter(self) -> None:
        hit = report.gis_features(self.repo, bbox=(119.0, 29.0, 121.0, 31.0))
        self.assertEqual(hit["count"], 1)
        self.assertEqual(hit["features"][0]["properties"]["image_id"], self.image_a)
        empty = report.gis_features(self.repo, bbox=(0.0, 0.0, 1.0, 1.0))
        self.assertEqual(empty["count"], 0)
        self.assertEqual(empty["without_gps"], 1, "bbox 只作用于有 GPS 的点，不影响缺 GPS 计数")
        with self.assertRaises(ValueError):
            report.gis_features(self.repo, bbox=(120.0, 30.0, 119.0, 31.0))
        with self.assertRaises(ValueError):
            report.gis_features(self.repo, bbox=(1.0, 2.0, 3.0))  # type: ignore[arg-type]

    def test_since_until_filter_uses_captured_at_then_created_at(self) -> None:
        window = report.gis_features(self.repo, since=self.day(-2), until=self.day(-1))
        self.assertEqual([feature["properties"]["image_id"] for feature in window["features"]],
                         [self.image_b])
        self.assertEqual(window["without_gps"], 1, "窗口内的无 GPS 标注应计入 without_gps")

        fallback = report.gis_features(self.repo, since=self.day(-40), until=self.day(-40))
        self.assertEqual([feature["properties"]["image_id"] for feature in fallback["features"]],
                         [self.image_old], "captured_at 为空时回退到 created_at")
        with self.assertRaises(ValueError):
            report.gis_features(self.repo, since=self.day(-1), until=self.day(-3))
        with self.assertRaises(ValueError):
            report.gis_features(self.repo, since="2026-13-99")
        with self.assertRaises(ValueError):
            report.gis_features(self.repo, limit=-1)

    def test_without_gps_counts_filtered_annotations(self) -> None:
        data = report.gis_features(self.repo)
        self.assertEqual(data["count"], 3)
        self.assertEqual(data["without_gps"], 1)
        only_gps_window = report.gis_features(self.repo, since=self.day(-2), until=self.day(-1))
        self.assertEqual(only_gps_window["without_gps"], 1)
        by_class = report.gis_features(self.repo, classes=("transverse_crack",))
        self.assertEqual(by_class["count"], 2, "只导出 transverse_crack 的两个 GPS 点")
        self.assertEqual(by_class["without_gps"], 1)

    def test_limit_truncates_and_keeps_stable_order(self) -> None:
        limited = report.gis_features(self.repo, limit=2)
        self.assertEqual(limited["count"], 2)
        self.assertTrue(limited["truncated"])
        self.assertEqual([feature["properties"]["image_id"] for feature in limited["features"]],
                         [self.image_a, self.image_b], "排序稳定：image_id, annotation id")
        single = report.gis_features(self.repo, limit=1)
        self.assertEqual(single["count"], 1)
        self.assertTrue(single["truncated"])
        zero = report.gis_features(self.repo, limit=0)
        self.assertEqual(zero["features"], [])
        self.assertTrue(zero["truncated"])

    def test_sources_default_excludes_model_and_explicit_includes(self) -> None:
        default = report.gis_features(self.repo)
        self.assertNotIn(self.image_model,
                         [feature["properties"]["image_id"] for feature in default["features"]])
        candidates = report.gis_features(self.repo, sources=("model",))
        self.assertEqual(candidates["count"], 1)
        self.assertEqual(candidates["features"][0]["properties"]["image_id"], self.image_model)
        self.assertEqual(candidates["features"][0]["properties"]["source"], "model")
        self.assertAlmostEqual(candidates["features"][0]["properties"]["score"], 0.77, places=6)
        self.assertEqual(candidates["without_gps"], 0, "无 GPS 的那张图只有人工标注，被 sources 过滤掉")
        both = report.gis_features(self.repo, sources=("human", "model", "model_edited", "import"))
        self.assertEqual(both["count"], 4, "显式包含 model 后候选也导出")


class ExportFormatTests(ReportTestBase):
    """GeoJSON / CSV 导出格式。"""

    def setUp(self) -> None:
        super().setUp()
        self.batch_id = self.repo.create_batch("photo", source="export")
        self.image_id = self.add_image(self.batch_id, gps=(30.5, 120.25), captured_at=self.stamp(-1))
        self.other_image = self.add_image(self.batch_id, gps=(31.0, 121.0), captured_at=self.stamp(-1))
        first = self.repo.create_task(self.image_id)
        second = self.repo.create_task(self.other_image)
        self.annotate(first, class_code="transverse_crack", bbox=BOX_A)
        self.annotate(second, class_code="pothole", bbox=BOX_B)
        self.payload = report.gis_features(self.repo)

    def test_geojson_structure_and_metadata(self) -> None:
        geo = report.to_geojson(self.payload, metadata={"name": "路检导出", "generated_at": self.stamp(0)})
        self.assertEqual(geo["type"], "FeatureCollection")
        self.assertEqual(len(geo["features"]), 2)
        self.assertEqual(geo["properties"]["name"], "路检导出")
        self.assertEqual(geo["properties"]["generated_at"], self.stamp(0))
        self.assertEqual(geo["properties"]["count"], self.payload["count"])
        self.assertEqual(geo["properties"]["without_gps"], 0)
        bare = report.to_geojson(self.payload)
        self.assertEqual(bare["properties"], {"count": 2, "without_gps": 0})
        self.assertEqual(self.payload["features"][0]["geometry"]["type"], "Point",
                         "导出不应改动原始 payload")

    def test_csv_header_and_trailing_newline(self) -> None:
        text = report.to_gis_csv(self.payload)
        self.assertTrue(text.endswith("\n"))
        lines = text.splitlines()
        self.assertEqual(lines[0], CSV_HEADER)
        self.assertEqual(len(lines), self.payload["count"] + 1)
        empty = report.to_gis_csv({"features": [], "count": 0, "without_gps": 0})
        self.assertEqual(empty, CSV_HEADER + "\n")

    def test_csv_values_and_missing_cells(self) -> None:
        rows = list(csv.reader(io.StringIO(report.to_gis_csv(self.payload))))
        header, body = rows[0], rows[1:]
        self.assertEqual(header, CSV_HEADER.split(","))
        geo = report.to_geojson(self.payload)["features"]
        for row, feature in zip(body, geo):
            record = dict(zip(header, row))
            lon, lat = feature["geometry"]["coordinates"]
            self.assertAlmostEqual(float(record["lon"]), lon, places=6, msg="CSV 经度应与 GeoJSON 一致")
            self.assertAlmostEqual(float(record["lat"]), lat, places=6, msg="CSV 纬度应与 GeoJSON 一致")
            self.assertEqual(record["class"], feature["properties"]["class_code"])
            self.assertEqual(record["x1"], repr(feature["properties"]["bbox"][0]))
            self.assertEqual(record["score"], "", "缺失的 score 输出空串")
            self.assertEqual(record["length_px"], "")
            self.assertEqual(record["width_px"], "")
            self.assertNotEqual(record["area_ratio"], "")


class SummaryTests(ReportTestBase):
    """summary() 与 render_markdown()。"""

    def test_summary_keys_and_generated_at(self) -> None:
        data = report.summary(self.repo, days=7, bucket="week", batch_limit=3)
        self.assertEqual(set(data), {"generated_at", "trends", "batches", "coverage", "gis"})
        self.assertRegex(data["generated_at"], ISO_STAMP)
        self.assertEqual(data["trends"]["days"], 7)
        self.assertEqual(data["trends"]["bucket"], "week")
        self.assertEqual(data["batches"], [])
        self.assertEqual(data["coverage"]["review"]["approve_ratio"], None)
        self.assertEqual(data["gis"], {"exportable": 0, "without_gps": 0})

    def test_summary_reflects_fixture_and_gis_gap(self) -> None:
        ingest = self.seed_photos(count=3)
        annotate_all(self.repo, class_code="transverse_crack", approve=True)
        self.update_image(ingest.image_ids[0], gps_lat=30.0, gps_lon=120.0,
                          gps_source="exif", captured_at=self.stamp(-1))
        data = report.summary(self.repo, days=7)
        self.assertEqual(len(data["batches"]), 1)
        self.assertEqual(data["batches"][0]["images"], 3)
        self.assertEqual(data["coverage"]["totals"]["approved_tasks"], 3)
        self.assertEqual(data["gis"], {"exportable": 1, "without_gps": 2})
        self.assertEqual(data["trends"]["totals"]["images"], 3)

    def test_render_markdown_has_chinese_tables_and_totals(self) -> None:
        ingest = self.seed_photos(count=3)
        annotate_all(self.repo, class_code="transverse_crack", approve=True)
        self.update_image(ingest.image_ids[0], gps_lat=30.0, gps_lon=120.0, gps_source="exif")
        text = report.render_markdown(report.summary(self.repo, days=7))
        self.assertIn("# 路检统计报表", text)
        self.assertIn("| 桶 | 影像 | 标注框 | 已提交任务 | 已通过任务 | 已打回任务 |", text)
        self.assertIn("| **总计** | 3 | 3 | 3 | 3 | 0 |", text)
        self.assertIn("窗口内分类别框数：横向裂缝 3", text)
        self.assertIn("## 2. 批次对比（最新在前）", text)
        self.assertIn("| 类别 | 中文名 | 已复核图像数 | 框数 | 候选数 | 采纳数 |", text)
        self.assertIn("| transverse_crack | 横向裂缝 | 3 | 3 | 0 | 0 |", text)
        self.assertIn("| **总计** | — | — | 3 | 0 | 0 |", text)
        self.assertIn("复核通过率：1.000（100.0%）", text)
        self.assertEqual(text.splitlines()[-1], "缺少 GPS 的点：2")

    def test_render_markdown_only_nonzero_buckets(self) -> None:
        ingest = self.seed_photos(count=2)
        self.update_image(ingest.image_ids[0], created_at=self.stamp(0))
        self.update_image(ingest.image_ids[1], created_at=self.stamp(0))
        text = report.render_markdown(report.summary(self.repo, days=5))
        table_rows = [line for line in text.splitlines() if line.startswith("| 20")]
        self.assertEqual(len(table_rows), 1, "只显示数值非零的桶")
        self.assertIn(f"| {self.day(0)} | 2 | 0 | 0 | 0 | 0 |", text)
        self.assertIn("| **总计** | 2 | 0 | 0 | 0 | 0 |", text)

    def test_render_markdown_missing_values_as_dash(self) -> None:
        text = report.render_markdown(report.summary(self.repo, days=3))
        self.assertIn("复核通过率：—（—）", text)
        self.assertIn("| **总计** | — | — | 0 | 0 | 0 |", text)
        self.assertIn("（窗口内没有非零桶）", text)
        self.assertIn("（无批次数据）", text)
        self.assertEqual(text.splitlines()[-1], "缺少 GPS 的点：0")
        without_gis = report.summary(self.repo, days=3)
        without_gis.pop("gis")
        self.assertEqual(report.render_markdown(without_gis).splitlines()[-1], "缺少 GPS 的点：—")

    def test_render_markdown_batch_row_and_import_stats(self) -> None:
        ingest = self.seed_photos(count=2)
        text = report.render_markdown(report.summary(self.repo, days=7))
        self.assertIn(f"| #{ingest.batch_id} | photo |", text)
        self.assertIn("added=2", text)
        self.assertIn(" | 2 | 0 | 0 | 0 | 0 | 0.000 |", text)


if __name__ == "__main__":
    unittest.main()
