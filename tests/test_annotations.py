"""任务租约、标注全量提交（差异+审计）、模型候选采纳、复核流程测试。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from rdinspect.core.ingest import import_path

from helpers import make_config, open_repo, synth_images

BOX_A = {"x1": 0.10, "y1": 0.20, "x2": 0.40, "y2": 0.35}
BOX_B = {"x1": 0.55, "y1": 0.50, "x2": 0.80, "y2": 0.70}


class TestTasksAndAnnotations(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        tmp_path = Path(self._tmp.name)
        self.config = make_config(tmp_path)
        self.conn, self.repo = open_repo(self.config)
        source = self.config.data_dir / "inbox" / "photos"
        synth_images(source, count=3)
        self.report = import_path(self.config, self.repo, source, kind="photo")

    def tearDown(self) -> None:
        self.conn.close()
        self._tmp.cleanup()

    def _first_task(self) -> int:
        return self.repo.list_tasks(status="pending", limit=1)[0]["id"]

    def test_lease_moves_task_and_is_exclusive(self) -> None:
        leased = self.repo.lease_tasks(count=1, assignee="tester", lease_seconds=60)
        self.assertEqual(len(leased), 1)
        self.assertEqual(leased[0]["status"], "annotating")
        self.assertEqual(leased[0]["assignee"], "tester")
        self.assertIsNotNone(leased[0]["lease_until"])
        second = self.repo.lease_tasks(count=1, assignee="tester2", lease_seconds=60)
        self.assertEqual(len(second), 1)
        self.assertNotEqual(second[0]["id"], leased[0]["id"], "同一任务不得被重复领取")
        self.assertEqual(second[0]["assignee"], "tester2")

    def test_replace_annotations_diff_counts_and_audit(self) -> None:
        task_id = self._first_task()
        first = self.repo.replace_annotations(task_id, [
            {"class_code": "transverse_crack", "kind": "bbox", "bbox": BOX_A},
            {"class_code": "pothole", "kind": "bbox", "bbox": BOX_B},
        ], actor="tester")
        self.assertEqual((first["added"], first["updated"], first["deleted"]), (2, 0, 0))
        self.assertEqual(len(first["annotations"]), 2)

        # 第二次提交：保留 A、改 B 为 垃圾、删除 B 之外的旧框 → 1 updated + 1 added + 1 deleted
        second = self.repo.replace_annotations(task_id, [
            {"class_code": "transverse_crack", "kind": "bbox", "bbox": BOX_A},
            {"class_code": "garbage", "kind": "bbox", "bbox": BOX_B},
        ], actor="tester")
        self.assertEqual((second["added"], second["updated"], second["deleted"]), (1, 1, 1))
        codes = sorted(row["class_code"] for row in second["annotations"])
        self.assertEqual(codes, ["garbage", "transverse_crack"])

        audit = self.repo.audit_tail(limit=5)
        self.assertTrue(any(row["entity"] == "annotation" and row["action"] == "replace" for row in audit))

    def test_explicit_none_source_falls_back_to_human(self) -> None:
        """API 层可能显式传 source=None（前端不传该字段），必须落回 human 而非违反 NOT NULL。"""
        task_id = self._first_task()
        result = self.repo.replace_annotations(task_id, [
            {"class_code": "pothole", "kind": "bbox", "bbox": BOX_A, "source": None}])
        self.assertEqual(result["added"], 1)
        self.assertEqual(result["annotations"][0]["source"], "human")

    def test_invalid_bbox_and_unknown_class_are_rejected(self) -> None:
        task_id = self._first_task()
        with self.assertRaises(ValueError):
            self.repo.replace_annotations(task_id, [
                {"class_code": "pothole", "kind": "bbox", "bbox": {"x1": 0.5, "y1": 0.1, "x2": 0.4, "y2": 0.2}}])
        with self.assertRaises(ValueError):
            self.repo.replace_annotations(task_id, [
                {"class_code": "not_a_class", "kind": "bbox", "bbox": BOX_A}])

    def test_model_candidates_are_kept_and_adoptable(self) -> None:
        task_id = self._first_task()
        inserted = self.repo.add_model_candidates(task_id, [
            {"class_code": "transverse_crack", "bbox": BOX_A, "score": 0.82},
            {"class_code": "pothole", "bbox": BOX_B, "score": 0.61},
        ])
        self.assertEqual(inserted, 2)
        task = self.repo.get_task(task_id)
        self.assertEqual(task["prelabel_state"], "done")

        # 人工提交不含模型候选 → 候选必须保留（不被覆盖）
        self.repo.replace_annotations(task_id, [
            {"class_code": "garbage", "kind": "bbox", "bbox": {"x1": 0.05, "y1": 0.05, "x2": 0.25, "y2": 0.25}}])
        sources = {row["source"] for row in self.repo.list_annotations(task_id=task_id)}
        self.assertEqual(sources, {"model", "human"})

        adopted = self.repo.adopt_model_candidates(task_id, actor="tester")
        self.assertEqual(adopted, 2)
        sources = {row["source"] for row in self.repo.list_annotations(task_id=task_id)}
        self.assertEqual(sources, {"model_edited", "human"})

        # 采纳后再提交：被删除的 model_edited 会被软删
        result = self.repo.replace_annotations(task_id, [
            {"class_code": "garbage", "kind": "bbox", "bbox": {"x1": 0.05, "y1": 0.05, "x2": 0.25, "y2": 0.25}}])
        self.assertEqual(result["deleted"], 2)
        self.assertEqual(len(self.repo.list_annotations(task_id=task_id)), 1)

    def test_submit_and_review_flow(self) -> None:
        task_id = self._first_task()
        self.repo.replace_annotations(task_id, [{"class_code": "pothole", "kind": "bbox", "bbox": BOX_A}])
        submitted = self.repo.submit_task(task_id, actor="tester")
        self.assertEqual(submitted["status"], "annotated")
        self.assertIsNotNone(submitted["submitted_at"])

        with self.assertRaises(ValueError):
            self.repo.add_review(task_id, decision="reject", reviewer="tester")  # 缺原因码

        review = self.repo.add_review(task_id, decision="reject", reason_code="loose_box",
                                      note="框偏松", reviewer="tester")
        self.assertEqual(review["decision"], "reject")
        self.assertEqual(self.repo.get_task(task_id)["status"], "annotating")

        self.repo.submit_task(task_id, actor="tester")
        self.repo.add_review(task_id, decision="approve", reviewer="tester")
        self.assertEqual(self.repo.get_task(task_id)["status"], "approved")
        stats = self.repo.review_stats()
        self.assertEqual(stats["total"], 2)
        self.assertAlmostEqual(stats["approve_rate"], 0.5)

    def test_class_registry_is_extensible(self) -> None:
        created = self.repo.add_class("water_puddle", "积水", "Water Puddle", color="#00aaff",
                                      order_index=10, actor="tester")
        self.assertEqual(created["code"], "water_puddle")
        self.assertIn("water_puddle", [row["code"] for row in self.repo.list_classes(active_only=True)])
        self.assertTrue(self.repo.set_class_active("water_puddle", False, actor="tester"))
        self.assertNotIn("water_puddle", [row["code"] for row in self.repo.list_classes(active_only=True)])

    def test_task_detail_contains_image_and_annotations(self) -> None:
        task_id = self._first_task()
        self.repo.replace_annotations(task_id, [{"class_code": "transverse_crack", "kind": "bbox", "bbox": BOX_A}])
        detail = self.repo.get_task_detail(task_id)
        self.assertEqual(detail["id"], task_id)
        self.assertIn("image", detail)
        self.assertEqual(len(detail["annotations"]), 1)
        self.assertEqual(detail["annotations"][0]["bbox"], BOX_A)

    def test_missing_task_raises(self) -> None:
        with self.assertRaises(KeyError):
            self.repo.replace_annotations(999_999, [])


if __name__ == "__main__":
    unittest.main()
