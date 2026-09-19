"""数据集选样、划分确定性、冻结与导出测试（ADR-0005）。"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from rdinspect.core import datasets as ds
from rdinspect.core.ingest import import_path

from helpers import annotate_all, make_config, open_repo, synth_images

BOX = {"x1": 0.10, "y1": 0.20, "x2": 0.40, "y2": 0.35}


class TestDatasets(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmp.name)
        self.config = make_config(self.tmp_path)
        self.conn, self.repo = open_repo(self.config)
        source = self.config.data_dir / "inbox" / "photos"
        synth_images(source, count=6)
        import_path(self.config, self.repo, source, kind="photo")

    def tearDown(self) -> None:
        self.conn.close()
        self._tmp.cleanup()

    def test_select_requires_approved_by_default(self) -> None:
        # 仅导入（pending）：任何条件都选不到样本
        self.assertEqual(ds.select_images(self.repo, {"review_status": "approved"}), [])
        self.assertEqual(ds.select_images(self.repo, {"review_status": "annotated"}), [])
        # 标注并提交但未复核：annotated 可见，approved 仍为空
        annotate_all(self.repo, class_code="transverse_crack", bbox=BOX, approve=False)
        self.assertEqual(len(ds.select_images(self.repo, {"review_status": "annotated"})), 6)
        self.assertEqual(ds.select_images(self.repo, {"review_status": "approved"}), [])
        # 复核通过后 approved 可见
        for task in self.repo.list_tasks(status="annotated", limit=100):
            self.repo.add_review(task["id"], decision="approve", reviewer="test")
        self.assertEqual(len(ds.select_images(self.repo, {"review_status": "approved"})), 6)

    def test_split_is_deterministic_and_disjoint(self) -> None:
        annotate_all(self.repo, class_code="pothole", bbox=BOX, approve=True)
        rows = ds.select_images(self.repo, {"review_status": "approved"})
        first = ds.assign_splits(rows, {"train": 0.6, "val": 0.2, "test": 0.2, "seed": 42, "group_by": "gps_grid"})
        second = ds.assign_splits(list(reversed(rows)), {"train": 0.6, "val": 0.2, "test": 0.2, "seed": 42,
                                                         "group_by": "gps_grid"})
        self.assertEqual({row["id"]: split for row, split in first},
                         {row["id"]: split for row, split in second}, "划分必须与输入顺序无关")
        assignment = {row["id"]: split for row, split in first}
        self.assertEqual(len(assignment), 6)
        self.assertEqual(set(assignment.values()) - {"train", "val", "test"}, set())
        # 同一 GPS 网格的行必须落在同一 split（合成数据无 GPS → 每图独立，同样成立）
        groups: dict[str, set[str]] = {}
        for row, split in first:
            groups.setdefault(ds.group_key(row), set()).add(split)
        for splits in groups.values():
            self.assertEqual(len(splits), 1, "同一分组不得跨 split（防泄漏）")

    def test_draft_freeze_export_and_immutability(self) -> None:
        annotate_all(self.repo, class_code="garbage", bbox=BOX, approve=True)
        draft = ds.create_draft(self.config, self.repo, "ds-test-01",
                                {"review_status": "approved"},
                                {"train": 0.5, "val": 0.25, "test": 0.25, "seed": 7, "group_by": "gps_grid"})
        self.assertEqual(draft["status"], "draft")
        stats = draft["stats"]
        self.assertEqual(stats["images"], 6)
        self.assertEqual(sum(stats["splits"].values()), 6)
        self.assertEqual(stats["classes"].get("garbage"), 6)

        frozen = ds.freeze_dataset(self.config, self.repo, name="ds-test-01",
                                   export_formats=("yolo", "coco", "labelme"))
        self.assertEqual(frozen["status"], "frozen")
        manifest = frozen["manifest_hash"]
        self.assertEqual(len(manifest), 64)
        root = Path(frozen["root_path"])
        self.assertTrue((root / "manifest.sha256").exists())
        self.assertEqual((root / "manifest.sha256").read_text(encoding="utf-8").strip(), manifest)
        self.assertTrue((root / "data.yaml").exists())
        self.assertTrue((root / "coco.json").exists())
        self.assertTrue((root / "labelme").is_dir())
        labels = list((root / "labels").rglob("*.txt"))
        self.assertEqual(len(labels), 6)

        summary = json.loads((root / "dataset.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["manifest_hash"], manifest)
        self.assertEqual(summary["class_order"][:4],
                         ["longitudinal_crack", "transverse_crack", "pothole", "garbage"])

        # 冻结后不可再冻结（不可变）
        with self.assertRaises(ValueError):
            ds.freeze_dataset(self.config, self.repo, name="ds-test-01")

    def test_manifest_hash_is_stable_and_content_sensitive(self) -> None:
        annotate_all(self.repo, class_code="transverse_crack", bbox=BOX, approve=True)
        draft = ds.create_draft(self.config, self.repo, "ds-hash-01", {"review_status": "approved"},
                                {"train": 1.0, "val": 0.0, "test": 0.0, "seed": 1, "group_by": "none"})
        rows = ds.fetch_dataset_rows(self.repo, self.repo.dataset_items(draft["id"]))
        classes = self.repo.list_classes()
        order = [cls["code"] for cls in sorted(classes, key=lambda c: (c["order_index"], c["code"]))]
        hash_a = ds.compute_manifest_hash(rows, order)
        hash_b = ds.compute_manifest_hash(list(reversed(rows)), order)
        self.assertEqual(hash_a, hash_b, "清单哈希必须与顺序无关")

        # 改动一条标注坐标 → 哈希变化
        task_id = rows[0]["image"]["id"]
        task = self.repo.conn.execute("SELECT id FROM tasks WHERE image_id = ?", (task_id,)).fetchone()
        self.repo.replace_annotations(task["id"], [
            {"class_code": "transverse_crack", "kind": "bbox",
             "bbox": {"x1": 0.11, "y1": 0.21, "x2": 0.41, "y2": 0.36}}])
        rows2 = ds.fetch_dataset_rows(self.repo, self.repo.dataset_items(draft["id"]))
        self.assertNotEqual(hash_a, ds.compute_manifest_hash(rows2, order))

    def test_dataset_requires_samples_to_freeze(self) -> None:
        draft = ds.create_draft(self.config, self.repo, "ds-empty-01", {"review_status": "approved"}, {})
        with self.assertRaises(ValueError):
            ds.freeze_dataset(self.config, self.repo, name="ds-empty-01")
        del draft


if __name__ == "__main__":
    unittest.main()
