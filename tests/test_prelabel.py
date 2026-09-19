"""M2 测试：检测器抽象、切片合并、预标注编排、掩膜辅助、质量看板、HTTP 端点。

全部用假检测器/假掩膜器注入，**不需要安装 torch/ultralytics**（真机验证见 scripts/e2e_m2.py）。
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from rdinspect.api.app import create_app
from rdinspect.config import Config
from rdinspect.core.ingest import import_path
from rdinspect.prelabel.detector import (
    Detection,
    DetectorUnavailable,
    map_class,
    model_version_of,
    to_normalized_bbox,
)
from rdinspect.prelabel.metrics import prelabel_metrics
from rdinspect.prelabel.sahi import iou as det_iou
from rdinspect.prelabel.sahi import nms, predict_sliced, should_slice
from rdinspect.prelabel.sam import mask_metrics, mask_to_bbox, save_mask, load_mask
from rdinspect.prelabel.service import attach_mask_for_annotation, prelabel_tasks

from helpers import make_config, open_repo, synth_images

ALIASES = {"d10": "transverse_crack", "d00": "longitudinal_crack", "d40": "pothole",
           "garbage": "garbage", "transverse crack": "transverse_crack"}


class FakeDetector:
    """确定性假检测器：每张图（或每个切片）在固定相对位置输出一个框。"""

    def __init__(self, class_name: str = "D10", box_fraction: tuple[float, float, float, float] = (0.1, 0.2, 0.5, 0.6),
                 score: float = 0.9, conf: float = 0.25, iou: float = 0.5, weights: str = "fake.pt") -> None:
        self.class_name = class_name
        self.box_fraction = box_fraction
        self.score = score
        self.conf = conf
        self.iou = iou
        self.weights = weights
        self.device = "cpu"
        self.imgsz = 640
        self.names = {0: class_name}
        self.calls = 0

    def predict(self, image: Image.Image) -> list[Detection]:
        self.calls += 1
        width, height = image.size
        x1, y1, x2, y2 = self.box_fraction
        return [Detection(0, self.class_name, self.score,
                          (x1 * width, y1 * height, x2 * width, y2 * height))]


class FakeMasker:
    """假 SAM：返回一个位于给定框内的矩形掩膜。"""

    def __init__(self) -> None:
        self.calls = 0

    def mask_for_box(self, image: Image.Image, bbox_px: tuple[float, float, float, float]) -> np.ndarray:
        self.calls += 1
        mask = np.zeros((image.height, image.width), dtype=bool)
        x1, y1, x2, y2 = (int(round(v)) for v in bbox_px)
        mask[max(0, y1):min(image.height, y2), max(0, x1):min(image.width, x2)] = True
        return mask


class TestDetectorHelpers(unittest.TestCase):
    def test_map_class_aliases_and_substring(self) -> None:
        codes = ["transverse_crack", "longitudinal_crack", "pothole", "garbage"]
        self.assertEqual(map_class("D10", ALIASES, codes), "transverse_crack")
        self.assertEqual(map_class("d10", ALIASES, codes), "transverse_crack")
        self.assertEqual(map_class("Transverse Crack", ALIASES, codes), "transverse_crack")
        self.assertIsNone(map_class("person", ALIASES, codes))
        self.assertIsNone(map_class("d10", ALIASES, ["pothole"]), "目标类别不存在于注册表时必须丢弃")

    def test_to_normalized_bbox_clamps_and_filters(self) -> None:
        bbox = to_normalized_bbox((-10, -10, 50, 40), 100, 100)
        self.assertEqual(bbox, {"x1": 0.0, "y1": 0.0, "x2": 0.5, "y2": 0.4})
        self.assertIsNone(to_normalized_bbox((10, 10, 11, 11), 100, 100), "过小的框应丢弃")
        self.assertIsNone(to_normalized_bbox((90, 90, 200, 200), 100, 100, min_side_px=20))

    def test_model_version_of_is_stable(self) -> None:
        name, version = model_version_of("yolo11n.pt")
        self.assertEqual(name, "yolo11n")
        self.assertEqual(len(version), 8)
        self.assertEqual(model_version_of("yolo11n.pt"), (name, version))


class TestSlicing(unittest.TestCase):
    def test_iou_and_nms_are_class_aware(self) -> None:
        self.assertAlmostEqual(det_iou((0, 0, 10, 10), (0, 0, 10, 10)), 1.0)
        self.assertEqual(det_iou((0, 0, 10, 10), (20, 20, 30, 30)), 0.0)
        same_class = [
            Detection(0, "a", 0.9, (0, 0, 10, 10)),
            Detection(0, "a", 0.6, (1, 1, 11, 11)),   # 与上面重叠 → 应被抑制
            Detection(0, "a", 0.5, (20, 20, 30, 30)),  # 不重叠 → 保留
        ]
        kept = nms(same_class, 0.5)
        # 输出按 (y1, x1) 稳定排序：上方的框在前
        self.assertEqual([round(det.score, 2) for det in kept], [0.9, 0.5])

        cross_class = [
            Detection(0, "a", 0.9, (0, 0, 10, 10)),
            Detection(1, "b", 0.6, (0, 0, 10, 10)),  # 不同类：不互相抑制
        ]
        self.assertEqual(len(nms(cross_class, 0.5)), 2)

    def test_should_slice_modes(self) -> None:
        class Tile:
            def __init__(self, enabled):
                self.enabled = enabled

        small = Image.new("RGB", (800, 600))
        large = Image.new("RGB", (4000, 3000))
        self.assertFalse(should_slice(large, Tile(False)))
        self.assertTrue(should_slice(large, Tile(True)))
        self.assertFalse(should_slice(small, Tile("auto")))
        self.assertTrue(should_slice(large, Tile("auto")))

    def test_predict_sliced_merges_overlapping_tiles(self) -> None:
        """用"红色标记"模拟真实目标：它落在多个重叠切片里，合并后应只剩一个框。"""

        class Tile:
            enabled = True
            size = 256
            overlap = 0.25
            merge_iou = 0.5

        class MarkerDetector:
            names = {0: "D10"}
            conf = 0.25
            iou = 0.5
            weights = "marker.pt"
            device = "cpu"
            imgsz = 640

            def predict(self, image: Image.Image):
                array = np.asarray(image.convert("RGB"))
                mask = (array[:, :, 0] > 200) & (array[:, :, 1] < 60) & (array[:, :, 2] < 60)
                if not mask.any():
                    return []
                ys, xs = np.nonzero(mask)
                return [Detection(0, "D10", 0.9,
                                  (float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)))]

        image = Image.new("RGB", (512, 512), (70, 70, 72))
        for x in range(300, 340):                     # 一个 40×30 的"病害"
            for y in range(150, 180):
                image.putpixel((x, y), (255, 0, 0))

        detections, tiles = predict_sliced(image, MarkerDetector(), tile_config=Tile())
        self.assertGreater(tiles, 1, "大图应切片")
        self.assertEqual(len(detections), 1, "同一目标被多个切片命中后必须合并为一个")
        x1, y1, x2, y2 = detections[0].bbox
        self.assertAlmostEqual(x1, 300, delta=2)      # 坐标已从切片局部平移回原图
        self.assertAlmostEqual(y1, 150, delta=2)
        self.assertAlmostEqual(x2, 340, delta=2)

    def test_predict_sliced_small_image_uses_single_pass(self) -> None:
        class Tile:
            enabled = "auto"
            size = 1024
            overlap = 0.2
            merge_iou = 0.5

        detector = FakeDetector()
        detections, tiles = predict_sliced(Image.new("RGB", (320, 240)), detector, tile_config=Tile())
        self.assertEqual(tiles, 1)
        self.assertEqual(len(detections), 1)
        self.assertEqual(detector.calls, 1)


class TestPrelabelService(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmp.name)
        self.config = make_config(self.tmp_path)
        self.config.prelabel.class_aliases = dict(ALIASES)
        self.conn, self.repo = open_repo(self.config)
        source = self.config.data_dir / "inbox" / "photos"
        synth_images(source, count=4)
        import_path(self.config, self.repo, source, kind="photo")

    def tearDown(self) -> None:
        self.conn.close()
        self._tmp.cleanup()

    def test_prelabel_writes_candidates_and_run(self) -> None:
        detector = FakeDetector()
        report = prelabel_tasks(self.config, self.repo, limit=4, detector=detector, actor="test")
        self.assertEqual(report.requested, 4)
        self.assertEqual(report.processed, 4)
        self.assertEqual(report.candidates, 4)
        self.assertGreater(report.run_id or 0, 0)

        run = self.repo.get_run(report.run_id)
        self.assertEqual(run["kind"], "prelabel")
        self.assertEqual(run["status"], "succeeded")
        self.assertIn("candidates", run["metrics_json"])

        tasks = self.repo.list_tasks(status="prelabeled", limit=10)
        self.assertEqual(len(tasks), 4)
        self.assertTrue(all(task["prelabel_state"] == "done" for task in tasks))

        # 模型注册表登记了权重
        models = self.repo.list_model_versions(task="detection")
        self.assertEqual(len(models), 1)
        self.assertEqual(models[0]["weights_path"], "fake.pt")

        # 候选的来源、分数与坐标
        task_id = tasks[0]["id"]
        candidates = [row for row in self.repo.list_annotations(task_id=task_id) if row["source"] == "model"]
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["class_code"], "transverse_crack")
        self.assertAlmostEqual(candidates[0]["score"], 0.9, places=3)
        self.assertAlmostEqual(candidates[0]["bbox"]["x1"], 0.1, places=3)

    def test_prelabel_is_idempotent_by_run_key(self) -> None:
        first = prelabel_tasks(self.config, self.repo, limit=2, detector=FakeDetector(), actor="test")
        second = prelabel_tasks(self.config, self.repo, limit=2, detector=FakeDetector(), actor="test")
        self.assertEqual(first.run_id, second.run_id, "相同参数必须复用同一 run")
        self.assertEqual(self.repo.list_model_versions(), self.repo.list_model_versions())
        candidates = self.repo.conn.execute(
            "SELECT COUNT(*) AS n FROM annotations WHERE source = 'model'").fetchone()["n"]
        self.assertEqual(candidates, 2, "重复预标注不得产生重复候选（同框去重）")

    def test_unmapped_classes_are_counted_and_dropped(self) -> None:
        report = prelabel_tasks(self.config, self.repo, limit=2,
                                detector=FakeDetector(class_name="person"), actor="test")
        self.assertEqual(report.candidates, 0)
        self.assertEqual(report.unmapped, 2)
        self.assertEqual(self.repo.conn.execute(
            "SELECT COUNT(*) AS n FROM annotations WHERE source = 'model'").fetchone()["n"], 0)

    def test_disabled_prelabel_raises_unavailable(self) -> None:
        self.config.prelabel.enabled = False
        with self.assertRaises(DetectorUnavailable):
            prelabel_tasks(self.config, self.repo, limit=1, actor="test")

    def test_missing_image_file_is_skipped_without_crashing(self) -> None:
        task = self.repo.list_tasks(status="pending", limit=1)[0]
        image = self.repo.get_image(task["image_id"])
        self.config.abs_data_path(image["path"]).unlink()
        report = prelabel_tasks(self.config, self.repo, limit=1, detector=FakeDetector(), actor="test")
        self.assertEqual(report.processed, 0)
        self.assertEqual(report.skipped, 1)
        self.assertEqual(self.repo.get_task(task["id"])["prelabel_state"], "failed")

    def test_mask_attachment_via_marker_creates_mask_annotation(self) -> None:
        self.repo.add_model_candidates(1, [])
        task = self.repo.list_tasks(status="pending", limit=1)[0]
        result = self.repo.replace_annotations(task["id"], [
            {"class_code": "transverse_crack", "kind": "bbox",
             "bbox": {"x1": 0.1, "y1": 0.2, "x2": 0.4, "y2": 0.5}}])
        annotation_id = result["annotations"][0]["id"]

        masker = FakeMasker()
        payload = attach_mask_for_annotation(self.config, self.repo, annotation_id, masker=masker)
        self.assertEqual(masker.calls, 1)
        self.assertTrue(payload["mask_path"].startswith("masks/"))
        mask_file = self.config.abs_data_path(payload["mask_path"])
        self.assertTrue(mask_file.exists())
        metrics = payload["metrics"]
        self.assertGreater(metrics["area_ratio"], 0)
        self.assertGreater(metrics["width_px"], 0)

        masks = self.repo.mask_annotations(task_id=task["id"])
        self.assertEqual(len(masks), 1)
        self.assertEqual(masks[0]["kind"], "mask")
        self.assertEqual(masks[0]["source"], "human")
        self.assertIn("width_px", masks[0]["mask_metrics"])

    def test_mask_helpers(self) -> None:
        mask = np.zeros((20, 40), dtype=bool)
        mask[5:8, 3:33] = True
        metrics = mask_metrics(mask, pixel_size_m=0.01)
        self.assertAlmostEqual(metrics["length_px"], 30.0, places=1)
        self.assertAlmostEqual(metrics["width_px"], 3.0, places=1)
        self.assertAlmostEqual(metrics["length_m"], 0.3, places=3)
        bbox = mask_to_bbox(mask, 40, 20)
        self.assertEqual(bbox, {"x1": 0.075, "y1": 0.25, "x2": 0.825, "y2": 0.4})

        with tempfile.TemporaryDirectory() as tmp:
            config = make_config(Path(tmp))
            rel = save_mask(config, 7, mask)
            self.assertTrue(config.abs_data_path(rel).exists())
            self.assertTrue(load_mask(config.abs_data_path(rel)).any())


class TestPrelabelMetrics(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmp.name)
        self.config = make_config(self.tmp_path)
        self.config.prelabel.class_aliases = dict(ALIASES)
        self.conn, self.repo = open_repo(self.config)
        source = self.config.data_dir / "inbox" / "photos"
        synth_images(source, count=3)
        import_path(self.config, self.repo, source, kind="photo")
        prelabel_tasks(self.config, self.repo, limit=3, detector=FakeDetector(), actor="test")

    def tearDown(self) -> None:
        self.conn.close()
        self._tmp.cleanup()

    def test_adoption_and_ignore_metrics(self) -> None:
        tasks = self.repo.list_tasks(status="prelabeled", limit=10)
        # 任务 1：采纳（并微调位置，产生 human 框 → 计入一致性）
        task_one = tasks[0]
        candidate = [row for row in self.repo.list_annotations(task_id=task_one["id"])
                     if row["source"] == "model"][0]
        self.repo.adopt_model_candidates(task_one["id"], actor="test")
        self.repo.replace_annotations(task_one["id"], [
            {"class_code": "transverse_crack", "kind": "bbox",
             "bbox": {"x1": 0.12, "y1": 0.22, "x2": 0.45, "y2": 0.55}}])
        # 任务 2：忽略候选
        task_two = tasks[1]
        ignored_candidate = [row for row in self.repo.list_annotations(task_id=task_two["id"])
                             if row["source"] == "model"][0]
        self.assertTrue(self.repo.delete_annotation(ignored_candidate["id"], actor="test"))
        self.assertFalse(self.repo.delete_annotation(ignored_candidate["id"], actor="test"),
                         "重复删除应返回 False")

        metrics = prelabel_metrics(self.repo)
        self.assertEqual(metrics["tasks_prelabeled"], 3)
        self.assertEqual(metrics["candidates_total"], 1)   # 任务 3 的候选仍未处理
        self.assertEqual(metrics["adopted"], 1)
        self.assertEqual(metrics["ignored"], 1)
        self.assertAlmostEqual(metrics["adoption_rate"], 1 / 3, places=4)
        self.assertIsNotNone(metrics["model_human_iou_mean"])
        self.assertGreater(metrics["model_human_iou_mean"], 0.3)
        self.assertEqual(metrics["agreement_sampled_tasks"], 1)
        self.assertIn("done", metrics["prelabel_states"])
        del candidate


class TestPrelabelApi(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmp.name)
        self.config = make_config(self.tmp_path)
        self.config.prelabel.class_aliases = dict(ALIASES)
        self.conn, self.repo = open_repo(self.config)
        source = self.config.data_dir / "inbox" / "photos"
        synth_images(source, count=3)
        import_path(self.config, self.repo, source, kind="photo")
        self.masker = FakeMasker()
        self.app = create_app(self.config, detector_factory=lambda _cfg, _w: FakeDetector(),
                              masker_factory=lambda _cfg: self.masker)
        self.client = TestClient(self.app)

    def tearDown(self) -> None:
        self.client.close()
        self.conn.close()
        self._tmp.cleanup()

    def test_batch_prelabel_and_task_prelabel(self) -> None:
        response = self.client.post("/api/prelabel/batches", json={"limit": 2, "status": "pending"})
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["requested"], 2)
        self.assertEqual(payload["report"]["candidates"], 2)
        self.assertEqual(payload["run"]["kind"], "prelabel")

        remaining = self.client.get("/api/tasks?status=pending&limit=10").json()["items"]
        self.assertEqual(len(remaining), 1)
        single = self.client.post(f"/api/tasks/{remaining[0]['id']}/prelabel", json={"limit": 1})
        self.assertEqual(single.status_code, 200)
        self.assertEqual(single.json()["candidates"], 1)

    def test_candidate_lifecycle_over_http(self) -> None:
        self.client.post("/api/prelabel/batches", json={"limit": 1})
        task = self.client.get("/api/tasks?status=prelabeled&limit=1").json()["items"][0]
        detail = self.client.get(f"/api/tasks/{task['id']}").json()
        candidates = [row for row in detail["annotations"] if row["source"] == "model"]
        self.assertEqual(len(candidates), 1)
        self.assertIn("bbox", candidates[0])

        # 忽略候选
        deleted = self.client.delete(f"/api/annotations/{candidates[0]['id']}")
        self.assertEqual(deleted.status_code, 200)
        self.assertEqual(self.client.delete(f"/api/annotations/{candidates[0]['id']}").status_code, 404)

        metrics = self.client.get("/api/prelabel/metrics").json()
        self.assertEqual(metrics["ignored"], 1)
        self.assertEqual(metrics["candidates_total"], 0)

    def test_mask_assist_over_http(self) -> None:
        task = self.client.get("/api/tasks?status=pending&limit=1").json()["items"][0]
        put = self.client.put(f"/api/tasks/{task['id']}/annotations", json={
            "annotations": [{"class_code": "transverse_crack", "kind": "bbox",
                             "bbox": {"x1": 0.1, "y1": 0.2, "x2": 0.4, "y2": 0.5}}]})
        annotation_id = put.json()["annotations"][0]["id"]
        created = self.client.post(f"/api/annotations/{annotation_id}/mask")
        self.assertEqual(created.status_code, 200)
        body = created.json()
        self.assertGreater(body["metrics"]["area_ratio"], 0)
        mask_annotation_id = body["mask_annotation_id"]
        self.assertGreater(mask_annotation_id, annotation_id)

        # 掩膜 id 直接可寻址；用框标注 id 也会回退到同任务同类最新掩膜（标注台便利路径）
        for target in (mask_annotation_id, annotation_id):
            mask_response = self.client.get(f"/api/annotations/{target}/mask")
            self.assertEqual(mask_response.status_code, 200)
            self.assertEqual(mask_response.headers["content-type"], "image/png")
        self.assertEqual(self.client.get("/api/annotations/999999/mask").status_code, 404)

    def test_models_endpoint_lists_detector(self) -> None:
        self.client.post("/api/prelabel/batches", json={"limit": 1})
        models = self.client.get("/api/models?task=detection").json()
        self.assertEqual(len(models), 1)
        self.assertEqual(models[0]["weights_path"], "fake.pt")

    def test_prelabel_returns_501_when_ml_unavailable(self) -> None:
        """未注入工厂且未安装 ML 依赖时必须是 501（人工标注不受影响）。"""
        from rdinspect.prelabel.detector import ml_available

        if ml_available():
            self.skipTest("本机已安装 ultralytics，501 分支不适用")
        plain_app = create_app(self.config)
        with TestClient(plain_app) as client:
            response = client.post("/api/prelabel/batches", json={"limit": 1})
            self.assertEqual(response.status_code, 501)
            self.assertIn("ultralytics", response.json()["message"])


class TestRuntimeDirs(unittest.TestCase):
    def test_runtime_dirs_are_inside_data_dir(self) -> None:
        """ultralytics/matplotlib 的配置必须落在 data_dir（否则受限环境下导入即失败）。"""
        import os

        with tempfile.TemporaryDirectory() as tmp:
            config = make_config(Path(tmp))
            for key in ("YOLO_CONFIG_DIR", "MPLCONFIGDIR"):
                os.environ.pop(key, None)
            config.export_runtime_env()
            self.assertEqual(os.environ["YOLO_CONFIG_DIR"], str(config.ultralytics_dir))
            self.assertEqual(os.environ["MPLCONFIGDIR"], str(config.mpl_dir))
            for path in (config.weights_dir, config.ultralytics_dir, config.mpl_dir):
                self.assertTrue(path.is_dir(), f"{path} 应已创建")
                self.assertTrue(str(path).startswith(str(config.data_dir)))

    def test_bare_weight_names_are_localised(self) -> None:
        from rdinspect.prelabel.service import _localize_weights

        with tempfile.TemporaryDirectory() as tmp:
            config = make_config(Path(tmp))
            self.assertEqual(_localize_weights(config, "yolo11n.pt"),
                             str(config.weights_dir / "yolo11n.pt"))
            explicit = tmp + "/custom/best.pt"
            self.assertEqual(_localize_weights(config, explicit), explicit)


class TestApiWithoutMl(unittest.TestCase):
    def test_config_defaults_keep_prelabel_usable(self) -> None:
        config = Config(raw={}, data_dir=Path(tempfile.mkdtemp()) / "data", allowed_roots=())
        self.assertTrue(config.prelabel.enabled)
        self.assertEqual(config.prelabel.device, "auto")
        self.assertIn("d10", config.prelabel.class_aliases)


if __name__ == "__main__":
    unittest.main()
