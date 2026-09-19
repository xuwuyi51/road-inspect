"""HTTP 契约测试：端点形状、状态码与错误语义（docs/06-api-spec.md）。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from rdinspect.api.app import create_app
from rdinspect.core.ingest import import_path

from helpers import make_config, open_repo, synth_images

BOX = {"x1": 0.10, "y1": 0.20, "x2": 0.40, "y2": 0.35}


class TestApi(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        tmp_path = Path(self._tmp.name)
        self.config = make_config(tmp_path)
        self.conn, self.repo = open_repo(self.config)
        source = self.config.data_dir / "inbox" / "photos"
        synth_images(source, count=3)
        self.report = import_path(self.config, self.repo, source, kind="photo")
        self.app = create_app(self.config)
        self.client = TestClient(self.app)

    def tearDown(self) -> None:
        self.client.close()
        self.conn.close()
        self._tmp.cleanup()

    def _lease(self) -> int:
        response = self.client.post("/api/tasks/lease",
                                    json={"count": 1, "status": "pending", "assignee": "test"})
        self.assertEqual(response.status_code, 200)
        leased = response.json()["leased"]
        self.assertEqual(len(leased), 1)
        return leased[0]["id"]

    def test_health_and_classes(self) -> None:
        health = self.client.get("/api/health")
        self.assertEqual(health.status_code, 200)
        self.assertEqual(health.json()["status"], "ok")
        self.assertEqual(health.json()["images"], 3)

        classes = self.client.get("/api/classes").json()
        self.assertEqual([cls["code"] for cls in classes][:4],
                         ["longitudinal_crack", "transverse_crack", "pothole", "garbage"])
        self.assertTrue(all("color" in cls and "order_index" in cls for cls in classes))

    def test_annotation_contract_shape_and_error_codes(self) -> None:
        task_id = self._lease()
        response = self.client.put(f"/api/tasks/{task_id}/annotations",
                                   json={"annotations": [{"class_code": "pothole", "kind": "bbox", "bbox": BOX}]})
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual((payload["added"], payload["deleted"]), (1, 0))
        annotation = payload["annotations"][0]
        self.assertEqual(set(annotation) >= {"id", "class_code", "kind", "bbox", "source", "score"}, True)
        self.assertEqual(annotation["bbox"], BOX)
        self.assertEqual(annotation["source"], "human")
        self.assertNotIn("bbox_x1", annotation, "不应泄露数据库原始列")

        bad_box = self.client.put(f"/api/tasks/{task_id}/annotations",
                                  json={"annotations": [{"class_code": "pothole", "kind": "bbox",
                                                         "bbox": {"x1": 0.5, "y1": 0.5, "x2": 0.1, "y2": 0.2}}]})
        self.assertEqual(bad_box.status_code, 400)
        unknown = self.client.put(f"/api/tasks/{task_id}/annotations",
                                  json={"annotations": [{"class_code": "nope", "kind": "bbox", "bbox": BOX}]})
        self.assertEqual(unknown.status_code, 400)
        self.assertEqual(self.client.get("/api/tasks/999999").status_code, 404)

    def test_submit_and_review_semantics(self) -> None:
        task_id = self._lease()
        self.client.put(f"/api/tasks/{task_id}/annotations",
                        json={"annotations": [{"class_code": "garbage", "kind": "bbox", "bbox": BOX}]})
        submitted = self.client.post(f"/api/tasks/{task_id}/submit")
        self.assertEqual(submitted.status_code, 200)
        self.assertEqual(submitted.json()["status"], "annotated")

        missing_reason = self.client.post(f"/api/tasks/{task_id}/review", json={"decision": "reject"})
        self.assertEqual(missing_reason.status_code, 400)
        approved = self.client.post(f"/api/tasks/{task_id}/review",
                                    json={"decision": "approve", "reviewer": "test"})
        self.assertEqual(approved.status_code, 200)
        self.assertEqual(approved.json()["task"]["status"], "approved")

    def test_dataset_draft_freeze_conflict_and_export(self) -> None:
        for _ in range(3):
            task_id = self._lease()
            self.client.put(f"/api/tasks/{task_id}/annotations",
                            json={"annotations": [{"class_code": "transverse_crack", "kind": "bbox", "bbox": BOX}]})
            self.client.post(f"/api/tasks/{task_id}/submit")
            self.client.post(f"/api/tasks/{task_id}/review", json={"decision": "approve", "reviewer": "test"})

        draft = self.client.post("/api/datasets", json={
            "name": "ds-api-01", "filter": {"review_status": "approved"},
            "split": {"train": 1.0, "val": 0.0, "test": 0.0, "seed": 1, "group_by": "none"}})
        self.assertEqual(draft.status_code, 201)
        dataset_id = draft.json()["id"]
        self.assertEqual(draft.json()["stats"]["images"], 3)

        duplicate = self.client.post("/api/datasets", json={"name": "ds-api-01"})
        self.assertEqual(duplicate.status_code, 409)

        frozen = self.client.post(f"/api/datasets/{dataset_id}/freeze")
        self.assertEqual(frozen.status_code, 200)
        self.assertEqual(frozen.json()["status"], "frozen")
        self.assertEqual(len(frozen.json()["manifest_hash"]), 64)
        self.assertIn("yolo", frozen.json()["exports"])

        again = self.client.post(f"/api/datasets/{dataset_id}/freeze")
        self.assertEqual(again.status_code, 409, "重复冻结必须是 409（状态冲突）")

        exported = self.client.post(f"/api/datasets/{dataset_id}/export", json={"formats": ["coco"]})
        self.assertEqual(exported.status_code, 200)
        self.assertIn("coco", exported.json()["exports"])

    def test_ingest_reports_and_image_streams(self) -> None:
        batches = self.client.get("/api/ingest/batches").json()
        self.assertGreaterEqual(len(batches["items"]), 1)
        batch_id = batches["items"][0]["id"]
        detail = self.client.get(f"/api/ingest/batches/{batch_id}").json()
        self.assertEqual(detail["outcome_counts"].get("added"), 3)

        images = self.client.get("/api/images?limit=5").json()["items"]
        self.assertTrue(images)
        image_id = images[0]["id"]
        original = self.client.get(f"/api/images/{image_id}/file")
        self.assertEqual(original.status_code, 200)
        self.assertEqual(original.headers["content-type"], "image/jpeg")
        thumb = self.client.get(f"/api/images/{image_id}/file?thumb=1")
        self.assertEqual(thumb.headers["content-type"], "image/webp")
        self.assertEqual(self.client.get("/api/images/999999/file").status_code, 404)

    def test_index_page_is_served(self) -> None:
        index = self.client.get("/")
        self.assertEqual(index.status_code, 200)
        self.assertIn("标注台", index.text)

    def test_ingest_endpoint_imports_directory(self) -> None:
        source = self.config.data_dir / "inbox" / "second"
        synth_images(source, count=2, seed=9)
        response = self.client.post("/api/ingest/batches",
                                    json={"kind": "photo", "source_dir": str(source), "note": "api"})
        self.assertEqual(response.status_code, 202)
        self.assertEqual(response.json()["added"], 2)

        outside = self.config.data_dir.parent / "outside"
        synth_images(outside, count=1, seed=5)
        rejected = self.client.post("/api/ingest/batches",
                                    json={"kind": "photo", "source_dir": str(outside)})
        self.assertEqual(rejected.status_code, 400, "白名单外的目录必须拒绝")

    def test_stats_endpoints(self) -> None:
        overview = self.client.get("/api/stats/overview").json()
        self.assertEqual(overview["images"], 3)
        self.assertIn("pending", overview["tasks"])
        csv_response = self.client.get("/api/stats/export?format=csv")
        self.assertEqual(csv_response.status_code, 200)
        self.assertTrue(csv_response.text.startswith("image_id,captured_at"))

    def test_unimplemented_endpoints_return_501(self) -> None:
        """训练/模型晋级属 M3：当前必须明确 501，而不是静默 404 或假装成功。"""
        response = self.client.post("/api/runs", json={"kind": "train", "dataset_id": 1})
        self.assertEqual(response.status_code, 501)
        self.assertIn("M3", response.json()["message"])


if __name__ == "__main__":
    unittest.main()
