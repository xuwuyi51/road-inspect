"""M5 主动学习测试：选样打分、队列落地、门禁连续失败告警、API 契约。"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from rdinspect.active import scoring
from rdinspect.active import service as active_service

from helpers import make_config, open_repo, synth_images
from rdinspect.core.ingest import import_path

BOX = {"x1": 0.10, "y1": 0.20, "x2": 0.40, "y2": 0.35}


def fake_evaluation(*, confusion=None, per_class=None) -> dict:
    labels, matrix = confusion if confusion else (["a", "b", "__background__"],
                                                  [[8, 2, 0], [1, 7, 2], [3, 1, 0]])
    return {"internal": {"confusion_matrix": {"labels": labels, "matrix": matrix},
                         "precision_recall": {"per_class": per_class or {}}},
            "ultralytics": {"map50": 0.5, "per_class": {}}}


class TestScoringPure(unittest.TestCase):
    def test_hamming_hex(self) -> None:
        self.assertEqual(scoring.hamming_hex("0f", "0f"), 0)
        self.assertEqual(scoring.hamming_hex("0", "1"), 1)
        self.assertEqual(scoring.hamming_hex("zz", "0f"), 64, "非法哈希按最大距离处理")

    def test_confusion_weights_ignore_background(self) -> None:
        weights = scoring.confusion_weights(fake_evaluation())
        self.assertAlmostEqual(weights["a"], 0.2, places=6)
        # 行 [1,7,2]：错在 a 的 1 例算错误；最后一列（漏检）不计入"混淆"，由分类别召回覆盖
        self.assertAlmostEqual(weights["b"], 0.1, places=6)
        self.assertNotIn("__background__", weights)

    def test_confusion_weights_empty_when_no_matrix(self) -> None:
        self.assertEqual(scoring.confusion_weights({"internal": {}}), {})
        self.assertEqual(scoring.confusion_weights({}), {})

    def test_weak_classes_from_recall(self) -> None:
        evaluation = fake_evaluation(per_class={"a": {"recall": 0.4}, "b": {"recall": 0.9},
                                                "c": {"recall": None}})
        weak = scoring.weak_classes(evaluation, recall_threshold=0.6)
        self.assertAlmostEqual(weak["a"], 0.2, places=6)
        self.assertNotIn("b", weak)
        self.assertNotIn("c", weak, "没有召回数据的类不能猜")

    def test_uncertainty_conf_band(self) -> None:
        signals = scoring.ImageSignals(image_id=1, task_id=1,
                                       candidates=[{"class_code": "a", "score": 0.3},
                                                   {"class_code": "a", "score": 0.9}])
        score, detail = scoring.uncertainty_score(signals, conf_band=(0.25, 0.45))
        self.assertGreater(score, 0)
        self.assertEqual(detail["in_band"], 1)
        self.assertEqual(detail["evidence"], "band")
        high_conf = scoring.ImageSignals(image_id=2, task_id=2,
                                        candidates=[{"class_code": "a", "score": 0.95}])
        self.assertEqual(scoring.uncertainty_score(high_conf)[0], 0.0)

    def test_uncertainty_margin_evidence(self) -> None:
        signals = scoring.ImageSignals(image_id=1, task_id=1,
                                       candidates=[{"class_code": "a", "score": 0.8}],
                                       margin_evidence={"margins": [0.02, 0.5], "min_margin": 0.02})
        score, detail = scoring.uncertainty_score(signals, margin_threshold=0.1)
        self.assertGreater(score, 0)
        self.assertEqual(detail["evidence"], "margin")
        self.assertAlmostEqual(detail["margin_ratio"], 0.5, places=6)

    def test_uncertainty_empty_weight_is_opt_in(self) -> None:
        empty = scoring.ImageSignals(image_id=1, task_id=1, candidates=[])
        self.assertEqual(scoring.uncertainty_score(empty)[0], 0.0)
        self.assertGreater(scoring.uncertainty_score(empty, empty_weight=1.0)[0], 0.0)

    def test_error_score_uses_confusion_and_weak(self) -> None:
        signals = scoring.ImageSignals(image_id=1, task_id=1,
                                       candidates=[{"class_code": "a", "score": 0.9}])
        score, detail = scoring.error_score(signals, confusion={"a": 0.5}, weak={"a": 0.3})
        self.assertGreater(score, 0.4)
        self.assertEqual(detail["classes"], ["a"])
        self.assertEqual(scoring.error_score(signals)[0], 0.0)

    def test_base_score_and_priority_mapping(self) -> None:
        self.assertAlmostEqual(scoring.base_score(1.0, 1.0), 1.0, places=6)
        self.assertAlmostEqual(scoring.base_score(1.0, 0.0), 0.625, places=6)
        self.assertEqual(scoring.priority_from_score(1.0), 1)
        self.assertEqual(scoring.priority_from_score(0.0), 100)
        self.assertEqual(scoring.priority_from_score(0.5), 50)
        self.assertEqual(scoring.priority_from_score(2.0), 1, "越界分数被夹紧")

    def test_diversity_select_respects_phash_distance(self) -> None:
        def item(index: int, phash: str) -> scoring.QueueItem:
            return scoring.QueueItem(task_id=index, image_id=index, score=1.0 - index * 0.1,
                                     priority=1, reason="score", phash=phash)

        pool = [item(1, "0000000000000000"), item(2, "0000000000000000"),
                item(3, "ffffffffffffffff"), item(4, "00ff00ff00ff00ff")]
        picked = scoring.diversity_select(pool, phash_hamming=6, ratio=0.75)
        self.assertEqual([entry.image_id for entry in picked], [1, 3, 4],
                         "同簇只取一张，其余取代表；配额 3/4")
        filled = scoring.diversity_select(pool, phash_hamming=6, ratio=1.0)
        self.assertEqual([entry.image_id for entry in filled], [1, 3, 4, 2],
                         "配额大于可用簇数时按分数顺序补齐（并保持可解释顺序）")

    def test_select_queue_strategies(self) -> None:
        signals = [
            scoring.ImageSignals(image_id=1, task_id=1, phash="0000000000000000",
                                 candidates=[{"class_code": "a", "score": 0.30}]),
            scoring.ImageSignals(image_id=2, task_id=2, phash="0000000000000000",
                                 candidates=[{"class_code": "a", "score": 0.31}]),
            scoring.ImageSignals(image_id=3, task_id=3, phash="ffffffffffffffff",
                                 candidates=[{"class_code": "b", "score": 0.95}]),
        ]
        hybrid = scoring.select_queue(signals, strategy="hybrid", confusion={"b": 0.9})
        self.assertEqual(len(hybrid), 3)
        self.assertEqual(hybrid[0].priority, min(entry.priority for entry in hybrid))
        uncertainty_only = scoring.select_queue(signals, strategy="uncertainty", limit=2)
        self.assertEqual({entry.image_id for entry in uncertainty_only}, {1, 2})
        self.assertGreaterEqual(uncertainty_only[0].score, uncertainty_only[1].score)
        error_only = scoring.select_queue(signals, strategy="error", confusion={"b": 0.9})
        self.assertEqual(error_only[0].image_id, 3)
        diversity_only = scoring.select_queue(signals, strategy="diversity", phash_hamming=6)
        self.assertTrue(all(entry.reason == "diversity" for entry in diversity_only))
        random_a = scoring.select_queue(signals, strategy="random", limit=2, seed=7)
        random_b = scoring.select_queue(signals, strategy="random", limit=2, seed=7)
        self.assertEqual([entry.image_id for entry in random_a],
                         [entry.image_id for entry in random_b], "随机策略固定 seed 可复现")
        with self.assertRaises(ValueError):
            scoring.select_queue(signals, strategy="nope")

    def test_summarize_selection(self) -> None:
        signals = [scoring.ImageSignals(image_id=1, task_id=1,
                                        candidates=[{"class_code": "a", "score": 0.3}])]
        items = scoring.select_queue(signals, strategy="hybrid", confusion={"a": 0.5})
        summary = scoring.summarize_selection(items, total_candidates=5)
        self.assertEqual(summary["total_candidates"], 5)
        self.assertEqual(summary["selected"], 1)
        self.assertIsNotNone(summary["score"]["mean"])

    def test_budget_effect_verdicts(self) -> None:
        baseline = {"ultralytics": {"map50": 0.40, "per_class": {"a": {"mAP50": 0.4}}}}
        better = {"ultralytics": {"map50": 0.50, "per_class": {"a": {"mAP50": 0.55}}}}
        same = {"ultralytics": {"map50": 0.402, "per_class": {"a": {"mAP50": 0.4}}}}
        worse = {"ultralytics": {"map50": 0.30, "per_class": {"a": {"mAP50": 0.3}}}}
        record = scoring.budget_effect(baseline, better, labeled_images=100)
        self.assertAlmostEqual(record["delta_map50"], 0.10, places=6)
        self.assertAlmostEqual(record["per_100_images"], 0.10, places=6)
        self.assertEqual(record["verdict"], "主动学习更优")
        self.assertEqual(record["per_class_delta"]["a"], 0.15)
        self.assertEqual(scoring.budget_effect(baseline, same, labeled_images=50)["verdict"], "无显著差异")
        self.assertEqual(scoring.budget_effect(baseline, worse, labeled_images=200)["verdict"],
                         "基线更优（需复查选样策略）")

    def test_class_histogram(self) -> None:
        signals = [scoring.ImageSignals(image_id=1, task_id=1,
                                        candidates=[{"class_code": "a"}, {"class_code": "a"}]),
                   scoring.ImageSignals(image_id=2, task_id=2, candidates=[{"class_code": "b"}])]
        self.assertEqual(scoring.class_histogram(signals), {"a": 2, "b": 1})


class ActiveFixture(unittest.TestCase):
    """夹具：8 张图 + 待标注任务 + 模型候选 + 一次评估记录。"""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmp.name)
        self.config = make_config(self.tmp_path)
        self.conn, self.repo = open_repo(self.config)
        source = self.config.data_dir / "inbox" / "photos"
        synth_images(source, count=8, size=(160, 120))
        import_path(self.config, self.repo, source, kind="photo")

    def tearDown(self) -> None:
        self.conn.close()
        self._tmp.cleanup()

    def add_candidates(self, task_ids, *, score: float = 0.32,
                       class_code: str = "transverse_crack") -> None:
        for task_id in task_ids:
            self.repo.conn.execute(
                """INSERT INTO annotations(task_id, image_id, class_code, kind, bbox_x1, bbox_y1,
                                           bbox_x2, bbox_y2, source, score)
                   SELECT ?, image_id, ?, 'bbox', 0.1, 0.2, 0.4, 0.35, 'model', ? FROM tasks WHERE id = ?""",
                (task_id, class_code, score, task_id))
        self.repo.conn.commit()

    def register_model_with_evaluation(self, name: str = "m-road", version: str = "v1", *,
                                       status: str = "production", gate_passed=None,
                                       evaluation: dict | None = None) -> dict:
        weights = self.config.weights_dir / f"{name}-{version}.pt"
        weights.write_bytes(b"w")
        model = self.repo.upsert_model_version(name=name, version=version, status=status,
                                              weights_path=str(weights))
        if evaluation is not None:
            self.repo.update_model_version(int(model["id"]), metrics_json={"evaluation": evaluation})
        if gate_passed is not None:
            self.repo.update_model_version(
                int(model["id"]), gate_json={"passed": gate_passed,
                                             "reasons": [] if gate_passed else ["x"]})
        return self.repo.get_model_version(int(model["id"])) or model


class TestActiveService(ActiveFixture):
    def test_collect_signals_skips_labeled_images(self) -> None:
        signals = active_service.collect_signals(self.config, self.repo)
        self.assertEqual(len(signals), 8)
        # 直接插人工框（避免 replace_annotations 把任务推进到 annotating 而离开 pending）
        self.repo.conn.execute(
            """INSERT INTO annotations(task_id, image_id, class_code, kind, bbox_x1, bbox_y1,
                                       bbox_x2, bbox_y2, source)
               SELECT ?, image_id, 'pothole', 'bbox', 0.1, 0.2, 0.4, 0.35, 'human' FROM tasks WHERE id = ?""",
            (signals[0].task_id, signals[0].task_id))
        self.repo.conn.commit()
        again = active_service.collect_signals(self.config, self.repo)
        self.assertEqual(len(again), 7, "已有人工标注的图不再排队")
        self.assertEqual(len(active_service.collect_signals(self.config, self.repo, include_labeled=True)), 8)
        # scan_limit 限制的是"扫描多少个待标注任务"，被过滤掉的已标图仍占扫描名额
        self.assertEqual(len(active_service.collect_signals(self.config, self.repo, scan_limit=3)), 2)

    def test_build_queue_writes_priorities_and_detail(self) -> None:
        tasks = [task["id"] for task in self.repo.list_tasks(status="pending", limit=100)]
        self.add_candidates(tasks[:4], score=0.30)
        model = self.register_model_with_evaluation(evaluation=fake_evaluation())
        report = active_service.build_queue(self.config, self.repo, limit=3, strategy="hybrid", actor="test")
        self.assertTrue(report.applied)
        self.assertEqual(report.selected, 3)
        run = self.repo.get_run(int(report.run_id))
        self.assertEqual(run["kind"], "active")
        self.assertEqual(run["status"], "succeeded")
        items = self.repo.active_queue(run_id=int(report.run_id))
        self.assertEqual(len(items), 3)
        self.assertLessEqual(items[0]["priority"], items[-1]["priority"])
        for item in items:
            self.assertIn("uncertainty", json.loads(item["components_json"]))
            self.assertEqual(item["strategy"], "hybrid")
        priorities = {item["task_id"]: item["priority"] for item in items}
        for task in self.repo.list_tasks(status="pending", limit=100):
            if task["id"] in priorities:
                self.assertEqual(task["priority"], priorities[task["id"]])
        artifact = Path(str(report.artifact))
        self.assertTrue(artifact.exists())
        self.assertEqual(len(json.loads(artifact.read_text(encoding="utf-8"))["items"]), 3)
        self.assertEqual(report.evaluation_source["model_id"], int(model["id"]))

    def test_build_queue_dry_run_writes_nothing(self) -> None:
        report = active_service.build_queue(self.config, self.repo, limit=5, apply=False)
        self.assertTrue(report.dry_run)
        self.assertFalse(report.applied)
        self.assertIsNone(report.run_id)
        self.assertEqual(self.repo.list_runs(kind="active"), [])
        self.assertTrue(all(task["priority"] == 100 for task in self.repo.list_tasks(limit=100)))

    def test_bad_package_does_not_break_queue(self) -> None:
        report = active_service.build_queue(self.config, self.repo, limit=2,
                                            package_dir=self.tmp_path / "not-a-package", apply=False)
        self.assertEqual(report.selected, 2, "margin 打分失败要退回置信度区间而不是整体失败")
        self.assertIn("error", report.margin_source or {})

    def test_queue_respects_limit_and_strategy(self) -> None:
        tasks = [task["id"] for task in self.repo.list_tasks(status="pending", limit=100)]
        self.add_candidates(tasks, score=0.30)
        random_a = active_service.build_queue(self.config, self.repo, limit=4, strategy="random",
                                              apply=False, seed=3)
        random_b = active_service.build_queue(self.config, self.repo, limit=4, strategy="random",
                                              apply=False, seed=3)
        self.assertEqual([item["task_id"] for item in random_a.items],
                         [item["task_id"] for item in random_b.items])
        uncertainty = active_service.build_queue(self.config, self.repo, limit=2,
                                                 strategy="uncertainty", apply=False)
        self.assertEqual(len(uncertainty.items), 2)
        self.assertTrue(all(item["components"]["uncertainty"] > 0 for item in uncertainty.items))

    def test_active_strategy_view_filters_to_queue(self) -> None:
        tasks = [task["id"] for task in self.repo.list_tasks(status="pending", limit=100)]
        self.add_candidates(tasks, score=0.3)
        report = active_service.build_queue(self.config, self.repo, limit=3, apply=True)
        self.assertEqual(report.selected, 3)
        self.assertEqual(len(report.items), 3, "limit 限制入选数量")
        queued = self.repo.list_tasks(status="pending", strategy="active", limit=50)
        self.assertEqual({task["id"] for task in queued}, {item["task_id"] for item in report.items})

    def test_degenerate_queue_is_reported_not_hidden(self) -> None:
        report = active_service.build_queue(self.config, self.repo, limit=2, apply=False)
        self.assertEqual(report.candidates, 8, "候选池是全部待标注任务（不被 limit 截断）")
        self.assertTrue(any("没有任何不确定性" in note for note in report.summary["notes"]))


class TestGateAlerts(ActiveFixture):
    def test_streak_counts_consecutive_failures(self) -> None:
        for version, passed in (("v1", False), ("v2", False), ("v3", True)):
            model = self.repo.upsert_model_version(name="s-road", version=version, status="candidate")
            self.repo.update_model_version(int(model["id"]), gate_json={"passed": passed, "reasons": ["r"]})
        self.assertEqual(active_service.gate_failure_streak(self.repo, "s-road")["streak"], 0,
                         "最新一次通过了就不再累计")
        model = self.repo.upsert_model_version(name="s-road", version="v4", status="candidate")
        self.repo.update_model_version(int(model["id"]), gate_json={"passed": False, "reasons": ["整体下降"]})
        self.assertEqual(active_service.gate_failure_streak(self.repo, "s-road")["streak"], 1)
        model = self.repo.upsert_model_version(name="s-road", version="v5", status="candidate")
        self.repo.update_model_version(int(model["id"]), gate_json={"passed": False, "reasons": ["类别塌陷"]})
        info = active_service.gate_failure_streak(self.repo, "s-road")
        self.assertEqual(info["streak"], 2)
        self.assertEqual(info["history"][0]["reasons"], ["类别塌陷"], "history 首项是最新一次")
        self.assertEqual(info["history"][1]["reasons"], ["整体下降"])

    def test_alerts_threshold(self) -> None:
        for version in ("v1", "v2"):
            model = self.repo.upsert_model_version(name="bad-road", version=version, status="candidate")
            self.repo.update_model_version(int(model["id"]), gate_json={"passed": False, "reasons": ["x"]})
        payload = active_service.gate_alerts(self.config, self.repo, threshold=2)
        self.assertEqual([alert["model"] for alert in payload["alerts"]], ["bad-road"])
        self.assertIn("回看标注规范", payload["alerts"][0]["action"])
        self.assertEqual(active_service.gate_alerts(self.config, self.repo, threshold=5)["alerts"], [])


class TestActiveApi(ActiveFixture):
    def setUp(self) -> None:
        super().setUp()
        from fastapi.testclient import TestClient

        from rdinspect.api.app import create_app

        self.client = TestClient(create_app(self.config))

    def tearDown(self) -> None:
        self.client.close()
        super().tearDown()

    def test_queue_endpoint_dry_run_and_fetch(self) -> None:
        response = self.client.post("/api/active/queue", json={"limit": 3, "dry_run": True})
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["dry_run"])
        self.assertEqual(response.json()["selected"], 3)
        self.assertEqual(self.client.get("/api/active/queue").status_code, 404,
                         "dry-run 不落库，因此还没有可查询的选样运行")
        tasks = [task["id"] for task in self.repo.list_tasks(status="pending", limit=100)]
        self.add_candidates(tasks, score=0.31)
        applied = self.client.post("/api/active/queue", json={"limit": 2, "strategy": "uncertainty"})
        self.assertEqual(applied.status_code, 200)
        self.assertEqual(applied.json()["updated_priorities"], 2)
        fetched = self.client.get("/api/active/queue")
        self.assertEqual(fetched.status_code, 200)
        self.assertEqual(len(fetched.json()["items"]), 2)
        self.assertEqual(fetched.json()["run"]["kind"], "active")

    def test_queue_rejects_unknown_strategy(self) -> None:
        response = self.client.post("/api/active/queue", json={"strategy": "nope"})
        self.assertEqual(response.status_code, 400)

    def test_alerts_endpoint(self) -> None:
        self.assertEqual(self.client.get("/api/active/alerts").json()["alerts"], [])
        for version in ("v1", "v2"):
            model = self.repo.upsert_model_version(name="api-road", version=version)
            self.repo.update_model_version(int(model["id"]), gate_json={"passed": False, "reasons": ["x"]})
        payload = self.client.get("/api/active/alerts?threshold=2").json()
        self.assertEqual([alert["model"] for alert in payload["alerts"]], ["api-road"])

    def test_tasks_active_strategy_endpoint(self) -> None:
        applied = self.client.post("/api/active/queue", json={"limit": 3})
        self.assertEqual(applied.json()["selected"], 3, "候选池是全部 8 个待标注任务，limit 只限制入选数量")
        response = self.client.get("/api/tasks?status=pending&strategy=active&limit=50")
        self.assertEqual(response.status_code, 200)
        # /api/tasks 是分页信封 {items, next_cursor}
        self.assertEqual(len(response.json()["items"]), 3)
        self.assertLess(len(response.json()["items"]),
                        len(self.client.get("/api/tasks?status=pending&limit=50").json()["items"]),
                        "active 视图应当只包含被选中的任务")

    def test_reports_endpoints(self) -> None:
        markdown = self.client.get("/api/reports/summary?format=markdown")
        self.assertEqual(markdown.status_code, 200)
        payload = self.client.get("/api/reports/summary").json()
        self.assertIn("trends", payload)
        self.assertIn("coverage", payload)
        gis = self.client.get("/api/reports/gis?format=geojson").json()
        self.assertEqual(gis["type"], "FeatureCollection")
        csv_response = self.client.get("/api/reports/gis?format=csv")
        self.assertEqual(csv_response.status_code, 200)
        self.assertTrue(csv_response.text.startswith("image_id,path,captured_at"))


if __name__ == "__main__":
    unittest.main()
