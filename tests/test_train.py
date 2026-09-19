"""M3 训练闭环测试：硬约束、幂等键、评估口径、门禁状态机、导出包与 ONNX 预处理。

原则：**不依赖真实 ML 依赖**——真实训练/导出由 ``scripts/e2e_m3.py`` 覆盖，
这里用假检测器与合成的 metrics 结构把「判定逻辑」钉死，跑得快且可在 CI 里跑。
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from rdinspect.config import ConfigError, GateConfig
from rdinspect.core import datasets as ds
from rdinspect.core.ingest import import_path
from rdinspect.errors import ConflictError, NotFoundError
from rdinspect.prelabel.detector import Detection, DetectorUnavailable
from rdinspect.train import evaluate as eval_mod
from rdinspect.train import export_onnx as export_mod
from rdinspect.train import gate as gate_mod
from rdinspect.train import matching
from rdinspect.train import service as train_service
from rdinspect.train.runner import (TrainOutcome, build_train_request,
                                    dataset_split_counts, enforce_hard_constraints, is_run_checkpoint,
                                    normalize_metrics, read_results_csv, register_trained_model)

from helpers import annotate_all, make_config, open_repo, synth_images

BOX = {"x1": 0.10, "y1": 0.20, "x2": 0.40, "y2": 0.35}


class FakeDetector:
    """按预置脚本返回检测结果：{image_id: [(class_name, (x1,y1,x2,y2) px, score)]}。"""

    def __init__(self, script: dict[int, list[tuple[str, tuple[float, float, float, float], float]]]) -> None:
        self.names = {0: "d00", 1: "d10", 2: "d40"}
        self.script = script
        self.calls: list[tuple[int, int]] = []
        self.image_ids: list[int] = []

    def predict(self, image: Image.Image) -> list[Detection]:
        image_id = self.image_ids.pop(0) if self.image_ids else -1
        self.calls.append((image_id, image.size[0]))
        return [Detection(class_index=0, class_name=name, score=score, bbox=bbox)
                for name, bbox, score in self.script.get(image_id, [])]


class TrainFixture(unittest.TestCase):
    """公共夹具：6 张合成图 → 标注 → 冻结数据集（含 train/val/test 划分）。"""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmp.name)
        self.config = make_config(self.tmp_path)
        self.conn, self.repo = open_repo(self.config)
        source = self.config.data_dir / "inbox" / "photos"
        synth_images(source, count=8, size=(160, 120))
        import_path(self.config, self.repo, source, kind="photo")
        annotate_all(self.repo, class_code="transverse_crack", bbox=BOX, approve=True)
        draft = ds.create_draft(self.config, self.repo, "ds-train",
                                {"review_status": "approved"},
                                {"train": 0.5, "val": 0.25, "test": 0.25, "seed": 7})
        self.dataset = ds.freeze_dataset(self.config, self.repo, dataset_id=int(draft["id"]))

    def tearDown(self) -> None:
        self.conn.close()
        self._tmp.cleanup()

    def image_ids(self, split: str) -> list[int]:
        return [int(item["id"]) for item in self.repo.dataset_items(int(self.dataset["id"]), split=split)]

    def _patch_runner(self, status: str = "succeeded") -> list[int]:
        """把 service.run_training 换成假实现，返回被调用的 run_id 列表。"""
        calls: list[int] = []

        def fake_run_training(config, repo, request, *, control=None, progress=None, log_sink=None,
                              run_id=None):
            calls.append(int(run_id))
            weights = config.weights_dir / f"fake-{run_id}.pt"
            weights.write_bytes(b"trained")
            outcome = TrainOutcome(run_id=int(run_id), status=status, dataset=request.dataset_name,
                                   dataset_id=int(request.dataset["id"]),
                                   manifest_hash=request.manifest_hash, weights_path=str(weights),
                                   last_weights=str(weights), work_dir=str(self.tmp_path / "run"),
                                   log_path=str(config.train_logs_dir / f"{run_id}.log"),
                                   metrics={"map50": 0.42, "epoch": 1}, epochs_done=1,
                                   params=dict(request.params))
            repo.finish_run(int(run_id), status=status,
                            metrics={"weights_path": str(weights), "final": outcome.metrics},
                            error=None if status == "succeeded" else "boom")
            return outcome

        original = train_service.run_training
        train_service.run_training = fake_run_training       # type: ignore[assignment]
        self.addCleanup(lambda: setattr(train_service, "run_training", original))
        return calls


# ─────────────────────────── 硬约束与参数解析 ───────────────────────────
class TestTrainRequest(TrainFixture):
    def test_flipud_must_be_zero(self) -> None:
        with self.assertRaises(ConfigError) as ctx:
            enforce_hard_constraints({"flipud": 0.5, "mosaic": 1.0}, allow_vertical_flip=False)
        self.assertIn("横向裂缝", str(ctx.exception))
        # 显式允许时才放行
        cleaned = enforce_hard_constraints({"flipud": 0.5}, allow_vertical_flip=True)
        self.assertEqual(cleaned["flipud"], 0.5)
        self.assertEqual(enforce_hard_constraints({"flipud": 0.0}, allow_vertical_flip=False)["flipud"], 0.0)

    def test_unknown_augment_keys_are_dropped(self) -> None:
        cleaned = enforce_hard_constraints({"mosaic": 1.0, "typo_key": 9.0}, allow_vertical_flip=False)
        self.assertNotIn("typo_key", cleaned)
        self.assertEqual(cleaned["mosaic"], 1.0)

    def test_build_request_requires_frozen_dataset(self) -> None:
        draft = ds.create_draft(self.config, self.repo, "ds-draft", {"review_status": "approved"}, {})
        with self.assertRaises(ConflictError) as ctx:
            build_train_request(self.config, self.repo, dataset="ds-draft")
        self.assertIn("frozen", str(ctx.exception))
        with self.assertRaises(NotFoundError):
            build_train_request(self.config, self.repo, dataset="不存在的数据集")
        self.assertEqual(int(draft["id"]) > 0, True)

    def test_run_key_is_stable_and_parameter_sensitive(self) -> None:
        first = build_train_request(self.config, self.repo, dataset="ds-train")
        again = build_train_request(self.config, self.repo, dataset="ds-train")
        self.assertEqual(first.run_key, again.run_key, "同参数必须得到同一 run_key（幂等）")
        for changed in ({"epochs": first.params["epochs"] + 1}, {"imgsz": 320}, {"batch": 2}):
            other = build_train_request(self.config, self.repo, dataset="ds-train", **changed)
            self.assertNotEqual(first.run_key, other.run_key)
        # 数据集内容变化（重新冻结产生新 manifest）也会换 key —— 用替换 manifest 近似验证
        self.assertIn(first.manifest_hash, json.dumps(first.as_dict()))

    def test_resolved_params_record_device_and_amp(self) -> None:
        request = build_train_request(self.config, self.repo, dataset="ds-train", device="cpu")
        self.assertEqual(request.params["device"], "cpu")
        self.assertFalse(request.params["amp"], "CPU 上 AMP 必须关闭")
        self.assertIn("flipud", request.augment)

    def test_is_run_checkpoint(self) -> None:
        self.assertTrue(is_run_checkpoint("/runs/run-1/weights/last.pt"))
        self.assertFalse(is_run_checkpoint("/runs/run-1/weights/best.pt"))
        self.assertFalse(is_run_checkpoint(None))

    def test_split_counts_detects_empty_val(self) -> None:
        counts = dataset_split_counts(Path(str(self.dataset["root_path"])) / "data.yaml")
        self.assertEqual(sum(counts.values()), 8)
        self.assertTrue(counts["train"] > 0 and counts["val"] > 0)


# ─────────────────────────── 指标解析与评估 ───────────────────────────
class TestMetricsParsing(unittest.TestCase):
    def test_read_results_csv_and_normalize(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "results.csv"
            path.write_text(
                "epoch,time,train/box_loss,metrics/precision(B),metrics/recall(B),"
                "metrics/mAP50(B),metrics/mAP50-95(B)\n"
                "1,12.5,1.234,0.51,0.44,0.37,0.21\n"
                "2,25.0,1.100,0.60,0.55,0.48,0.30\n", encoding="utf-8")
            rows = read_results_csv(path)
            self.assertEqual(len(rows), 2)
            metrics = normalize_metrics(rows[-1])
            self.assertEqual(metrics["epoch"], 2)
            self.assertEqual(metrics["map50"], 0.48)
            self.assertEqual(metrics["map50_95"], 0.30)
            self.assertEqual(metrics["precision"], 0.60)
            self.assertEqual(metrics["train_box_loss"], 1.100)
            self.assertEqual(read_results_csv(Path(tmp) / "missing.csv"), [])


class TestEvaluatePredictions(TrainFixture):
    def _evaluate(self, script: dict, split: str = "val", **kwargs) -> dict:
        detector = FakeDetector(script)
        detector.image_ids = self.image_ids(split)
        return eval_mod.evaluate_predictions(
            self.config, self.repo, dataset=self.dataset, split=split, detector=detector,
            classes=self.repo.list_classes(), conf=0.25, iou=0.5,
            work_dir=self.tmp_path / "eval", export_failure_limit=kwargs.pop("failure_limit", 5),
            **kwargs)

    def test_perfect_predictions_hit_full_recall(self) -> None:
        ids = self.image_ids("val")
        script = {image_id: [("d10", (16.0, 24.0, 64.0, 42.0), 0.9)] for image_id in ids}
        payload = self._evaluate(script)
        self.assertEqual(payload["internal"]["images"], len(ids))
        pr = payload["internal"]["precision_recall"]["overall"]
        self.assertEqual(pr["fp"], 0)
        self.assertEqual(pr["fn"], 0)
        self.assertAlmostEqual(pr["precision"], 1.0)
        self.assertAlmostEqual(pr["recall"], 1.0)
        ap = payload["internal"]["map50"]
        self.assertAlmostEqual(ap["per_class"]["transverse_crack"]["ap"], 1.0, places=6)
        self.assertAlmostEqual(ap["map50_classes_with_gt"], 1.0, places=6)
        self.assertEqual(ap["classes_with_gt"], ["transverse_crack"])
        # 内部口径仍按类别表全量平均（缺真值的 4 类记 0），这是"保守口径"，官方 val 才是门禁主判据
        self.assertAlmostEqual(ap["map50"], 0.2, places=6)
        self.assertEqual(payload["failures"]["total"], 0)

    def test_misses_are_exported_as_failures(self) -> None:
        ids = self.image_ids("val")
        script = {ids[0]: [], ids[1]: [("d10", (16.0, 24.0, 64.0, 42.0), 0.9)]}
        payload = self._evaluate(script, failure_limit=5)
        self.assertEqual(payload["internal"]["precision_recall"]["overall"]["fn"], len(ids) - 1)
        self.assertGreaterEqual(payload["failures"]["exported"], 1)
        overlay = Path(payload["failures"]["worst"][0]["overlay"])
        self.assertTrue((self.tmp_path / "eval" / overlay).exists(), "失败样例必须落盘叠加图")

    def test_class_confusion_and_unmapped_classes(self) -> None:
        ids = self.image_ids("val")
        script = {image_id: [("d00", (16.0, 24.0, 64.0, 42.0), 0.8)] for image_id in ids}
        payload = self._evaluate({**script, ids[0]: script[ids[0]] + [("unknown_thing", (1.0, 1.0, 9.0, 9.0), 0.7)]})
        cm = payload["internal"]["confusion_matrix"]
        labels = cm["labels"]
        gt_index = labels.index("transverse_crack")
        pred_index = labels.index("longitudinal_crack")
        self.assertGreater(cm["matrix"][gt_index][pred_index], 0, "纵/横混淆必须落在对应单元格")
        self.assertEqual(payload["internal"]["unmapped_classes"], {"unknown_thing": 1})

    def test_size_bucket_recall_uses_gt_pixel_area(self) -> None:
        ids = self.image_ids("val")
        # d10 → transverse_crack（与真值同类，class-aware 匹配才能命中）
        script = {image_id: [("d10", (16.0, 24.0, 64.0, 42.0), 0.9)] for image_id in ids}
        payload = self._evaluate(script)
        buckets = payload["internal"]["size_bucket_recall"]["buckets"]
        self.assertEqual(buckets["small"]["gt"], len(ids))
        self.assertAlmostEqual(buckets["small"]["recall"], 1.0)
        self.assertIsNone(buckets["large"]["recall"], "无真值的桶必须是 None 而不是 0")

    def test_sahi_ablation_reports_both_modes(self) -> None:
        ids = self.image_ids("val")
        script = {image_id: [("d10", (16.0, 24.0, 64.0, 42.0), 0.9)] for image_id in ids}
        payload = self._evaluate(script, sahi_ablation=True)
        ablation = payload["sahi_ablation"]
        self.assertIsNotNone(ablation)
        self.assertIn("recall_delta", ablation)
        self.assertGreaterEqual(ablation["map50_off"], 0.0)
        self.assertEqual(payload["internal"]["images"], len(ids))

    def test_empty_split_raises(self) -> None:
        detector = FakeDetector({})
        with self.assertRaises(NotFoundError):
            eval_mod.evaluate_predictions(self.config, self.repo, dataset=self.dataset, split="nope",
                                          detector=detector, classes=self.repo.list_classes(),
                                          conf=0.25, iou=0.5)

    def test_report_renders_markdown(self) -> None:
        ids = self.image_ids("val")
        script = {image_id: [("d10", (16.0, 24.0, 64.0, 42.0), 0.9)] for image_id in ids}
        payload = self._evaluate(script)
        text = eval_mod.render_report(payload)
        self.assertIn("评估报告", text)
        self.assertIn("大小桶召回", text)
        self.assertIn("transverse_crack", text)


# ─────────────────────────── 门禁判定 ───────────────────────────
def _metrics(map50: float, per_class: dict[str, float] | None = None,
             recall: dict[str, float] | None = None) -> dict:
    return {
        "ultralytics": {"map50": map50, "map50_95": map50 / 2,
                        "per_class": {code: {"Box-P": 0.5, "Box-R": 0.5, "mAP50": value,
                                             "mAP50-95": value / 2, "Instances": 10}
                                      for code, value in (per_class or {}).items()}},
        "internal": {"precision_recall": {"per_class": {code: {"recall": value, "tp": 1, "fp": 0, "fn": 0}
                                                        for code, value in (recall or {}).items()}}},
        "weights": "/tmp/w.pt", "split": "val", "dataset": "ds-train",
    }


class TestGate(unittest.TestCase):
    def _compare(self, candidate: dict, baseline: dict | None, **kwargs):
        defaults = {"map50_tolerance": 0.005, "per_class_tolerance": 0.02, "min_map50": 0.0}
        defaults.update(kwargs)
        return gate_mod.compare_metrics(candidate, baseline, **defaults)

    def test_improvement_passes(self) -> None:
        decision = self._compare(_metrics(0.50, {"transverse_crack": 0.5}),
                                 _metrics(0.40, {"transverse_crack": 0.4}))
        self.assertTrue(decision.passed)
        self.assertAlmostEqual(decision.delta["map50"], 0.10, places=6)

    def test_overall_regression_is_rejected(self) -> None:
        decision = self._compare(_metrics(0.30), _metrics(0.40))
        self.assertFalse(decision.passed)
        self.assertTrue(any("整体 mAP50 下降" in reason for reason in decision.reasons))

    def test_tolerance_allows_small_drop(self) -> None:
        self.assertTrue(self._compare(_metrics(0.398), _metrics(0.40)).passed)
        self.assertFalse(self._compare(_metrics(0.390), _metrics(0.40)).passed)

    def test_class_collapse_is_rejected_even_when_overall_improves(self) -> None:
        candidate = _metrics(0.45, {"transverse_crack": 0.60, "pothole": 0.0})
        baseline = _metrics(0.40, {"transverse_crack": 0.30, "pothole": 0.55})
        decision = self._compare(candidate, baseline)
        self.assertFalse(decision.passed)
        self.assertTrue(any("类别塌陷" in reason for reason in decision.reasons), decision.reasons)

    def test_missing_class_in_candidate_is_rejected(self) -> None:
        decision = self._compare(_metrics(0.45, {"transverse_crack": 0.5}),
                                 _metrics(0.40, {"transverse_crack": 0.4, "pothole": 0.4}))
        self.assertFalse(decision.passed)
        self.assertTrue(any("pothole" in reason and "缺失" in reason for reason in decision.reasons))

    def test_absolute_floor_applies_without_baseline(self) -> None:
        decision = self._compare(_metrics(0.01), None, min_map50=0.05)
        self.assertFalse(decision.passed)
        self.assertTrue(any("绝对下限" in reason for reason in decision.reasons))
        self.assertTrue(any("无基线" in note for note in decision.notes))
        self.assertTrue(self._compare(_metrics(0.10), None, min_map50=0.05).passed)

    def test_invalid_metrics_are_rejected(self) -> None:
        decision = self._compare({"ultralytics": {"map50": float("nan")}}, None)
        self.assertFalse(decision.passed)
        self.assertTrue(any("缺失或非法" in reason for reason in decision.reasons))

    def test_min_class_recall_floor(self) -> None:
        candidate = _metrics(0.5, {"transverse_crack": 0.5}, recall={"transverse_crack": 0.25})
        decision = self._compare(candidate, None, min_class_recall=0.3)
        self.assertFalse(decision.passed)
        self.assertTrue(any("召回" in reason for reason in decision.reasons))


# ─────────────────────────── 状态机与导出包 ───────────────────────────
class TestModelLifecycle(TrainFixture):
    def _register(self, *, name: str = "yolo11n-road", version: str = "v1",
                  status: str = "candidate") -> dict:
        weights = self.config.weights_dir / f"{name}-{version}.pt"
        weights.write_bytes(b"fake-weights")
        return self.repo.upsert_model_version(name=name, version=version, status=status,
                                              weights_path=str(weights), dataset_id=int(self.dataset["id"]),
                                              labels_json={"names": [cls["code"] for cls in self.repo.list_classes()]})

    def test_promote_requires_validated_status(self) -> None:
        model = self._register()
        with self.assertRaises(ConflictError) as ctx:
            gate_mod.promote_model(self.config, self.repo, int(model["id"]))
        self.assertIn("只有 validated", str(ctx.exception))

    def test_promote_requires_passed_gate_record(self) -> None:
        model = self._register(status="validated")
        with self.assertRaises(ConflictError) as ctx:
            gate_mod.promote_model(self.config, self.repo, int(model["id"]))
        self.assertIn("门禁", str(ctx.exception))

    def test_promote_archives_previous_production(self) -> None:
        old = self._register(name="old-road", version="v0", status="production")
        new = self._register(name="new-road", version="v1", status="validated")
        self.repo.update_model_version(int(new["id"]), gate_json={"passed": True, "reasons": []})
        result = gate_mod.promote_model(self.config, self.repo, int(new["id"]))
        self.assertEqual(result["status"], "production")
        self.assertEqual(result["archived"], [int(old["id"])])
        self.assertEqual(self.repo.get_model_version(int(old["id"]))["status"], "archived")
        self.assertEqual(int(self.repo.production_model()["id"]), int(new["id"]))
        # 幂等：再提升一次不报错、不重复归档
        again = gate_mod.promote_model(self.config, self.repo, int(new["id"]))
        self.assertFalse(again["changed"])

    def test_gate_model_writes_decision_and_rejects_regression(self) -> None:
        baseline = self._register(name="base-road", version="v1", status="production")
        self.repo.update_model_version(
            int(baseline["id"]), metrics_json={"evaluation": _metrics(0.50, {"transverse_crack": 0.5})})
        candidate = self._register(name="cand-road", version="v2")
        with self.assertRaises(ConflictError) as ctx:
            gate_mod.gate_model(self.config, self.repo, int(candidate["id"]),
                                evaluation=_metrics(0.20, {"transverse_crack": 0.1}))
        self.assertIn("门禁未通过", str(ctx.exception))
        row = self.repo.get_model_version(int(candidate["id"]))
        self.assertEqual(row["status"], "candidate", "门禁未通过不得改状态")
        self.assertFalse(gate_mod.model_gate(row)["passed"])
        # 通过后状态变 validated 且门禁记录在案
        outcome = gate_mod.gate_model(self.config, self.repo, int(candidate["id"]),
                                      evaluation=_metrics(0.60, {"transverse_crack": 0.7}))
        self.assertEqual(outcome["status"], "validated")
        self.assertTrue(gate_mod.model_gate(self.repo.get_model_version(int(candidate["id"])))["passed"])

    def test_gate_without_evaluation_is_rejected(self) -> None:
        model = self._register()
        with self.assertRaises(ConflictError) as ctx:
            gate_mod.gate_model(self.config, self.repo, int(model["id"]))
        self.assertIn("还没有评估结果", str(ctx.exception))

    def test_export_requires_validated_model(self) -> None:
        model = self._register()
        with self.assertRaises(ConflictError) as ctx:
            export_mod.export_onnx(self.config, self.repo, int(model["id"]))
        self.assertIn("只有 validated", str(ctx.exception))

    def test_export_package_layout_and_class_order_guard(self) -> None:
        model = self._register(status="validated")
        onnx = self.tmp_path / "model.onnx"
        onnx.write_bytes(b"not-a-real-onnx")
        package = export_mod.build_export_package(
            self.config, self.repo, model, onnx, opset=17, dynamic_batch=True, half=False,
            simplify=True, imgsz=320, dataset=self.dataset, export_run_id=1, tolerance=1e-3)
        root = Path(package["dir"])
        for name in ("model.onnx", "labels.txt", "preprocess.json", "manifest.json", "README.md"):
            self.assertTrue((root / name).exists(), f"导出包缺少 {name}")
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        preprocess = json.loads((root / "preprocess.json").read_text(encoding="utf-8"))
        labels = (root / "labels.txt").read_text(encoding="utf-8").split()
        self.assertEqual(manifest["labels"], labels)
        self.assertEqual(manifest["dataset"]["manifest_hash"], self.dataset["manifest_hash"])
        self.assertEqual(preprocess["input_size"], [320, 320])
        self.assertEqual(preprocess["resize"]["mode"], "letterbox")
        self.assertEqual(preprocess["resize"]["pad_value"], 114)
        # 类别顺序漂移必须被拦住（新增类别 → 重新冻结再训练）
        self.repo.add_class("water_puddle", "积水", "Water Puddle", order_index=10)
        with self.assertRaises(ConflictError) as ctx:
            export_mod.build_export_package(
                self.config, self.repo, model, onnx, opset=17, dynamic_batch=True, half=False,
                simplify=True, imgsz=320, dataset=self.dataset, export_run_id=1, tolerance=1e-3)
        self.assertIn("类别顺序", str(ctx.exception))


class TestOnnxRuntimeGeometry(unittest.TestCase):
    """ONNX 预处理/后处理几何（不依赖 onnxruntime 与 torch）。"""

    def test_letterbox_geometry_matches_ultralytics_convention(self) -> None:
        from rdinspect.edge.onnx_runtime import letterbox

        image = Image.new("RGB", (1280, 720), (10, 20, 30))
        tensor, meta = letterbox(image, 640)
        self.assertEqual(tensor.shape, (1, 3, 640, 640))
        self.assertAlmostEqual(meta.ratio, 0.5, places=6)
        self.assertEqual(meta.resized, (640, 360))
        self.assertEqual((int(meta.pad_x), int(meta.pad_y)), (0, 140))
        self.assertAlmostEqual(float(tensor[0, :, 0, 0].mean()), 114 / 255, places=6, msg="填充值必须是 114")
        interior = tensor[0, :, 320, 320]          # 图像区域内（top=140，缩放后高 360）
        self.assertAlmostEqual(float(interior[0]), 10 / 255, places=6)
        self.assertAlmostEqual(float(interior[1]), 20 / 255, places=6)
        self.assertAlmostEqual(float(interior[2]), 30 / 255, places=6)

    def test_letterbox_undo_is_inverse(self) -> None:
        from rdinspect.edge.onnx_runtime import letterbox

        image = Image.new("RGB", (800, 600), (5, 5, 5))
        _, meta = letterbox(image, 320)
        x, y = meta.undo(meta.pad_x + 100 * meta.ratio, meta.pad_y + 50 * meta.ratio)
        self.assertAlmostEqual(x, 100, places=6)
        self.assertAlmostEqual(y, 50, places=6)

    def test_decode_predictions_maps_to_original_coordinates(self) -> None:
        from rdinspect.edge.onnx_runtime import decode_predictions, letterbox

        image = Image.new("RGB", (1280, 720))
        _, meta = letterbox(image, 640)
        # (1, 4+nc, N)：中心点放在 letterbox 后的 (320, 320)，框 100×50
        output = np.zeros((1, 4 + 2, 2), dtype=np.float32)
        output[0, :4, 0] = [320.0, 320.0, 100.0, 50.0]
        output[0, 4:, 0] = [0.9, 0.1]
        output[0, 4:, 1] = [0.2, 0.2]      # 低于阈值
        decoded = decode_predictions(output, meta, conf=0.25, class_names=["transverse_crack", "pothole"])
        self.assertEqual(len(decoded), 1)
        box = decoded[0]["bbox"]
        self.assertEqual(decoded[0]["class_name"], "transverse_crack")
        self.assertAlmostEqual(box[0], 540.0, places=3)
        self.assertAlmostEqual(box[1], 310.0, places=3)
        self.assertAlmostEqual(box[2], 740.0, places=3)
        self.assertAlmostEqual(box[3], 410.0, places=3)

    def test_load_export_package_requires_manifest(self) -> None:
        from rdinspect.edge.onnx_runtime import load_export_package

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                load_export_package(tmp)


class TestTrainingService(TrainFixture):
    """后台/同步训练编排（用假 trainer 替换真实 ultralytics）。"""

    def test_sync_training_registers_candidate_model(self) -> None:
        calls = self._patch_runner()
        result = train_service.start_training(self.config, dataset="ds-train", name="smoke-road",
                                              version="v1", device="cpu", background=False)
        self.assertEqual(len(calls), 1)
        self.assertEqual(result["run"]["status"], "succeeded")
        self.assertIsNotNone(result["model"])
        self.assertEqual(result["model"]["status"], "candidate")
        self.assertEqual(result["model"]["name"], "smoke-road")
        self.assertEqual(result["splits"]["train"], 4)

    def test_idempotent_second_submit_returns_cached(self) -> None:
        self._patch_runner()
        first = train_service.start_training(self.config, dataset="ds-train", device="cpu", background=False)
        second = train_service.start_training(self.config, dataset="ds-train", device="cpu", background=False)
        self.assertTrue(second["cached"])
        self.assertEqual(int(second["run"]["id"]), int(first["run"]["id"]))
        self.assertEqual(len(self.repo.list_runs(kind="train")), 1)

    def test_failed_training_does_not_register_model(self) -> None:
        self._patch_runner(status="failed")
        result = train_service.start_training(self.config, dataset="ds-train", device="cpu", background=False)
        self.assertEqual(result["run"]["status"], "failed")
        self.assertIsNone(result["model"])
        self.assertEqual(len(self.repo.list_model_versions()), 0)

    def test_background_training_is_reported_as_run(self) -> None:
        self._patch_runner()
        result = train_service.start_training(self.config, dataset="ds-train", device="cpu", background=True)
        self.assertTrue(result["started"])
        self.assertEqual(result["run"]["status"], "running")
        self.assertIsNotNone(self.repo.get_run(int(result["run_id"])))
        # 必须等后台线程收尾：否则 cleanup 还原补丁后，线程会去调真实训练（会尝试联网下载权重）
        thread = train_service.JOBS.thread(int(result["run_id"]))
        self.assertIsNotNone(thread)
        thread.join(timeout=15)                                    # type: ignore[union-attr]
        self.assertFalse(train_service.JOBS.is_active(int(result["run_id"])))
        self.assertEqual(self.repo.get_run(int(result["run_id"]))["status"], "succeeded")
        self.assertEqual(len(self.repo.list_model_versions()), 1, "后台训练成功后应登记模型")

    def test_cancel_unknown_run_is_conflict(self) -> None:
        run = self.repo.create_run("train", run_key="x", config_json={})
        self.repo.finish_run(int(run["id"]), status="succeeded", metrics={})
        with self.assertRaises(ConflictError):
            train_service.cancel_training(self.repo, int(run["id"]))
        with self.assertRaises(NotFoundError):
            train_service.cancel_training(self.repo, 999999)

    def test_run_detail_exposes_progress_and_log_tail(self) -> None:
        run = self.repo.create_run("train", run_key="detail", config_json={})
        log_path = self.config.train_logs_dir / "detail.log"
        log_path.write_text("line-1\nline-2\nline-3\n", encoding="utf-8")
        self.repo.update_run(int(run["id"]), metrics={"progress": {"epoch": 2, "map50": 0.3}},
                             log_path=str(log_path))
        detail = train_service.run_detail(self.config, self.repo, int(run["id"]), log_lines=2)
        self.assertEqual(detail["progress"]["epoch"], 2)
        self.assertEqual(detail["log_tail"], ["line-2", "line-3"])


class TestRegisterTrainedModel(TrainFixture):
    def test_register_requires_successful_outcome(self) -> None:
        outcome = TrainOutcome(run_id=1, status="failed", dataset="ds-train", dataset_id=1,
                               manifest_hash="x", weights_path=None, last_weights=None,
                               work_dir=None, log_path=None)
        with self.assertRaises(ConflictError):
            register_trained_model(self.config, self.repo, outcome, name="nope")

    def test_register_is_idempotent_and_hashes_weights(self) -> None:
        weights = self.config.weights_dir / "best.pt"
        weights.write_bytes(b"weights-content")
        run = self.repo.create_run("train", run_key="reg", config_json={})
        outcome = TrainOutcome(run_id=int(run["id"]), status="succeeded", dataset="ds-train",
                               dataset_id=int(self.dataset["id"]), manifest_hash="mh",
                               weights_path=str(weights), last_weights=str(weights),
                               work_dir=str(self.tmp_path), log_path=None,
                               metrics={"map50": 0.5}, epochs_done=3)
        first = register_trained_model(self.config, self.repo, outcome, name="r-road", version="v9")
        second = register_trained_model(self.config, self.repo, outcome, name="r-road", version="v9")
        self.assertEqual(int(first["id"]), int(second["id"]))
        self.assertEqual(len(self.repo.list_model_versions()), 1)
        refreshed = self.repo.get_model_version(int(first["id"]))
        self.assertTrue(refreshed["sha256"])
        summary = gate_mod.model_summary(refreshed)
        self.assertEqual(summary["train_metrics"]["map50"], 0.5)
        self.assertIsNone(summary["map50"], "未评估的模型不应假装有 mAP")

    def test_matching_module_stays_ml_free(self) -> None:
        """matching 是纯函数模块：不得因为 train 包而被拖入 ML 依赖。"""
        source = Path(matching.__file__).read_text(encoding="utf-8")
        for forbidden in ("import torch", "import numpy", "import cv2", "ultralytics"):
            self.assertNotIn(forbidden, source)


class TestGateConfigDefaults(unittest.TestCase):
    def test_defaults_are_conservative(self) -> None:
        cfg = GateConfig()
        self.assertEqual(cfg.baseline, "production")
        self.assertGreater(cfg.per_class_tolerance, 0)
        self.assertLessEqual(cfg.map50_tolerance, 0.05)


if __name__ == "__main__":
    unittest.main()


class TestTrainApiEndpoints(TrainFixture):
    """M3 HTTP 契约：训练提交、运行详情、模型注册表、门禁与导出的错误语义。"""

    def setUp(self) -> None:
        super().setUp()
        from fastapi.testclient import TestClient

        from rdinspect.api.app import create_app

        self.client = TestClient(create_app(self.config))

    def tearDown(self) -> None:
        self.client.close()
        super().tearDown()

    def test_train_run_requires_dataset(self) -> None:
        response = self.client.post("/api/train/runs", json={"dataset": "没有这个数据集"})
        self.assertEqual(response.status_code, 404)

    def test_train_run_rejects_draft_dataset(self) -> None:
        ds.create_draft(self.config, self.repo, "ds-draft2", {"review_status": "approved"}, {})
        response = self.client.post("/api/train/runs", json={"dataset": "ds-draft2"})
        self.assertEqual(response.status_code, 409)
        self.assertIn("frozen", response.json()["message"])

    def test_train_run_reports_missing_ml_dependency_as_501(self) -> None:
        original = train_service.build_train_request

        def explode(*args, **kwargs):
            raise DetectorUnavailable("ML 依赖缺失（测试注入）")

        train_service.build_train_request = explode          # type: ignore[assignment]
        self.addCleanup(lambda: setattr(train_service, "build_train_request", original))
        # start_training 内部引用的是模块级名字，替换后即可触发 501 分支
        response = self.client.post("/api/train/runs", json={"dataset": "ds-train"})
        self.assertEqual(response.status_code, 501)

    def test_run_detail_and_cancel_semantics(self) -> None:
        run = self.repo.create_run("train", run_key="api-detail", config_json={})
        self.assertEqual(self.client.get(f"/api/train/runs/{run['id']}").status_code, 200)
        self.assertEqual(self.client.get("/api/train/runs/999999").status_code, 404)
        # 已完成（非 running）→ 409；不存在的 run → 404
        self.repo.finish_run(int(run["id"]), status="succeeded", metrics={})
        self.assertEqual(self.client.post(f"/api/train/runs/{run['id']}/cancel").status_code, 409)
        self.assertEqual(self.client.post("/api/train/runs/999999/cancel").status_code, 404)

    def test_model_registry_and_model_detail(self) -> None:
        registry = self.client.get("/api/models/registry")
        self.assertEqual(registry.status_code, 200)
        self.assertEqual(registry.json(), [])
        weights = self.config.weights_dir / "api.pt"
        weights.write_bytes(b"fake")
        model = self.repo.upsert_model_version(name="api-road", version="v1", status="candidate",
                                              weights_path=str(weights),
                                              dataset_id=int(self.dataset["id"]))
        detail = self.client.get(f"/api/models/{model['id']}")
        self.assertEqual(detail.status_code, 200)
        self.assertEqual(detail.json()["name"], "api-road")
        self.assertEqual(self.client.get("/api/models/424242").status_code, 404)
        self.assertEqual(len(self.client.get("/api/models/registry").json()), 1)

    def test_promote_unvalidated_model_is_conflict(self) -> None:
        weights = self.config.weights_dir / "cand.pt"
        weights.write_bytes(b"fake")
        model = self.repo.upsert_model_version(name="cand-road", version="v1", status="candidate",
                                              weights_path=str(weights),
                                              dataset_id=int(self.dataset["id"]))
        response = self.client.post(f"/api/models/{model['id']}/promote")
        self.assertEqual(response.status_code, 409)
        self.assertEqual(self.client.post("/api/models/424242/promote").status_code, 404)

    def test_export_unvalidated_model_is_conflict(self) -> None:
        weights = self.config.weights_dir / "cand2.pt"
        weights.write_bytes(b"fake")
        model = self.repo.upsert_model_version(name="cand2-road", version="v1", status="candidate",
                                              weights_path=str(weights),
                                              dataset_id=int(self.dataset["id"]))
        response = self.client.post(f"/api/models/{model['id']}/export", json={})
        self.assertEqual(response.status_code, 409)


class TestCompatLayer(unittest.TestCase):
    """受限容器兼容层：/dev/shm 不可用时把 ultralytics 线程池换成串行实现。"""

    def test_serial_pool_matches_pool_interface(self) -> None:
        from rdinspect.train.compat import _SerialPool

        pool = _SerialPool(4)
        with pool as active:
            self.assertEqual(list(active.imap(lambda x: x * 2, [1, 2, 3])), [2, 4, 6])
            self.assertEqual(active.map(str, [1, 2]), ["1", "2"])
            self.assertEqual(active.starmap(lambda a, b: a + b, [(1, 2), (3, 4)]), [3, 7])
            self.assertEqual(list(active.imap_unordered(lambda x: x + 1, [0])), [1])
        pool.close()
        pool.join()
        pool.terminate()

    def test_semaphore_probe_returns_bool(self) -> None:
        from rdinspect.train import compat

        self.assertIsInstance(compat.semaphores_available(), bool)

    def test_force_patch_is_idempotent_and_reports_reason(self) -> None:
        from rdinspect.train import compat

        compat.reset_state()
        first = compat.apply_compat_patches(force=True)
        second = compat.apply_compat_patches(force=True)
        self.assertTrue(second["checked"])
        self.assertEqual(first["patched"], second["patched"])
        # 未导入 ultralytics 时补丁数可能为 0（没有可替换的模块），但不允许抛异常
        self.assertIn("reason", first)
        compat.reset_state()

    def test_forced_patch_replaces_loaded_threadpool(self) -> None:
        import types

        from rdinspect.train import compat

        compat.reset_state()
        module_name = "ultralytics.data.dataset"
        original = sys.modules.get(module_name)
        stub = original or types.ModuleType(module_name)
        stub.ThreadPool = object()                                  # type: ignore[attr-defined]
        sys.modules[module_name] = stub
        try:
            payload = compat.apply_compat_patches(force=True)
            self.assertTrue(payload["patched"])
            self.assertIs(sys.modules[module_name].ThreadPool, compat._SerialPool)
        finally:
            if original is None:
                sys.modules.pop(module_name, None)
            compat.reset_state()


class TestValMetricExtraction(unittest.TestCase):
    """官方 val 指标解析：必须能吃 numpy 数组（历史上 `array or []` 在这里炸过）。"""

    class _Box:
        ap50 = np.array([0.995, 0.665])
        maps = np.array([0.71, 0.42])
        p = np.array([1.0, 0.98])
        r = np.array([0.92, 0.70])
        ap_class_index = np.array([0, 1])
        map50 = 0.9125
        map = 0.7249
        mp = 1.0
        mr = 0.9118

    class _Results:
        names = {0: "longitudinal_crack", 1: "transverse_crack"}
        nt_per_class = np.array([8, 3])

        def __init__(self) -> None:
            self.box = TestValMetricExtraction._Box()

    def test_extracts_per_class_from_numpy_arrays(self) -> None:
        classes = [{"code": "longitudinal_crack", "name_en": "Longitudinal Crack", "order_index": 0},
                   {"code": "transverse_crack", "name_en": "Transverse Crack", "order_index": 1},
                   {"code": "pothole", "name_en": "Pothole", "order_index": 2}]
        payload = eval_mod.extract_val_metrics(self._Results(), classes)
        self.assertAlmostEqual(payload["map50"], 0.9125, places=6)
        self.assertAlmostEqual(payload["map50_95"], 0.7249, places=6)
        self.assertAlmostEqual(payload["per_class"]["transverse_crack"]["map50"], 0.665, places=6)
        self.assertAlmostEqual(payload["per_class"]["longitudinal_crack"]["map50_95"], 0.71, places=6)
        self.assertEqual(payload["per_class"]["transverse_crack"]["instances"], 3)
        self.assertNotIn("pothole", payload["per_class"], "没有实例的类不出现（不是 0）")

    def test_missing_box_returns_empty(self) -> None:
        class _Empty:
            names = {}
            box = None

        payload = eval_mod.extract_val_metrics(_Empty(), [])
        self.assertEqual(payload["per_class"], {})

    def test_primary_map50_prefers_official_then_internal(self) -> None:
        self.assertAlmostEqual(eval_mod.primary_map50({"ultralytics": {"map50": 0.9}}), 0.9, places=6)
        self.assertAlmostEqual(
            eval_mod.primary_map50({"ultralytics": {}, "internal": {"map50": {"map50_classes_with_gt": 0.7}}}),
            0.7, places=6, msg="官方口径缺失时应退回内部口径")
        self.assertIsNone(eval_mod.primary_map50({"ultralytics": {"map50": float("nan")}}))


class TestDependencyStatusSemantics(TrainFixture):
    """依赖缺失必须是 501（环境不可用），不能混成 409（状态不允许）或先 202 再失败。"""

    def test_missing_ml_dependency_is_not_conflict(self) -> None:
        from rdinspect.prelabel.detector import DetectorUnavailable
        from rdinspect.train import export_onnx as export_mod2

        weights = self.config.weights_dir / "validated.pt"
        weights.write_bytes(b"fake")
        model = self.repo.upsert_model_version(name="v-road", version="v1", status="validated",
                                              weights_path=str(weights),
                                              dataset_id=int(self.dataset["id"]))
        original = export_mod2.ml_available
        export_mod2.ml_available = lambda: False          # type: ignore[assignment]
        self.addCleanup(lambda: setattr(export_mod2, "ml_available", original))
        with self.assertRaises(DetectorUnavailable):
            export_mod2.export_onnx(self.config, self.repo, int(model["id"]))

    def test_missing_ml_dependency_fails_before_accepting_training(self) -> None:
        from rdinspect.prelabel.detector import DetectorUnavailable
        from rdinspect.train import service as svc

        original = svc.ml_available
        svc.ml_available = lambda: False                  # type: ignore[assignment]
        self.addCleanup(lambda: setattr(svc, "ml_available", original))
        with self.assertRaises(DetectorUnavailable):
            svc.start_training(self.config, dataset="ds-train", device="cpu", background=True)
        self.assertEqual(self.repo.list_runs(kind="train"), [], "依赖缺失时不得留下 run 记录")


class TestStaleRunRecovery(TrainFixture):
    """进程被杀/重启后遗留的 running 必须能自愈，否则同一配置永远提交不上去。"""

    def test_reconcile_marks_orphan_running_as_failed(self) -> None:
        run = self.repo.create_run("train", run_key="orphan", config_json={})
        self.assertEqual(len(train_service.stale_runs(self.repo)), 1)
        fixed = train_service.reconcile_stale_runs(self.repo)
        self.assertEqual(fixed, [int(run["id"])])
        row = self.repo.get_run(int(run["id"]))
        self.assertEqual(row["status"], "failed")
        self.assertIn("进程中断", row["error"])
        self.assertEqual(train_service.reconcile_stale_runs(self.repo), [], "重复对账应为空操作")

    def test_active_job_is_not_reconciled(self) -> None:
        run = self.repo.create_run("train", run_key="alive", config_json={})
        control = train_service.TrainControl()
        train_service.JOBS.register(int(run["id"]), control)
        try:
            self.assertEqual(train_service.stale_runs(self.repo), [])
            self.assertEqual(train_service.reconcile_stale_runs(self.repo), [])
            self.assertEqual(self.repo.get_run(int(run["id"]))["status"], "running")
        finally:
            train_service.JOBS.unregister(int(run["id"]))

    def test_resubmit_after_stale_run_recovers_instead_of_conflict(self) -> None:
        calls = self._patch_runner()
        first = train_service.start_training(self.config, dataset="ds-train", device="cpu", background=True)
        run_id = int(first["run_id"])
        thread = train_service.JOBS.thread(run_id)
        thread.join(timeout=15)                                     # type: ignore[union-attr]
        # 伪造"进程被杀"：任务已从注册表移除，但库里的行仍是 running
        self.repo.conn.execute("UPDATE runs SET status = 'running' WHERE id = ?", (run_id,))
        self.repo.conn.commit()
        train_service.JOBS.unregister(run_id)
        again = train_service.start_training(self.config, dataset="ds-train", device="cpu", background=False)
        self.assertFalse(again["cached"], "僵尸 running 不应被当作可复用的成功运行")
        self.assertEqual(len(calls), 2, "应当真的重跑一次")
        self.assertEqual(again["run"]["status"], "succeeded")



class TestApiContractFixes(TrainFixture):
    """§6.5 里如实记录的几处契约差异，已在源码侧修正——这里用测试钉住，避免回退。"""

    def setUp(self) -> None:
        super().setUp()
        from fastapi.testclient import TestClient

        from rdinspect.api.app import create_app

        self.client = TestClient(create_app(self.config))

    def tearDown(self) -> None:
        self.client.close()
        super().tearDown()

    def test_range_violation_returns_400_not_422(self) -> None:
        response = self.client.post("/api/train/runs", json={"dataset": "ds-train", "epochs": 99999})
        self.assertEqual(response.status_code, 400)
        self.assertIn("请求参数不合法", response.json()["message"])

    def test_idempotent_hit_returns_run_id(self) -> None:
        self._patch_runner()
        first = train_service.start_training(self.config, dataset="ds-train", device="cpu", background=False)
        cached = train_service.start_training(self.config, dataset="ds-train", device="cpu", background=False)
        self.assertTrue(cached["cached"])
        self.assertEqual(cached["run_id"], first["run_id"])
        self.assertIsNotNone(cached["run_id"])

    def test_promote_idempotent_response_keeps_shape(self) -> None:
        weights = self.config.weights_dir / "prod.pt"
        weights.write_bytes(b"fake")
        model = self.repo.upsert_model_version(name="prod-road", version="v1", status="validated",
                                              weights_path=str(weights),
                                              dataset_id=int(self.dataset["id"]))
        self.repo.update_model_version(int(model["id"]), gate_json={"passed": True, "reasons": []})
        gate_mod.promote_model(self.config, self.repo, int(model["id"]))
        again = gate_mod.promote_model(self.config, self.repo, int(model["id"]))
        self.assertFalse(again["changed"])
        self.assertIn("gate", again)
        self.assertEqual(again["weights_path"], str(weights))


class TestGateZeroCapabilityWarning(unittest.TestCase):
    """门禁只防"变差"，不防"一开始就差"：mAP50=0 被放行时必须给出显式告警。"""

    def test_zero_map_pass_emits_warning_note(self) -> None:
        decision = gate_mod.compare_metrics(_metrics(0.0), None, map50_tolerance=0.005,
                                            per_class_tolerance=0.02, min_map50=0.0)
        self.assertTrue(decision.passed)
        self.assertTrue(any("仍被放行" in note for note in decision.notes), decision.notes)

    def test_healthy_map_pass_has_no_warning(self) -> None:
        decision = gate_mod.compare_metrics(_metrics(0.9), None, map50_tolerance=0.005,
                                            per_class_tolerance=0.02, min_map50=0.0)
        self.assertTrue(decision.passed)
        self.assertFalse(any("仍被放行" in note for note in decision.notes))


class TestTrainApiStatusCodes(TrainFixture):
    """同步训练用 200、异步用 202；幂等命中不得谎称"已启动"。"""

    def setUp(self) -> None:
        super().setUp()
        from fastapi.testclient import TestClient

        from rdinspect.api.app import create_app

        self.client = TestClient(create_app(self.config))

    def tearDown(self) -> None:
        self.client.close()
        super().tearDown()

    def test_sync_wait_returns_200_and_async_returns_202(self) -> None:
        calls = self._patch_runner()
        sync = self.client.post("/api/train/runs",
                                json={"dataset": "ds-train", "device": "cpu", "wait": True})
        self.assertEqual(sync.status_code, 200)
        self.assertEqual(sync.json()["status"], "succeeded")
        self.assertEqual(sync.json()["run_id"], 1)
        self.assertEqual(len(calls), 1)
        # 同参数再提交（异步分支）→ 命中幂等，202 + cached = true，且 message 不谎称已启动
        again = self.client.post("/api/train/runs", json={"dataset": "ds-train", "device": "cpu"})
        self.assertEqual(again.status_code, 202)
        self.assertTrue(again.json()["cached"])
        self.assertIn("未启动新训练", again.json()["message"])


class TestExportPackageIsolation(TrainFixture):
    """同一权重按不同 imgsz 导出两份包时，模型文件必须彼此独立（硬链接会让后导出覆盖前者）。"""

    def test_two_exports_do_not_share_model_file(self) -> None:
        weights = self.config.weights_dir / "iso.pt"
        weights.write_bytes(b"weights")
        model = self.repo.upsert_model_version(
            name="iso-road", version="v1", status="validated", weights_path=str(weights),
            dataset_id=int(self.dataset["id"]),
            labels_json={"names": [cls["code"] for cls in self.repo.list_classes()]})
        source = self.tmp_path / "best.onnx"          # 模拟 ultralytics 复用的导出路径
        source.write_bytes(b"first-export")
        first = export_mod.build_export_package(
            self.config, self.repo, model, source, opset=17, dynamic_batch=True, half=False,
            simplify=True, imgsz=320, dataset=self.dataset, export_run_id=1, tolerance=1e-3)
        source.write_bytes(b"second-export")          # 第二次导出覆盖同名源文件
        second = export_mod.build_export_package(
            self.config, self.repo, model, source, opset=17, dynamic_batch=True, half=False,
            simplify=True, imgsz=640, dataset=self.dataset, export_run_id=2, tolerance=1e-3)
        self.assertNotEqual(first["dir"], second["dir"], "不同 imgsz 的包目录必须不同")
        self.assertEqual(Path(first["model_path"]).read_bytes(), b"first-export")
        self.assertEqual(Path(second["model_path"]).read_bytes(), b"second-export")
        manifest = json.loads((Path(first["dir"]) / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["model_sha256"], export_mod.sha256_file(first["model_path"]),
                         "manifest 里的哈希必须与包内实际文件一致")
        self.assertEqual(manifest["schema_version"], 1, "边缘端校验依赖 schema_version")
