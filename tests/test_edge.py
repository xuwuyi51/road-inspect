"""M4 边缘离线推理测试：包校验、输出格式、断点续跑、输入源与推理编排。

原则：**不依赖真实 ONNX/torch**——校验用假会话（get_inputs/get_outputs），推理用假检测器。
真实 ONNX 的端到端行为由 `scripts/e2e_m4.py` 覆盖。
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

from rdinspect.edge import infer as infer_mod
from rdinspect.edge import sources as sources_mod
from rdinspect.edge import writers as writers_mod
from rdinspect.edge.package import PackageError, sha256_file, validate_package

LABELS = ["longitudinal_crack", "transverse_crack", "pothole", "garbage", "alligator_crack"]


# ─────────────────────────── 夹具：假导出包 ───────────────────────────
class _FakeInput:
    def __init__(self, shape) -> None:
        self.shape = shape
        self.name = "images"


class _FakeMeta:
    def __init__(self, metadata: dict | None) -> None:
        self.custom_metadata_map = metadata or {}


class _FakeSession:
    def __init__(self, input_shape=(1, 3, 320, 320), output_shape=(1, 9, 2100),
                 metadata: dict | None = None) -> None:
        self._inputs = [_FakeInput(list(input_shape))]
        self._outputs = [_FakeInput(list(output_shape))]
        self._meta = _FakeMeta(metadata)

    def get_modelmeta(self):
        return self._meta

    def get_inputs(self):
        return self._inputs

    def get_outputs(self):
        return self._outputs

    def get_providers(self):
        return ["CPUExecutionProvider"]


def make_package(root: Path, *, labels: list[str] | None = None, schema_version: int | None = 1,
                 imgsz: int = 320, model_bytes: bytes = b"fake-onnx", mutate: dict | None = None,
                 write_preprocess: bool = True, write_labels: bool = True) -> tuple[Path, dict]:
    """造一个最小导出包（内容可篡改，用于校验用例）。"""
    root.mkdir(parents=True, exist_ok=True)
    model = root / "model.onnx"
    model.write_bytes(model_bytes)
    resolved_labels = list(labels if labels is not None else LABELS)
    if write_labels:
        (root / "labels.txt").write_text("\n".join(resolved_labels) + "\n", encoding="utf-8")
    manifest = {
        "name": "yolo11n-road", "version": "2026.09.19-r1", "task": "detection",
        "model_file": "model.onnx", "model_sha256": sha256_file(model),
        "opset": 17, "dynamic_batch": True, "imgsz": imgsz,
        "nc": len(resolved_labels), "labels": resolved_labels,
        "classes": [{"code": code} for code in resolved_labels],
        "dataset": {"name": "ds", "manifest_hash": "abc123"},
    }
    if schema_version is not None:
        manifest["schema_version"] = schema_version
    if mutate:
        manifest.update(mutate)
    (root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    if write_preprocess:
        (root / "preprocess.json").write_text(json.dumps({
            "input_size": [imgsz, imgsz], "channel_order": "RGB", "layout": "NCHW",
            "scale": 1 / 255, "resize": {"mode": "letterbox", "pad_value": 114},
            "postprocess": {"conf": 0.25, "iou": 0.5, "max_detections": 100},
        }, ensure_ascii=False), encoding="utf-8")
    return root, manifest


class FakeDetector:
    """按图尺寸返回固定框的假检测器（与 `Detector` 协议兼容）。"""

    names = {index: code for index, code in enumerate(LABELS)}

    def __init__(self, boxes: list[tuple[str, tuple[float, float, float, float], float]] | None = None,
                 *, classes: list[str] | None = None) -> None:
        self.imgsz = 320
        self.calls = 0
        self._script = boxes if boxes is not None else [
            ("transverse_crack", (0.1, 0.2, 0.4, 0.35), 0.9)]

    def predict(self, image: Image.Image):
        from rdinspect.prelabel.detector import Detection

        self.calls += 1
        width, height = image.size
        results = []
        for index, (code, box, score) in enumerate(self._script):
            results.append(Detection(class_index=index, class_name=code, score=score,
                                     bbox=(box[0] * width, box[1] * height, box[2] * width, box[3] * height)))
        return results


class PackageFixture(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    @staticmethod
    def session_factory(session: _FakeSession):
        return lambda path: session

    def photos(self, count: int = 3, *, size: tuple[int, int] = (160, 120)) -> list[Path]:
        root = self.tmp / "photos"
        root.mkdir(parents=True, exist_ok=True)
        paths = []
        for index in range(count):
            image = Image.new("RGB", size, (60 + index * 20, 60, 62))
            draw = ImageDraw.Draw(image)
            draw.rectangle([10 + index, 20, 60 + index, 50], outline=(20, 20, 20), width=2)
            path = root / f"img_{index:03d}.jpg"
            image.save(path, quality=90)
            paths.append(path)
        return paths


# ─────────────────────────── 包校验 ───────────────────────────
class TestValidatePackage(PackageFixture):
    def test_valid_package_passes_and_reports_checks(self) -> None:
        root, _ = make_package(self.tmp / "pkg")
        info = validate_package(root, session_factory=self.session_factory(_FakeSession()))
        self.assertEqual(info.labels, LABELS)
        self.assertEqual(info.imgsz, 320)
        self.assertEqual(info.schema_version, 1)
        self.assertEqual(info.model_label, "yolo11n-road:2026.09.19-r1")
        self.assertTrue(any("model_sha256" in check for check in info.checks))

    def test_missing_manifest_is_refused(self) -> None:
        with self.assertRaises(PackageError) as ctx:
            validate_package(self.tmp / "missing-dir")
        self.assertIn("目录不存在", str(ctx.exception))
        empty = self.tmp / "empty"
        empty.mkdir()
        with self.assertRaises(PackageError) as ctx:
            validate_package(empty)
        self.assertIn("manifest.json", str(ctx.exception))

    def test_missing_schema_version_is_refused(self) -> None:
        root, _ = make_package(self.tmp / "pkg", schema_version=None)
        with self.assertRaises(PackageError) as ctx:
            validate_package(root, session_factory=self.session_factory(_FakeSession()))
        self.assertIn("schema_version", str(ctx.exception))

    def test_newer_schema_version_is_refused(self) -> None:
        root, _ = make_package(self.tmp / "pkg", schema_version=99)
        with self.assertRaises(PackageError) as ctx:
            validate_package(root, session_factory=self.session_factory(_FakeSession()))
        self.assertIn("高于本端支持", str(ctx.exception))

    def test_hash_mismatch_is_refused_but_can_be_skipped(self) -> None:
        root, _ = make_package(self.tmp / "pkg")
        (root / "model.onnx").write_bytes(b"tampered")
        with self.assertRaises(PackageError) as ctx:
            validate_package(root, session_factory=self.session_factory(_FakeSession()))
        self.assertIn("哈希不匹配", str(ctx.exception))
        info = validate_package(root, session_factory=self.session_factory(_FakeSession()), verify_hash=False)
        self.assertEqual(info.labels, LABELS)

    def test_label_drift_is_refused(self) -> None:
        root, _ = make_package(self.tmp / "pkg", labels=LABELS[:4])
        with self.assertRaises(PackageError) as ctx:
            validate_package(root, session_factory=self.session_factory(_FakeSession()))
        self.assertIn("4+nc", str(ctx.exception))

    def test_labels_file_disagreement_is_refused(self) -> None:
        root, _ = make_package(self.tmp / "pkg")
        (root / "labels.txt").write_text("\n".join(LABELS[:3]) + "\n", encoding="utf-8")
        with self.assertRaises(PackageError) as ctx:
            validate_package(root, session_factory=self.session_factory(_FakeSession()))
        self.assertIn("labels.txt", str(ctx.exception))

    def test_expected_label_order_mismatch_is_refused(self) -> None:
        root, _ = make_package(self.tmp / "pkg")
        with self.assertRaises(PackageError) as ctx:
            validate_package(root, session_factory=self.session_factory(_FakeSession()),
                             expect_labels=list(reversed(LABELS)))
        self.assertIn("类别顺序与期望不一致", str(ctx.exception))

    def test_missing_preprocess_is_refused(self) -> None:
        root, _ = make_package(self.tmp / "pkg", write_preprocess=False)
        with self.assertRaises(PackageError) as ctx:
            validate_package(root, session_factory=self.session_factory(_FakeSession()))
        self.assertIn("preprocess.json", str(ctx.exception))

    def test_non_rgb_channel_order_is_refused(self) -> None:
        root, _ = make_package(self.tmp / "pkg")
        payload = json.loads((root / "preprocess.json").read_text(encoding="utf-8"))
        payload["channel_order"] = "BGR"
        (root / "preprocess.json").write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaises(PackageError) as ctx:
            validate_package(root, session_factory=self.session_factory(_FakeSession()))
        self.assertIn("RGB", str(ctx.exception))

    def test_static_input_size_mismatch_is_refused(self) -> None:
        root, _ = make_package(self.tmp / "pkg", imgsz=640)
        session = _FakeSession(input_shape=(1, 3, 320, 320))
        with self.assertRaises(PackageError) as ctx:
            validate_package(root, session_factory=self.session_factory(session))
        self.assertIn("输入尺寸", str(ctx.exception))

    def test_static_output_with_anchor_dim_is_accepted(self) -> None:
        """静态导出时输出是 (1, 9, 2100)：2100 是 anchor 数，不能当成通道数。"""
        root, _ = make_package(self.tmp / "pkg")
        info = validate_package(root, session_factory=self.session_factory(_FakeSession()))
        self.assertTrue(any("输出通道=9" in check for check in info.checks))

    def test_output_without_channel_dim_is_refused(self) -> None:
        root, _ = make_package(self.tmp / "pkg")
        session = _FakeSession(output_shape=(1, 2100, 4))
        with self.assertRaises(PackageError) as ctx:
            validate_package(root, session_factory=self.session_factory(session))
        self.assertIn("4+nc", str(ctx.exception))

    def test_dynamic_output_shape_is_skipped(self) -> None:
        root, _ = make_package(self.tmp / "pkg")
        session = _FakeSession(input_shape=[1, 3, "height", "width"],
                               output_shape=[1, "channels", "anchors"])
        info = validate_package(root, session_factory=self.session_factory(session))
        self.assertTrue(any("动态" in check for check in info.checks))
        dynamic_batch = _FakeSession(input_shape=["batch", 3, 320, 320], output_shape=[1, 9, "anchors"])
        info = validate_package(root, session_factory=self.session_factory(dynamic_batch))
        self.assertTrue(info.dynamic_batch)

    def test_embedded_names_and_imgsz_are_cross_checked(self) -> None:
        """文本清单被整体改过（labels.txt 与 manifest 一致）时，靠模型内嵌 names 兜底。"""
        root, _ = make_package(self.tmp / "pkg")
        metadata = {"names": "{0: 'x', 1: 'y', 2: 'z', 3: 'w', 4: 'v'}", "imgsz": "[320, 320]"}
        with self.assertRaises(PackageError) as ctx:
            validate_package(root, session_factory=self.session_factory(_FakeSession(metadata=metadata)))
        self.assertIn("内嵌类别名", str(ctx.exception))

        mismatched_size = {"names": "{" + ", ".join(f"{i}: '{code}'" for i, code in enumerate(LABELS)) + "}",
                           "imgsz": "[640, 640]"}
        with self.assertRaises(PackageError) as ctx:
            validate_package(root, session_factory=self.session_factory(_FakeSession(metadata=mismatched_size)))
        self.assertIn("内嵌 imgsz", str(ctx.exception))

        good = {"names": "{" + ", ".join(f"{i}: '{code}'" for i, code in enumerate(LABELS)) + "}",
                "imgsz": "[320, 320]"}
        info = validate_package(root, session_factory=self.session_factory(_FakeSession(metadata=good)))
        self.assertTrue(any("内嵌类别名一致" in check for check in info.checks))

    def test_missing_model_file_is_refused(self) -> None:
        root, _ = make_package(self.tmp / "pkg")
        (root / "model.onnx").unlink()
        with self.assertRaises(PackageError) as ctx:
            validate_package(root, session_factory=self.session_factory(_FakeSession()))
        self.assertIn("缺少模型文件", str(ctx.exception))


# ─────────────────────────── 输出与断点续跑 ───────────────────────────
class TestWriters(PackageFixture):
    def _record(self, name: str = "a.jpg", detections: list[dict] | None = None) -> dict:
        return writers_mod.record_for(
            image=name, source=f"/in/{name}", image_sha256="deadbeef", index=0, frame_index=None,
            ts="2026-09-19T10:00:00", gps={"lat": 31.2, "lon": 121.4},
            model={"name": "m", "version": "v1", "imgsz": 320},
            detections=detections if detections is not None else
            [{"class": "pothole", "conf": 0.8765432, "bbox": [0.1, 0.2, 0.3, 0.4]}],
            tiles=1, elapsed_ms=12.3456, width=160, height=120)

    def test_record_matches_documented_schema(self) -> None:
        record = self._record()
        for key in ("image", "ts", "gps", "model", "detections", "tiles", "elapsed_ms"):
            self.assertIn(key, record, f"JSONL 契约字段缺失: {key}")
        self.assertEqual(record["detections"][0]["bbox"], [0.1, 0.2, 0.3, 0.4])
        self.assertEqual(record["detections"][0]["conf"], 0.876543)
        self.assertEqual(record["elapsed_ms"], 12.346)

    def test_jsonl_append_and_csv_shape(self) -> None:
        jsonl_path = self.tmp / "results.jsonl"
        csv_path = self.tmp / "results.csv"
        with writers_mod.JsonlWriter(jsonl_path) as jsonl, writers_mod.CsvWriter(csv_path) as csv:
            jsonl.write(self._record("a.jpg"))
            csv.write(self._record("a.jpg"))
            empty = self._record("b.jpg", detections=[])
            jsonl.write(empty)
            csv.write(empty)
        rows = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(rows), 2)
        lines = csv_path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(lines[0], ",".join(writers_mod.CSV_HEADER))
        self.assertEqual(len(lines), 3, "有检测的图 1 行 + 无检测的图也要占 1 行")
        self.assertTrue(lines[2].startswith("b.jpg,"))
        self.assertEqual(lines[2].split(",")[4], "", "无检测行的 class 为空")

    def test_csv_reopening_does_not_duplicate_header(self) -> None:
        csv_path = self.tmp / "results.csv"
        with writers_mod.CsvWriter(csv_path) as csv:
            csv.write(self._record("a.jpg"))
        with writers_mod.CsvWriter(csv_path) as csv:
            csv.write(self._record("b.jpg"))
        lines = csv_path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(lines[0], ",".join(writers_mod.CSV_HEADER))
        self.assertEqual(sum(1 for line in lines if line.startswith("image,")), 1)

    def test_state_roundtrip_and_jsonl_rebuild(self) -> None:
        state_path = self.tmp / ".infer-state.json"
        jsonl_path = self.tmp / "results.jsonl"
        state = writers_mod.ResumeState(path=state_path)
        state.mark(image_sha256="aaa", source="/in/a.jpg", detections=2)
        writers_mod.save_state(state)
        self.assertFalse(state_path.with_name(state_path.name + ".tmp").exists(), "原子写不留临时文件")
        loaded = writers_mod.load_state(state_path, jsonl_path)
        self.assertTrue(loaded.has(image_sha256="aaa", source="/in/a.jpg"))
        self.assertEqual(loaded.records, 1)

        # 状态文件丢失时用 JSONL 兜底
        state_path.unlink()
        with writers_mod.JsonlWriter(jsonl_path) as jsonl:
            jsonl.write(self._record("c.jpg"))
        rebuilt = writers_mod.load_state(state_path, jsonl_path)
        self.assertTrue(rebuilt.has(image_sha256="deadbeef", source="/in/c.jpg"))
        self.assertEqual(rebuilt.records, 1)
        self.assertEqual(rebuilt.detections, 1)

    def test_corrupt_jsonl_line_is_ignored(self) -> None:
        jsonl_path = self.tmp / "results.jsonl"
        jsonl_path.write_text('{"image":"a.jpg","detections":[]}\n{"broken":\n', encoding="utf-8")
        state = writers_mod.load_state(self.tmp / ".state", jsonl_path)
        self.assertEqual(state.records, 1)

    def test_summarize_records(self) -> None:
        records = [self._record("a.jpg"), self._record("b.jpg", detections=[]),
                   {**self._record("c.jpg"), "elapsed_ms": 30.0}]
        summary = writers_mod.summarize_records(records)
        self.assertEqual(summary["images"], 3)
        self.assertEqual(summary["detections"], 2)
        self.assertEqual(summary["per_class"], {"pothole": 2})
        self.assertIsNotNone(summary["elapsed_ms"]["p95"])


# ─────────────────────────── 输入源 ───────────────────────────
class TestSources(PackageFixture):
    def test_list_images_is_sorted_and_filtered(self) -> None:
        root = self.tmp / "in"
        (root / "sub").mkdir(parents=True)
        for name in ("b.jpg", "a.png", "note.txt", ".hidden.jpg"):
            (root / name).write_bytes(b"x")
        (root / "sub" / "c.jpeg").write_bytes(b"x")
        found = sources_mod.list_images(root)
        self.assertEqual([path.name for path in found], ["a.png", "b.jpg", "c.jpeg"])

    def test_iter_source_dispatches_by_type(self) -> None:
        config = sources_mod.EdgeConfig()
        image = self.photos(1)[0]
        items = list(sources_mod.iter_source(str(image), config=config))
        self.assertEqual(items[0]["kind"], "image")
        items = list(sources_mod.iter_source(str(image.parent), config=config))
        self.assertEqual(len(items), 1)
        video = self.tmp / "v.mp4"
        video.write_bytes(b"x")
        self.assertEqual(list(sources_mod.iter_source(str(video), config=config))[0]["kind"], "video")
        self.assertEqual(list(sources_mod.iter_source("rtsp://cam/1", config=config))[0]["kind"], "stream")

    def test_iter_source_rejects_unknown_and_missing(self) -> None:
        config = sources_mod.EdgeConfig()
        weird = self.tmp / "x.xyz"
        weird.write_bytes(b"x")
        with self.assertRaises(sources_mod.SourceError):
            list(sources_mod.iter_source(str(weird), config=config))
        with self.assertRaises(sources_mod.SourceError):
            list(sources_mod.iter_source(str(self.tmp / "nope"), config=config))
        empty = self.tmp / "empty"
        empty.mkdir()
        with self.assertRaises(sources_mod.SourceError):
            list(sources_mod.iter_source(str(empty), config=config))

    def test_config_defaults_and_yaml_roundtrip(self) -> None:
        config = sources_mod.EdgeConfig()
        self.assertIsNone(config.imgsz, "默认不覆盖导出包里的 imgsz")
        self.assertEqual(config.threads, 4)
        path = self.tmp / "edge.yaml"
        path.write_text("model:\n  device: cpu\ninference:\n  imgsz: null\n  tile:\n    enabled: true\n"
                        "runtime:\n  threads: 2\nlimits:\n  min_free_disk_mb: 1\n", encoding="utf-8")
        loaded = sources_mod.EdgeConfig.load(path)
        self.assertTrue(loaded.tile.enabled)
        self.assertEqual(loaded.threads, 2)
        self.assertIsNone(sources_mod.EdgeConfig.load(self.tmp / "missing.yaml").package_dir)

    def test_video_backend_is_known_value(self) -> None:
        self.assertIn(sources_mod.video_backend(), ("opencv", "ffmpeg", "none"))

    def test_check_disk_space(self) -> None:
        ok, free_mb = sources_mod.check_disk_space(self.tmp, min_free_mb=1)
        self.assertTrue(ok)
        self.assertGreater(free_mb, 0)
        ok, _ = sources_mod.check_disk_space(self.tmp, min_free_mb=10 ** 9)
        self.assertFalse(ok)


# ─────────────────────────── 推理编排 ───────────────────────────
class TestRunInference(PackageFixture):
    def _run(self, *, count: int = 3, resume: bool | None = None, detector=None, limit=None,
             out_name: str = "out", config: sources_mod.EdgeConfig | None = None) -> tuple[dict, Path]:
        photo_paths = self.photos(count)
        root, _ = make_package(self.tmp / "pkg")
        fake = detector or FakeDetector()
        cfg = config or sources_mod.EdgeConfig()
        out = self.tmp / out_name
        result = infer_mod.run_inference(
            package_dir=root, source=str(photo_paths[0].parent), out_dir=out, config=cfg,
            resume=resume, limit=limit, expect_labels=LABELS,
            session_factory=self.session_factory(_FakeSession()),
            detector_factory=lambda package, _fake=fake: _fake)
        return result.as_dict(), out

    def test_inference_writes_jsonl_csv_and_state(self) -> None:
        payload, out = self._run(count=3)
        self.assertEqual(payload["images"], 3)
        self.assertEqual(payload["detections"], 3)
        self.assertEqual(payload["cumulative_images"], 3)
        self.assertEqual(len(list(writers_mod.read_jsonl(out / "results.jsonl"))), 3)
        self.assertTrue((out / "results.csv").exists())
        self.assertTrue((out / "results.jsonl").exists())
        self.assertTrue((out / ".infer-state.json").exists())
        self.assertGreater(payload["fps"], 0)

    def test_resume_skips_everything_on_second_run(self) -> None:
        first, out = self._run(count=3)
        self.assertEqual(first["images"], 3)
        second = infer_mod.run_inference(
            package_dir=self.tmp / "pkg", source=str(self.tmp / "photos"), out_dir=out,
            config=sources_mod.EdgeConfig(), resume=True, expect_labels=LABELS,
            session_factory=self.session_factory(_FakeSession()),
            detector_factory=lambda package: FakeDetector())
        self.assertEqual(second.images, 0)
        self.assertEqual(second.skipped, 3)
        self.assertEqual(second.cumulative_images, 3, "累计口径不应因跳过而缩小")
        lines = list(writers_mod.read_jsonl(out / "results.jsonl"))
        self.assertEqual(len(lines), 3, "跳过时不得重复追加")

    def test_resume_rebuilds_from_jsonl_when_state_lost(self) -> None:
        _payload, out = self._run(count=2)
        (out / ".infer-state.json").unlink()                 # 模拟状态文件被删/损坏
        again = infer_mod.run_inference(
            package_dir=self.tmp / "pkg", source=str(self.tmp / "photos"), out_dir=out,
            config=sources_mod.EdgeConfig(), resume=True, expect_labels=LABELS,
            session_factory=self.session_factory(_FakeSession()),
            detector_factory=lambda package: FakeDetector())
        self.assertEqual(again.skipped, 2)
        self.assertEqual(again.images, 0)

    def test_no_resume_reprocesses_all(self) -> None:
        _first, out = self._run(count=2)
        again = infer_mod.run_inference(
            package_dir=self.tmp / "pkg", source=str(self.tmp / "photos"), out_dir=out,
            config=sources_mod.EdgeConfig(), resume=False, expect_labels=LABELS,
            session_factory=self.session_factory(_FakeSession()),
            detector_factory=lambda package: FakeDetector())
        self.assertEqual(again.images, 2)
        self.assertEqual(again.skipped, 0)

    def test_limit_caps_processed_images(self) -> None:
        payload, _out = self._run(count=5, limit=2)
        self.assertEqual(payload["images"], 2)

    def test_bad_image_is_recorded_and_does_not_abort(self) -> None:
        photo_paths = self.photos(2)
        (photo_paths[0].parent / "broken.jpg").write_bytes(b"not-an-image")
        root, _ = make_package(self.tmp / "pkg")
        result = infer_mod.run_inference(
            package_dir=root, source=str(photo_paths[0].parent), out_dir=self.tmp / "out",
            config=sources_mod.EdgeConfig(), expect_labels=LABELS,
            session_factory=self.session_factory(_FakeSession()),
            detector_factory=lambda package: FakeDetector())
        self.assertEqual(result.images, 2)
        self.assertEqual(len(result.errors), 1)
        self.assertIn("broken.jpg", result.errors[0]["source"])

    def test_invalid_package_is_refused_before_any_output(self) -> None:
        root, _ = make_package(self.tmp / "pkg")
        (root / "manifest.json").unlink()
        with self.assertRaises(PackageError):
            infer_mod.run_inference(
                package_dir=root, source=str(self.photos(1)[0]), out_dir=self.tmp / "out",
                config=sources_mod.EdgeConfig(), session_factory=self.session_factory(_FakeSession()),
                detector_factory=lambda package: FakeDetector())

    def test_tiling_path_is_used_when_enabled(self) -> None:
        config = sources_mod.EdgeConfig(tile=sources_mod.TileSpec(enabled=True, size=64, overlap=0.2))
        detector = FakeDetector()
        payload, out = self._run(count=1, detector=detector, config=config)
        self.assertEqual(payload["images"], 1)
        lines = list(writers_mod.read_jsonl(out / "results.jsonl"))
        self.assertGreater(lines[0]["tiles"], 1, "开启切片后应记录多个 tile")
        self.assertGreater(detector.calls, 1)

    def test_snapshots_are_written_only_for_hits(self) -> None:
        config = sources_mod.EdgeConfig(save_hit_snapshots=True)
        payload, out = self._run(count=2, config=config)
        self.assertEqual(payload["detections"], 2)
        snaps = sorted((out / "snaps").glob("*.jpg"))
        self.assertEqual(len(snaps), 2)
        empty = FakeDetector(boxes=[])
        infer_mod.run_inference(
            package_dir=self.tmp / "pkg", source=str(self.tmp / "photos"), out_dir=self.tmp / "out2",
            config=sources_mod.EdgeConfig(save_hit_snapshots=True), expect_labels=LABELS,
            session_factory=self.session_factory(_FakeSession()),
            detector_factory=lambda package: empty)
        self.assertFalse((self.tmp / "out2" / "snaps").exists(), "没有命中就不该产生快照")

    def test_exif_gps_and_time_are_captured(self) -> None:
        from PIL import ExifTags
        from PIL.TiffImagePlugin import IFDRational

        image = Image.new("RGB", (160, 120), (80, 80, 82))
        exif = Image.Exif()
        exif[36867] = "2026:09:19 08:30:00"
        gps = exif.get_ifd(ExifTags.IFD.GPSInfo)
        gps[1], gps[3] = "N", "E"
        gps[2] = (IFDRational(31, 1), IFDRational(14, 1), IFDRational(0, 1))
        gps[4] = (IFDRational(121, 1), IFDRational(28, 1), IFDRational(0, 1))
        source = self.tmp / "photos" / "gps.jpg"
        source.parent.mkdir(parents=True, exist_ok=True)
        image.save(source, exif=exif.tobytes(), quality=90)

        root, _ = make_package(self.tmp / "pkg")
        out = self.tmp / "out"
        infer_mod.run_inference(package_dir=root, source=str(source), out_dir=out,
                                config=sources_mod.EdgeConfig(), expect_labels=LABELS,
                                session_factory=self.session_factory(_FakeSession()),
                                detector_factory=lambda package: FakeDetector())
        record = next(writers_mod.read_jsonl(out / "results.jsonl"))
        self.assertEqual(record["ts"], "2026-09-19T08:30:00")
        self.assertAlmostEqual(record["gps"]["lat"], 31.233333, places=5)
        self.assertAlmostEqual(record["gps"]["lon"], 121.466667, places=5)

    def test_normalize_bbox_clips_and_drops_degenerate(self) -> None:
        self.assertEqual(infer_mod._normalize_bbox((-5, -5, 20, 20), 100, 100), [0.0, 0.0, 0.2, 0.2])
        self.assertIsNone(infer_mod._normalize_bbox((10, 10, 10, 40), 100, 100))
        self.assertIsNone(infer_mod._normalize_bbox((0, 0, 1, 1), 0, 0))

    def test_benchmark_reports_latency_and_fps(self) -> None:
        photos = self.photos(2)
        root, _ = make_package(self.tmp / "pkg")
        report = infer_mod.benchmark(root, photos, config=sources_mod.EdgeConfig(threads=4),
                                     repeat=2, warmup=1, expect_labels=LABELS,
                                     session_factory=self.session_factory(_FakeSession()),
                                     detector_factory=lambda package: FakeDetector())
        self.assertEqual(report["images"], 4)
        self.assertEqual(report["warmup"], 1)
        self.assertGreater(report["fps"], 0)
        self.assertIn("p95", report["latency_ms"])
        self.assertEqual(report["package"]["nc"], len(LABELS))


if __name__ == "__main__":
    unittest.main()


class TestPackagedInstallFallbacks(unittest.TestCase):
    """pip 安装（非 editable）时没有仓库 configs/：边缘命令必须仍能用内置默认值跑起来。"""

    def test_missing_default_config_falls_back_to_builtin_defaults(self) -> None:
        import os
        import tempfile

        from rdinspect import config as config_mod

        cwd = os.getcwd()
        original = config_mod.default_config_candidates
        with tempfile.TemporaryDirectory() as tmp:
            os.chdir(tmp)
            os.environ.pop("ROAD_INSPECT_CONFIG", None)
            # 模拟 pip 安装：仓库 configs/、CWD/configs/、包内 configs/ 全部不存在
            config_mod.default_config_candidates = lambda: [Path(tmp) / "configs" / "default.yaml"]
            try:
                config = config_mod.load_config(data_dir=Path(tmp) / "data")
            finally:
                config_mod.default_config_candidates = original
                os.chdir(cwd)
        self.assertEqual(config.host, "127.0.0.1")
        self.assertEqual(config.port, 8787)
        self.assertIsNone(config.config_path)
        self.assertIn("transverse_crack", config.prelabel.class_aliases.values())

    def test_explicit_missing_config_still_errors(self) -> None:
        import tempfile

        from rdinspect.config import ConfigError, load_config

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ConfigError):
                load_config(Path(tmp) / "nope.yaml")
