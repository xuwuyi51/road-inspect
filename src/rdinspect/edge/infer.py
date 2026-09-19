"""边缘推理编排（M4）：输入 → ONNX 推理 → JSONL/CSV，含断点续跑、切片与性能基准。

不依赖 torch/ultralytics/数据库：只用到 ``onnxruntime``（+ ``numpy``/``pillow``，视频另需
``opencv`` 或 ``ffmpeg``）。因此边缘设备可以只装最小依赖跑完整条链路。

关键行为：

* **拒绝启动**：推理前先 :func:`~rdinspect.edge.package.validate_package` 校验导出包，
  版本/类别/哈希任一不匹配就抛错，不会"先跑起来再发现结果不可信"；
* **断点续跑**：`--resume` 时以 JSONL + 状态文件为真相跳过已处理项，重复运行不重复计数；
* **单张失败不中断**：坏图/超时只记入 `errors` 并继续（边缘无人值守场景）；
* **性能可测**：`benchmark()` 给出 warmup + p50/p95 时延、FPS、RSS 峰值，用于验收性能目标。
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from PIL import Image, ImageDraw

from . import sources as sources_mod
from .package import PackageInfo, sha256_file, validate_package
from .writers import CsvWriter, JsonlWriter, ResumeState, load_state, record_for, save_state

BOX_COLOR = (231, 76, 60)
#: 单张图的失败信息最多保留多少条（避免边缘长跑把内存吃满）
MAX_ERRORS = 20


def _normalize_bbox(bbox: Sequence[float], width: int, height: int) -> list[float] | None:
    """像素框 → 归一化 [x1,y1,x2,y2]（裁剪越界；过小返回 None）。"""
    if width <= 0 or height <= 0:
        return None
    x1, x2 = sorted((max(0.0, min(float(width), float(bbox[0]))), max(0.0, min(float(width), float(bbox[2])))))
    y1, y2 = sorted((max(0.0, min(float(height), float(bbox[1]))), max(0.0, min(float(height), float(bbox[3])))))
    if x2 - x1 < 1.0 or y2 - y1 < 1.0:
        return None
    return [round(x1 / width, 6), round(y1 / height, 6), round(x2 / width, 6), round(y2 / height, 6)]


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _ratio_to_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        try:
            return float(value.numerator) / float(value.denominator)
        except Exception:  # noqa: BLE001 - IFDRational/元组等形态
            return None


def read_exif(image: Image.Image) -> tuple[dict[str, float] | None, str | None]:
    """从 EXIF 读 GPS 与拍摄时间（边缘端不引入额外依赖，PIL 自带）。"""
    gps: dict[str, float] | None = None
    ts: str | None = None
    try:
        exif = image.getexif()
        if not exif:
            return None, None
        raw_ts = exif.get(36867) or exif.get(306)      # DateTimeOriginal / DateTime
        if isinstance(raw_ts, str) and raw_ts.strip():
            ts = raw_ts.strip().replace(":", "-", 2).replace(" ", "T")
        gps_ifd = exif.get_ifd(34853)                  # GPSInfo
        if gps_ifd:
            lat, lon = gps_ifd.get(2), gps_ifd.get(4)
            if lat and lon:
                def _dms(values: Any) -> float | None:
                    parts = [_ratio_to_float(part) for part in (values or [])]
                    if len(parts) != 3 or any(part is None for part in parts):
                        return None
                    return parts[0] + parts[1] / 60.0 + parts[2] / 3600.0

                lat_value, lon_value = _dms(lat), _dms(lon)
                if lat_value is not None and lon_value is not None:
                    if str(gps_ifd.get(1, "N")).upper().startswith("S"):
                        lat_value = -lat_value
                    if str(gps_ifd.get(3, "E")).upper().startswith("W"):
                        lon_value = -lon_value
                    gps = {"lat": round(lat_value, 6), "lon": round(lon_value, 6)}
    except Exception:  # noqa: BLE001 - EXIF 损坏不影响推理
        return None, ts
    return gps, ts


@dataclass
class InferResult:
    """一次推理运行的汇总（CLI 直接打印/写盘）。"""

    images: int = 0                 # 本次运行新处理的数量
    detections: int = 0             # 本次运行新产生的检测数
    skipped: int = 0                # --resume 跳过的数量
    cumulative_images: int = 0      # 含历史结果的累计数量（状态文件口径）
    cumulative_detections: int = 0
    errors: list[dict[str, Any]] = field(default_factory=list)
    per_class: dict[str, int] = field(default_factory=dict)
    elapsed_ms: float = 0.0
    out_dir: str | None = None
    jsonl: str | None = None
    csv: str | None = None
    state_file: str | None = None
    backend: str = "onnxruntime"
    package: dict[str, Any] = field(default_factory=dict)
    started_at: str | None = None
    finished_at: str | None = None

    @property
    def fps(self) -> float | None:
        return None if self.elapsed_ms <= 0 else round(self.images / (self.elapsed_ms / 1000.0), 3)

    def as_dict(self) -> dict[str, Any]:
        return {"package": self.package, "images": self.images, "detections": self.detections,
                "skipped": self.skipped, "cumulative_images": self.cumulative_images,
                "cumulative_detections": self.cumulative_detections,
                "per_class": self.per_class,
                "elapsed_ms": round(self.elapsed_ms, 3), "fps": self.fps,
                "errors": self.errors, "error_count": len(self.errors),
                "out_dir": self.out_dir, "jsonl": self.jsonl, "csv": self.csv,
                "state_file": self.state_file, "backend": self.backend,
                "started_at": self.started_at, "finished_at": self.finished_at}


def build_detector(package: PackageInfo, *, threads: int | None = None,
                   providers: Sequence[str] | None = None, conf: float | None = None,
                   iou: float | None = None, imgsz: int | None = None,
                   max_detections: int | None = None) -> Any:
    """按导出包构造 ONNX 检测器（可限制算子内线程数以贴合边缘 CPU 预算）。"""
    from .onnx_runtime import OnnxDetector  # noqa: PLC0415

    return OnnxDetector(
        package.model_path, class_names=package.labels,
        imgsz=int(imgsz or package.imgsz), conf=float(conf if conf is not None else package.conf),
        iou=float(iou if iou is not None else package.iou),
        max_detections=int(max_detections or package.max_detections),
        providers=list(providers) if providers else None, threads=threads,
    )


def _detect(detector: Any, image: Image.Image, *, tile: sources_mod.TileSpec) -> tuple[list[dict[str, Any]], int]:
    """一次检测 → 归一化结果 + 切片数（开启切片时按 SAHI 策略合并）。"""
    width, height = image.size
    if tile.enabled:
        from ..prelabel.sahi import predict_sliced  # noqa: PLC0415 - 复用 ADR-0006 的同一套几何

        tile_config = _TileLike(size=tile.size, overlap=tile.overlap, merge_iou=tile.merge_iou)
        detections, tiles = predict_sliced(image, detector, tile_config=tile_config)
    else:
        detections, tiles = detector.predict(image), 1
    results: list[dict[str, Any]] = []
    for detection in detections:
        bbox = _normalize_bbox(detection.bbox, width, height)
        if bbox is None:
            continue
        results.append({"class": str(detection.class_name), "conf": float(detection.score), "bbox": bbox})
    return results, int(tiles)


@dataclass
class _TileLike:
    """给 `predict_sliced` 用的最小 tile 配置（避免依赖 configs 包）。"""

    size: int
    overlap: float
    merge_iou: float
    enabled: bool = True


def draw_detections(image: Image.Image, detections: Iterable[dict[str, Any]]) -> Image.Image:
    """把检测框画到副本上（用于命中快照）。"""
    canvas = image.convert("RGB").copy()
    if max(canvas.size) > 1600:
        scale = 1600 / max(canvas.size)
        canvas = canvas.resize((int(canvas.width * scale), int(canvas.height * scale)))
    draw = ImageDraw.Draw(canvas)
    width, height = canvas.size
    for item in detections:
        x1, y1, x2, y2 = item["bbox"]
        box = (x1 * width, y1 * height, x2 * width, y2 * height)
        draw.rectangle(box, outline=BOX_COLOR, width=2)
        draw.text((box[0] + 2, max(0, box[1] - 11)), f"{item['class']} {item['conf']:.2f}", fill=BOX_COLOR)
    return canvas


def run_inference(*, package_dir: str | Path, source: str, out_dir: str | Path,
                  config: sources_mod.EdgeConfig, resume: bool | None = None,
                  limit: int | None = None, snapshots: bool | None = None,
                  expect_labels: Sequence[str] | None = None, verify_hash: bool = True,
                  detector_factory: Callable[[PackageInfo], Any] | None = None,
                  session_factory: Callable[[str], Any] | None = None,
                  progress: Callable[[int, int, str], None] | None = None) -> InferResult:
    """执行一次边缘推理（目录/单图/视频/流），返回汇总。"""
    started_wall = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    package = validate_package(package_dir, expect_labels=expect_labels, verify_hash=verify_hash,
                               session_factory=session_factory)
    tile = config.tile
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_root / config.jsonl_name
    csv_path = out_root / config.csv_name
    state_path = out_root / config.state_name
    use_resume = config.resume_enabled if resume is None else bool(resume)
    snapshots_enabled = config.save_hit_snapshots if snapshots is None else bool(snapshots)
    snapshot_dir = out_root / config.snapshot_dir
    free_ok, free_mb = sources_mod.check_disk_space(out_root, min_free_mb=config.min_free_disk_mb)

    state = load_state(state_path, jsonl_path) if use_resume else ResumeState(path=state_path)
    result = InferResult(out_dir=str(out_root), jsonl=str(jsonl_path), csv=str(csv_path),
                         state_file=str(state_path), package=package.as_dict(), started_at=started_wall)
    if not free_ok:
        result.errors.append({"source": str(out_root),
                              "error": f"磁盘余量不足：{free_mb}MB < {config.min_free_disk_mb}MB"})
    detector = detector_factory(package) if detector_factory else build_detector(
        package, threads=config.threads, conf=config.conf, iou=config.iou,
        imgsz=config.imgsz, max_detections=config.max_detections)
    model_info = {"name": package.name, "version": package.version, "imgsz": package.imgsz,
                  "schema_version": package.schema_version, "labels": package.labels}

    def _note_error(source_label: str, exc: Exception) -> None:
        if len(result.errors) < MAX_ERRORS:
            result.errors.append({"source": source_label, "error": f"{type(exc).__name__}: {exc}"})

    def _handle(image: Image.Image, *, name: str, source_label: str, resume_key: str,
                digest: str, frame_index: int | None, tile_config: Any) -> None:
        """推理一张图并落盘（含 EXIF 的 GPS/时间）。"""
        if max(image.size) > config.max_image_side:
            ratio = config.max_image_side / max(image.size)
            image = image.resize((max(1, int(image.width * ratio)), max(1, int(image.height * ratio))))
        gps, exif_ts = read_exif(image)
        started = time.perf_counter()
        detections, tiles = _detect(detector, image, tile=tile_config)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        record = record_for(image=name, source=source_label, image_sha256=digest,
                            index=result.images, frame_index=frame_index, ts=exif_ts, gps=gps,
                            model=model_info, detections=detections, tiles=tiles,
                            elapsed_ms=elapsed_ms, width=image.width, height=image.height)
        if snapshots_enabled and detections:
            snapshot_dir.mkdir(parents=True, exist_ok=True)
            draw_detections(image, detections).save(snapshot_dir / f"{Path(name).stem}.jpg", quality=85)
        _emit(jsonl, csv, record, result)
        state.mark(image_sha256=digest, source=resume_key, detections=len(detections))
        save_state(state)
        if progress is not None:
            progress(result.images, result.skipped, name)

    with JsonlWriter(jsonl_path) as jsonl, CsvWriter(csv_path) as csv:
        for item in sources_mod.iter_source(source, config=config, limit=None):
            kind = item["kind"]
            if kind == "image":
                path = Path(item["path"])
                digest = sha256_file(path)
                if use_resume and state.has(image_sha256=digest, source=str(path)):
                    result.skipped += 1
                    continue
                try:
                    with Image.open(path) as handle:
                        image = handle.convert("RGB")
                    _handle(image, name=path.name, source_label=str(path), resume_key=str(path),
                            digest=digest, frame_index=None, tile_config=tile)
                except Exception as exc:  # noqa: BLE001 - 单张失败不中断整轮
                    _note_error(str(path), exc)
                if limit and result.images + result.skipped >= limit:
                    break
                continue
            # 视频 / 视频流：按 fps 抽帧
            produced = 0
            try:
                for frame_index, _timestamp, frame in sources_mod.iter_video_frames(
                        str(item["path"]), fps=config.video_fps, config=config):
                    digest = _sha256_bytes(frame.tobytes())
                    resume_key = f"{item['path']}#{frame_index}"
                    if use_resume and state.has(image_sha256=digest, source=resume_key):
                        result.skipped += 1
                        continue
                    name = f"{Path(str(item['path'])).stem}_frame{frame_index:06d}.jpg"
                    _handle(frame, name=name, source_label=str(item["path"]), resume_key=resume_key,
                            digest=digest, frame_index=frame_index, tile_config=tile)
                    produced += 1
                    if limit and result.images + result.skipped >= limit:
                        break
            except Exception as exc:  # noqa: BLE001 - 视频/流不可用时不要吞掉整轮
                _note_error(str(item["path"]), exc)
            if limit and result.images + result.skipped >= limit:
                break
            if kind == "video" and produced == 0 and not result.errors:
                _note_error(str(item["path"]), RuntimeError("视频未产出任何帧"))
    save_state(state)
    result.cumulative_images = state.records
    result.cumulative_detections = state.detections
    result.finished_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return result


def _emit(jsonl: JsonlWriter, csv: CsvWriter, record: dict[str, Any], result: InferResult) -> None:
    jsonl.write(record)
    csv.write(record)
    result.images += 1
    result.detections += len(record["detections"])
    result.elapsed_ms += float(record["elapsed_ms"])
    for item in record["detections"]:
        code = str(item["class"])
        result.per_class[code] = result.per_class.get(code, 0) + 1


def benchmark(package_dir: str | Path, images: Sequence[Path], *, config: sources_mod.EdgeConfig,
              repeat: int = 1, warmup: int | None = None, providers: Sequence[str] | None = None,
              expect_labels: Sequence[str] | None = None,
              detector_factory: Callable[[PackageInfo], Any] | None = None,
              session_factory: Callable[[str], Any] | None = None) -> dict[str, Any]:
    """性能基准：warmup 后测 p50/p95 时延、FPS、RSS 峰值（对应 docs/08 §2.1 的性能目标）。"""
    import resource  # noqa: PLC0415 - 仅 Unix；无该模块时跳过内存统计

    package = validate_package(package_dir, expect_labels=expect_labels, session_factory=session_factory)
    detector = (detector_factory(package) if detector_factory else
                build_detector(package, threads=config.threads, conf=config.conf, iou=config.iou,
                               imgsz=config.imgsz, max_detections=config.max_detections,
                               providers=providers))
    warmup_runs = config.warmup_runs if warmup is None else int(warmup)
    frames: list[Image.Image] = []
    for path in images:
        with Image.open(path) as handle:
            frames.append(handle.convert("RGB"))
    if not frames:
        raise ValueError("基准测试需要至少一张图")
    for index in range(max(0, warmup_runs)):                       # warmup 不计入统计
        _detect(detector, frames[index % len(frames)], tile=config.tile)

    latencies: list[float] = []
    detections = 0
    started_all = time.perf_counter()
    for _ in range(max(1, int(repeat))):
        for frame in frames:
            started = time.perf_counter()
            found, _tiles = _detect(detector, frame, tile=config.tile)
            latencies.append((time.perf_counter() - started) * 1000.0)
            detections += len(found)
    total_ms = (time.perf_counter() - started_all) * 1000.0
    latencies.sort()

    def _quantile(fraction: float) -> float:
        index = min(len(latencies) - 1, max(0, int(round(fraction * (len(latencies) - 1)))))
        return round(latencies[index], 3)

    rss_mb = None
    try:
        rss_mb = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1)
    except Exception:  # noqa: BLE001 - 非 Unix 平台
        rss_mb = None
    return {
        "package": package.as_dict(), "images": len(latencies), "repeat": max(1, int(repeat)),
        "warmup": warmup_runs, "tile": {"enabled": config.tile.enabled, "size": config.tile.size},
        "imgsz": package.imgsz, "detector_imgsz": int(getattr(detector, "imgsz", package.imgsz)),
        "threads": config.threads, "detections": detections,
        "latency_ms": {"mean": round(sum(latencies) / len(latencies), 3), "p50": _quantile(0.5),
                       "p95": _quantile(0.95), "max": round(latencies[-1], 3)},
        "fps": round(len(latencies) / (total_ms / 1000.0), 3),
        "rss_mb": rss_mb,
        "backend": "onnxruntime", "measured_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
