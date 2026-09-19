"""边缘运行配置（configs/edge.yaml）与输入源枚举 / 帧读取（M4）。

边缘端可能没有数据库、没有 ffmpeg，因此：

* 配置**可选**：没有 `edge.yaml` 时用内置默认值，命令行参数优先；
* 图像用 Pillow 读；视频/RSTP 优先用 OpenCV，没有 OpenCV 时退回 ffmpeg 管道（两者都没有则明确报错）；
* 文件枚举按扩展名白名单递归，跳过隐藏文件与状态文件，顺序稳定（按路径排序）以便断点续跑可复现。
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Sequence

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")
VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv", ".m4v")


class SourceError(RuntimeError):
    """输入源不可用（路径不存在、类型不支持、缺解码器）。"""


@dataclass
class TileSpec:
    enabled: bool = False
    size: int = 1024
    overlap: float = 0.2
    merge_iou: float = 0.5


@dataclass
class EdgeConfig:
    """边缘推理配置（未提供的键用默认值；CLI 参数最终覆盖）。"""

    package_dir: str | None = None
    device: str = "cpu"
    precision: str = "fp32"
    threads: int = 4
    warmup_runs: int = 2
    imgsz: int | None = None          # None → 用导出包 preprocess.input_size
    conf: float | None = None
    iou: float | None = None
    max_detections: int | None = None
    tile: TileSpec = field(default_factory=TileSpec)
    image_exts: tuple[str, ...] = IMAGE_EXTS
    video_fps: float = 2.0
    rtsp_timeout_seconds: float = 10.0
    jsonl_name: str = "results.jsonl"
    csv_name: str = "results.csv"
    save_hit_snapshots: bool = False
    snapshot_dir: str = "snaps"
    resume_enabled: bool = True
    state_name: str = ".infer-state.json"
    max_image_side: int = 8192
    min_free_disk_mb: int = 500
    source: str | None = None

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "EdgeConfig":
        model = raw.get("model") or {}
        runtime = raw.get("runtime") or {}
        inference = raw.get("inference") or {}
        tile = inference.get("tile") or {}
        inputs = raw.get("input") or {}
        output = raw.get("output") or {}
        resume = raw.get("resume") or {}
        limits = raw.get("limits") or {}
        return cls(
            package_dir=model.get("package_dir"),
            device=str(model.get("device", "cpu")),
            precision=str(model.get("precision", "fp32")),
            threads=int(runtime.get("threads", 4)),
            warmup_runs=int(runtime.get("warmup_runs", 2)),
            imgsz=int(inference["imgsz"]) if inference.get("imgsz") else None,
            conf=float(inference["conf"]) if inference.get("conf") is not None else None,
            iou=float(inference["iou"]) if inference.get("iou") is not None else None,
            max_detections=(int(inference["max_detections"]) if inference.get("max_detections") else None),
            tile=TileSpec(enabled=bool(tile.get("enabled", False)), size=int(tile.get("size", 1024)),
                          overlap=float(tile.get("overlap", 0.2)),
                          merge_iou=float(tile.get("merge_iou", 0.5))),
            image_exts=tuple(str(ext).lower() for ext in (inputs.get("image_exts") or IMAGE_EXTS)),
            video_fps=float(inputs.get("video_fps", 2.0)),
            rtsp_timeout_seconds=float(inputs.get("rtsp_timeout_seconds", 10.0)),
            jsonl_name=str(output.get("jsonl", "results.jsonl")),
            csv_name=str(output.get("csv", "results.csv")),
            save_hit_snapshots=bool(output.get("save_hit_snapshots", False)),
            snapshot_dir=str(output.get("snapshot_dir", "snaps")),
            resume_enabled=bool(resume.get("enabled", True)),
            state_name=str(resume.get("state_file", ".infer-state.json")),
            max_image_side=int(limits.get("max_image_side", 8192)),
            min_free_disk_mb=int(limits.get("min_free_disk_mb", 500)),
        )

    @classmethod
    def load(cls, path: str | Path | None) -> "EdgeConfig":
        """读取 YAML 配置；文件不存在时返回默认值（边缘端允许"零配置文件"运行）。"""
        if not path:
            return cls()
        candidate = Path(path)
        if not candidate.exists():
            return cls()
        import yaml  # noqa: PLC0415 - pyyaml 是 rdinspect 的基础依赖

        raw = yaml.safe_load(candidate.read_text(encoding="utf-8")) or {}
        return cls.from_dict(raw if isinstance(raw, dict) else {})


# ─────────────────────────── 输入源 ───────────────────────────
def is_stream(source: str) -> bool:
    lowered = source.lower()
    return lowered.startswith(("rtsp://", "rtmp://", "http://", "https://", "udp://"))


def list_images(root: Path, *, exts: Sequence[str] = IMAGE_EXTS) -> list[Path]:
    """递归枚举图像（稳定排序、跳过隐藏文件）。"""
    allowed = {ext.lower() for ext in exts}
    files = [path for path in root.rglob("*")
             if path.is_file() and path.suffix.lower() in allowed and not path.name.startswith(".")]
    return sorted(files, key=lambda path: path.as_posix())


def iter_source(source: str, *, config: EdgeConfig, limit: int | None = None) -> Iterator[dict[str, Any]]:
    """把输入源展开成 `{kind, path, index, frame_index, ts}` 序列。

    * 目录 → 递归图像；
    * 单文件 → 图像（或视频，按扩展名判断）；
    * `rtsp://…` / `rtmp://…` / `http(s)://…` → 视频流（抽帧交给 :func:`iter_video_frames`）。
    """
    if is_stream(source):
        # 流是"无限"输入：这里只声明一条 stream 项，实际抽帧由主循环按 limit / 断点控制
        yield {"kind": "stream", "path": source, "index": 0, "frame_index": None, "ts": None}
        return

    path = Path(source).expanduser()
    if not path.exists():
        raise SourceError(f"输入源不存在: {path}")
    if path.is_dir():
        images = list_images(path, exts=config.image_exts)
        if not images:
            raise SourceError(f"目录里没有支持的图像（{','.join(config.image_exts)}）: {path}")
        for index, image in enumerate(images[:limit] if limit else images):
            yield {"kind": "image", "path": str(image), "index": index, "frame_index": None, "ts": None}
        return
    suffix = path.suffix.lower()
    if suffix in config.image_exts:
        yield {"kind": "image", "path": str(path), "index": 0, "frame_index": None, "ts": None}
        return
    if suffix in VIDEO_EXTS:
        yield {"kind": "video", "path": str(path), "index": 0, "frame_index": None, "ts": None}
        return
    raise SourceError(f"不支持的文件类型: {suffix}（图像 {','.join(config.image_exts)}；"
                      f"视频 {','.join(VIDEO_EXTS)}）")


def _cv2() -> Any | None:
    try:
        import cv2  # noqa: PLC0415

        return cv2
    except ImportError:
        return None


def video_backend() -> str:
    """可用的视频解码后端：`opencv` | `ffmpeg` | `none`。"""
    if _cv2() is not None:
        return "opencv"
    if shutil.which("ffmpeg"):
        return "ffmpeg"
    return "none"


def iter_video_frames(source: str, *, fps: float, config: EdgeConfig) -> Iterator[tuple[int, float, Any]]:
    """按 fps 抽帧，产出 `(frame_index, 时间戳秒, PIL.Image)`。

    无 OpenCV 时退回 `ffmpeg -i <src> -vf fps=<fps> -f image2pipe -vcodec png -`，
    因此边缘最小依赖里 OpenCV 是可选的。
    """
    backend = video_backend()
    if backend == "none":
        raise SourceError("视频/流输入需要 OpenCV 或 ffmpeg：pip install opencv-python-headless 或安装 ffmpeg")
    if backend == "opencv":
        yield from _iter_video_opencv(source, fps=fps, config=config)
    else:  # pragma: no cover - 依赖系统 ffmpeg
        yield from _iter_video_ffmpeg(source, fps=fps)


def _iter_video_opencv(source: str, *, fps: float, config: EdgeConfig) -> Iterator[tuple[int, float, Any]]:
    import time  # noqa: PLC0415

    from PIL import Image  # noqa: PLC0415

    cv2 = _cv2()
    if is_stream(source):
        capture = cv2.VideoCapture(source)
        if not capture.isOpened():
            raise SourceError(f"无法打开视频流: {source}")
    else:
        capture = cv2.VideoCapture(source)
        if not capture.isOpened():
            raise SourceError(f"无法打开视频文件: {source}")
    native_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0) or 25.0
    step = max(1, int(round(native_fps / fps))) if fps > 0 else 1
    index = 0
    kept = 0
    deadline = time.monotonic() + config.rtsp_timeout_seconds
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                if is_stream(source) and time.monotonic() < deadline:
                    continue
                break
            if index % step == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                yield kept, index / native_fps, Image.fromarray(rgb)
                kept += 1
            index += 1
    finally:
        capture.release()


def _iter_video_ffmpeg(source: str, *, fps: float) -> Iterator[tuple[int, float, Any]]:  # pragma: no cover
    import io  # noqa: PLC0415

    from PIL import Image  # noqa: PLC0415

    command = ["ffmpeg", "-loglevel", "error", "-i", source, "-vf", f"fps={fps}",
               "-f", "image2pipe", "-vcodec", "png", "-"]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    assert process.stdout is not None
    buffer = b""
    index = 0
    try:
        while True:
            chunk = process.stdout.read(65536)
            if not chunk:
                break
            buffer += chunk
            while True:
                start = buffer.find(b"\x89PNG")
                if start < 0:
                    break
                end = buffer.find(b"IEND\xaeB`\x82", start)
                if end < 0:
                    break
                blob = buffer[start:end + 8]
                buffer = buffer[end + 8:]
                yield index, index / fps, Image.open(io.BytesIO(blob)).convert("RGB")
                index += 1
    finally:
        process.terminate()


def check_disk_space(path: Path, *, min_free_mb: int) -> tuple[bool, int]:
    """磁盘余量检查：返回 (是否充足, 剩余 MB)。"""
    target = path if path.exists() else path.parent
    usage = shutil.disk_usage(target)
    free_mb = int(usage.free / (1024 * 1024))
    return free_mb >= int(min_free_mb), free_mb
