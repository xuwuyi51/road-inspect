"""导入管线：照片 / 视频抽帧 / 航拍切片 → 去重 → 入库 → 生成标注任务。

设计要点（docs/03-data-model.md §6、ADR-0004）：
  * 幂等：文件 sha256 为幂等键，重复文件记为 dup_sha 不重复入库；
  * 近似重复：pHash 汉明距离 ≤ 阈值记为 dup_phash，仍入库但降低任务优先级（避免漏检）；
  * 原图只写一次（内容寻址），缩略图可重建；
  * 视频抽帧调用系统 ffmpeg；抽帧失败的视频不阻塞其他文件。
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

from ..config import IMAGE_EXTS, VIDEO_EXTS, Config, ConfigError
from ..storage.db import transaction
from ..storage.files import atomic_write_bytes, content_path, make_thumbnail, sha256_file, utc_now
from ..storage.repo import Repo
from .hashing import find_near_duplicate, phash_file
from .images import iter_tiles, quality_metrics, read_meta, save_jpeg

#: 任务优先级：数字越小越先被标注；近似重复排到最后
PRIORITY_NORMAL = 100
PRIORITY_NEAR_DUPLICATE = 300


@dataclass
class IngestReport:
    batch_id: int
    kind: str
    added: int = 0
    dup_sha: int = 0
    dup_phash: int = 0
    skipped_type: int = 0
    skipped_size: int = 0
    error: int = 0
    tiles: int = 0
    image_ids: list[int] = field(default_factory=list)
    errors: list[dict[str, str]] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return {
            "batch_id": self.batch_id,
            "kind": self.kind,
            "added": self.added,
            "dup_sha": self.dup_sha,
            "dup_phash": self.dup_phash,
            "tiles": self.tiles,
            "skipped_type": self.skipped_type,
            "skipped_size": self.skipped_size,
            "error": self.error,
            "image_ids": self.image_ids,
            "errors": self.errors[:20],
        }


def _iter_files(source: Path, extensions: set[str]) -> list[Path]:
    if source.is_file():
        return [source] if source.suffix.lower() in extensions else []
    files: list[Path] = []
    for path in sorted(source.rglob("*")):
        if path.is_file() and not path.name.startswith(".") and path.suffix.lower() in extensions:
            files.append(path)
    return files


def _insert_image_with_dedup(config: Config, repo: Repo, report: IngestReport, *, batch_id: int,
                             file_path: Path, candidates: list[tuple[int, str]],
                             source_kind: str, parent_image_id: int | None = None,
                             tile_json: str | None = None, stamp: str | None = None,
                             captured_at: str | None = None, origin_path: Path | None = None,
                             make_thumb: bool = True) -> int | None:
    """单文件入库（含去重、缩略图、任务创建）；返回 image_id 或 None。"""
    size_mb = file_path.stat().st_size / 1048576
    if size_mb > config.ingest.max_file_mb:
        report.skipped_size += 1
        repo.audit_ingest_items(batch_id, str(file_path), "skipped_size",
                                f"{size_mb:.1f}MB > {config.ingest.max_file_mb}MB")
        return None

    digest = sha256_file(file_path)
    existing = repo.find_image_by_sha(digest)
    if existing is not None:
        report.dup_sha += 1
        repo.audit_ingest_items(batch_id, str(file_path), "dup_sha", "sha256 已存在", existing["id"])
        return None

    try:
        meta = read_meta(file_path)
        digest_phash = phash_file(file_path)
    except Exception as exc:  # 损坏/非图片
        report.error += 1
        report.errors.append({"path": str(file_path), "error": str(exc)})
        repo.audit_ingest_items(batch_id, str(file_path), "error", str(exc)[:200])
        return None

    near_dup_id = find_near_duplicate(digest_phash, candidates, config.ingest.dedup.phash_hamming_threshold)
    keep_near = config.ingest.dedup.keep_near_duplicates
    if near_dup_id is not None and not keep_near:
        report.dup_phash += 1
        repo.audit_ingest_items(batch_id, str(file_path), "dup_phash", f"近似重复 #{near_dup_id}", near_dup_id)
        return None
    if near_dup_id is not None:
        report.dup_phash += 1
        repo.audit_ingest_items(batch_id, str(file_path), "dup_phash", f"近似重复 #{near_dup_id}", near_dup_id)

    stored = content_path(config, digest, file_path.suffix or ".jpg",
                          kind="tile" if source_kind == "aerial_tile" else "frame" if source_kind == "video_frame" else "photo",
                          stamp=stamp)
    if not stored.exists():
        stored.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_bytes(stored, file_path.read_bytes())

    quality = {}
    try:
        quality = quality_metrics(stored)
    except Exception:
        quality = {}

    image_id = repo.insert_image(
        batch_id=batch_id, path=config.rel_data_path(stored), sha256=digest, phash=digest_phash,
        width=meta.width, height=meta.height, bytes_=stored.stat().st_size, source_kind=source_kind,
        parent_image_id=parent_image_id, tile_json=tile_json,
        captured_at=captured_at or meta.captured_at, gps_lat=meta.gps_lat, gps_lon=meta.gps_lon,
        gps_source=meta.gps_source, device=meta.device,
        quality_json=json.dumps(quality, ensure_ascii=False) if quality else None,
    )
    if near_dup_id is not None:
        repo.conn.execute("UPDATE images SET duplicate_of = ? WHERE id = ?", (near_dup_id, image_id))

    repo.create_task(image_id, priority=PRIORITY_NEAR_DUPLICATE if near_dup_id is not None else PRIORITY_NORMAL)
    if make_thumb:
        try:
            make_thumbnail(config, stored, image_id)
        except Exception:
            pass
    candidates.append((image_id, digest_phash))
    report.added += 1
    report.image_ids.append(image_id)
    repo.audit_ingest_items(batch_id, str(file_path), "added", None, image_id)
    return image_id


def _import_photos(config: Config, repo: Repo, report: IngestReport, *, files: Sequence[Path],
                   batch_id: int, source_kind: str, tile: bool, stamp: str | None) -> None:
    candidates = repo.list_phashes()
    for file_path in files:
        image_id = _insert_image_with_dedup(
            config, repo, report, batch_id=batch_id, file_path=file_path,
            candidates=candidates, source_kind=source_kind, stamp=stamp,
        )
        if image_id is None or not tile:
            continue
        # 航拍/大图切片：切片作为独立影像入库（parent 指回原图），便于按 tile 标注与训练
        origin = repo.get_image(image_id)
        if origin is None:
            continue
        try:
            for index, (tile_image, tile_meta) in enumerate(
                    iter_tiles(config.abs_data_path(origin["path"]), config.ingest.tile.size,
                               config.ingest.tile.overlap)):
                fd = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False, dir=str(config.tiles_dir))
                fd.close()
                tile_path = Path(fd.name)
                save_jpeg(tile_image, tile_path)
                tile_id = _insert_image_with_dedup(
                    config, repo, report, batch_id=batch_id, file_path=tile_path,
                    candidates=candidates, source_kind="aerial_tile",
                    parent_image_id=image_id, tile_json=json.dumps(tile_meta, ensure_ascii=False),
                    stamp=str(image_id), captured_at=origin["captured_at"], make_thumb=False,
                )
                tile_path.unlink(missing_ok=True)
                if tile_id is not None:
                    report.tiles += 1
                if index >= 4096:  # 防御：异常参数导致切片爆炸
                    break
        except Exception as exc:
            report.errors.append({"path": origin["path"], "error": f"切片失败: {exc}"})


def _import_video(config: Config, repo: Repo, report: IngestReport, *, video: Path, batch_id: int,
                  fps: float, max_frames: int) -> None:
    if shutil.which("ffmpeg") is None:
        report.error += 1
        report.errors.append({"path": str(video), "error": "未找到 ffmpeg，无法抽帧"})
        return
    with tempfile.TemporaryDirectory(prefix="rdinspect-frames-") as tmp_dir:
        pattern = str(Path(tmp_dir) / "frame-%06d.jpg")
        command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(video),
                   "-vf", f"fps={fps}", "-frames:v", str(max_frames), "-q:v", "2", pattern]
        result = subprocess.run(command, capture_output=True, text=True, timeout=3600)
        frames = sorted(Path(tmp_dir).glob("frame-*.jpg"))
        if result.returncode != 0 or not frames:
            report.error += 1
            report.errors.append({"path": str(video),
                                  "error": (result.stderr or "抽帧失败").strip()[:200]})
            return
        candidates = repo.list_phashes()
        # 帧时间戳与 GPS 轨迹对齐在 M1+ 引入（见 docs/03-data-model.md §3）
        for frame in frames:
            _insert_image_with_dedup(
                config, repo, report, batch_id=batch_id, file_path=frame, candidates=candidates,
                source_kind="video_frame", stamp=video.stem, captured_at=None,
            )


def import_path(config: Config, repo: Repo, source: str | Path, *, kind: str | None = None,
                fps: float | None = None, max_frames: int | None = None, tile: bool | None = None,
                note: str | None = None, actor: str = "cli", limit: int | None = None) -> IngestReport:
    """导入目录或单个文件。

    @param kind - photo | video | aerial | external；None 时按扩展名自动判定
    @param tile - 是否切片（默认取配置；aerial 时默认强制开启）
    """
    resolved = config.resolve_source(source)
    images = _iter_files(resolved, IMAGE_EXTS)
    videos = _iter_files(resolved, VIDEO_EXTS)
    if kind is None:
        kind = "video" if (not images and videos) else "photo"
    if kind not in ("photo", "video", "aerial", "external"):
        raise ConfigError(f"未知导入类型: {kind}")
    if limit is not None:
        images = images[:limit]
        videos = videos[:limit]
    if not images and not videos:
        raise ConfigError(f"导入源中没有可处理的图片/视频: {resolved}")

    report = IngestReport(batch_id=0, kind=kind)
    batch_id = repo.create_batch(kind, source=str(resolved), note=note, created_by=actor)
    report.batch_id = batch_id

    use_tile = tile if tile is not None else (config.ingest.tile.enabled or kind == "aerial")
    stamp = utc_now()[:10].replace("-", "")
    # 元数据在一个事务中提交（文件已先落盘；事务失败只会留下未被引用的文件，可清理）
    with transaction(repo.conn):
        if images:
            _import_photos(config, repo, report, files=images, batch_id=batch_id,
                           source_kind="aerial" if kind == "aerial" else "photo",
                           tile=use_tile, stamp=stamp)
        for video in videos:
            _import_video(config, repo, report, video=video, batch_id=batch_id,
                          fps=fps or config.ingest.video.default_fps,
                          max_frames=max_frames or config.ingest.video.max_frames_per_video)
        repo.update_batch_stats(batch_id, {
            "added": report.added, "dup_sha": report.dup_sha, "dup_phash": report.dup_phash,
            "tiles": report.tiles, "skipped_size": report.skipped_size,
            "skipped_type": report.skipped_type, "error": report.error,
        })
    return report
