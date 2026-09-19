"""文件仓储：原子写、缩略图与内容寻址路径。

所有落盘文件都写「临时文件 + os.replace」；路径一律相对 data_dir 存库。
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageOps

from ..config import Config

JPEG_QUALITY = 92


def utc_now() -> str:
    """数据库统一使用的 UTC ISO8601 文本。"""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_bytes(target: Path, data: bytes) -> Path:
    """原子写入字节（同分区 rename）。"""
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(target.parent), prefix=".tmp-", suffix=target.suffix)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        os.replace(tmp_name, target)
    except Exception:
        Path(tmp_name).unlink(missing_ok=True)
        raise
    return target


def content_path(config: Config, sha256: str, extension: str, kind: str = "photo",
                 stamp: str | None = None) -> Path:
    """内容寻址路径：raw/<日期>/<sha256>.jpg 或 frames/<video>/<ts>.jpg。"""
    extension = extension if extension.startswith(".") else f".{extension}"
    if kind == "frame":
        base = config.frames_dir / (stamp or "video")
    elif kind == "tile":
        base = config.tiles_dir / (stamp or "aerial")
    else:
        base = config.raw_dir / (stamp or datetime.now(timezone.utc).strftime("%Y%m%d"))
    return base / f"{sha256}{extension.lower()}"


def make_thumbnail(config: Config, source: Path, image_id: int) -> Path:
    """生成/覆盖长边缩略图（WebP），返回绝对路径。"""
    target = config.thumbs_dir / f"{image_id}.webp"
    with Image.open(source) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        image.thumbnail((config.thumb_long_edge, config.thumb_long_edge), Image.LANCZOS)
        with tempfile.SpooledTemporaryFile(max_size=4 << 20) as buffer:
            image.save(buffer, format="WEBP", quality=82, method=4)
            buffer.seek(0)
            atomic_write_bytes(target, buffer.read())
    return target


def copy_into_store(config: Config, source: Path, sha256: str, kind: str = "photo",
                    stamp: str | None = None) -> Path:
    """把原始文件复制进内容寻址目录（同内容只存一份）。"""
    target = content_path(config, sha256, source.suffix or ".jpg", kind=kind, stamp=stamp)
    if target.exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(target.parent), prefix=".tmp-", suffix=target.suffix)
    os.close(fd)
    try:
        shutil.copy2(source, tmp_name)
        os.replace(tmp_name, target)
    except Exception:
        Path(tmp_name).unlink(missing_ok=True)
        raise
    return target


def rebuild_thumbnails(config: Config, rows: list[tuple[int, str]]) -> int:
    """按 (image_id, rel_path) 重建缩略图，返回成功数（缩略图可随时重建）。"""
    done = 0
    for image_id, rel_path in rows:
        try:
            make_thumbnail(config, config.abs_data_path(rel_path), image_id)
            done += 1
        except Exception:
            continue
    return done
