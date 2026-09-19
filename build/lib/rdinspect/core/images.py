"""影像元数据：EXIF（时间/GPS/设备）、尺寸、质量指标、切片。

注意：EXIF 时间是相机本地时间（无时区），按 `YYYY-MM-DDTHH:MM:SS` 原样存储，
不做时区猜测；只有系统产生的审计时间使用 UTC（带 Z）。
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from PIL import ExifTags, Image, ImageOps

from .geometry import tiled_origins


@dataclass
class ImageMeta:
    width: int
    height: int
    bytes: int
    captured_at: str | None = None
    gps_lat: float | None = None
    gps_lon: float | None = None
    device: str | None = None
    gps_source: str = "none"


def _ratio_to_float(value: Any) -> float | None:
    try:
        if isinstance(value, tuple) and len(value) == 2:
            num, den = value
            return float(num) / float(den) if den else None
        if isinstance(value, (int, float, Fraction)):
            return float(value)
        return float(value)
    except Exception:
        return None


def _dms_to_decimal(dms: Any, ref: Any) -> float | None:
    """EXIF GPS (度, 分, 秒) + 半球参考 → 十进制度。"""
    try:
        degrees = _ratio_to_float(dms[0])
        minutes = _ratio_to_float(dms[1])
        seconds = _ratio_to_float(dms[2])
        if degrees is None or minutes is None or seconds is None:
            return None
        value = degrees + minutes / 60 + seconds / 3600
        if str(ref).upper() in ("S", "W"):
            value = -value
        return round(value, 8)
    except Exception:
        return None


def read_meta(path: str | Path) -> ImageMeta:
    """读取尺寸/字节数/EXIF（时间、GPS、设备）。"""
    file_path = Path(path)
    with Image.open(file_path) as image:
        width, height = image.size
        exif_raw = image.getexif()
        captured_at = None
        device = None
        gps_lat = gps_lon = None
        if exif_raw:
            try:
                tags = {ExifTags.TAGS.get(k, k): v for k, v in exif_raw.items()}
                for key in ("DateTimeOriginal", "DateTimeDigitized", "DateTime"):
                    raw_time = tags.get(key)
                    if isinstance(raw_time, str) and len(raw_time) >= 19:
                        captured_at = raw_time.strip().replace(":", "-", 2).replace(" ", "T")[:19]
                        break
                make = str(tags.get("Make") or "").strip()
                model = str(tags.get("Model") or "").strip()
                device = " ".join(part for part in (make, model) if part) or None
            except Exception:
                pass
            gps_lat = gps_lon = None
            try:
                gps = exif_raw.get_ifd(ExifTags.IFD.GPSInfo) if hasattr(exif_raw, "get_ifd") else None
                if gps:
                    lat = _dms_to_decimal(gps.get(2), gps.get(1))
                    lon = _dms_to_decimal(gps.get(4), gps.get(3))
                    if lat is not None and lon is not None and (lat, lon) != (0.0, 0.0):
                        gps_lat, gps_lon = lat, lon
            except Exception:
                gps_lat = gps_lon = None
    return ImageMeta(
        width=width, height=height, bytes=file_path.stat().st_size,
        captured_at=captured_at, gps_lat=gps_lat, gps_lon=gps_lon, device=device,
        gps_source="exif" if gps_lat is not None else "none",
    )


def quality_metrics(path: str | Path, sample_long_edge: int = 512) -> dict[str, float]:
    """轻量质量指标：亮度均值/标准差、清晰度（拉普拉斯能量）与过曝/欠曝比例。

    用于标注台排序（模糊/过暗的图优先人工确认）与导入报告，不做质量淘汰判定。
    """
    with Image.open(path) as image:
        gray = ImageOps.exif_transpose(image).convert("L")
        gray.thumbnail((sample_long_edge, sample_long_edge), Image.LANCZOS)
        array = np.asarray(gray, dtype=np.float32) / 255.0
    if array.size == 0:
        return {"brightness": 0.0, "contrast": 0.0, "sharpness": 0.0, "overexposed": 0.0, "underexposed": 0.0}
    gy, gx = np.gradient(array)
    laplacian = np.abs(gy) + np.abs(gx)
    return {
        "brightness": round(float(array.mean()), 5),
        "contrast": round(float(array.std()), 5),
        "sharpness": round(float((laplacian ** 2).mean()), 6),
        "overexposed": round(float((array > 0.98).mean()), 5),
        "underexposed": round(float((array < 0.05).mean()), 5),
    }


def iter_tiles(path: str | Path, tile: int, overlap: float
               ) -> Iterator[tuple[Image.Image, dict[str, int | float]]]:
    """按切片参数产出 (tile 图像, tile_json)，坐标原点为原图左上角。"""
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        width, height = image.size
        for (x, y, w, h) in tiled_origins(width, height, tile, overlap):
            yield image.crop((x, y, x + w, y + h)), {"x": x, "y": y, "w": w, "h": h, "overlap": overlap,
                                                     "origin_width": width, "origin_height": height}


def save_jpeg(image: Image.Image, target: Path, quality: int = 92) -> None:
    """JPEG 原子保存（临时文件 + rename，由调用方保证目录存在）。"""
    import os
    import tempfile

    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(target.parent), prefix=".tmp-", suffix=target.suffix)
    os.close(fd)
    try:
        image.convert("RGB").save(tmp, format="JPEG", quality=quality)
        os.replace(tmp, target)
    except Exception:
        Path(tmp).unlink(missing_ok=True)
        raise
