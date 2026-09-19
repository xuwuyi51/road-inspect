#!/usr/bin/env python3
"""生成合成巡查素材：路面照片（含缺陷）+ 一段短视频。

用途：M1 验收与回归（磁盘仅 27GB，用合成数据避免下载公开数据集）。
用法：
  python3 scripts/make_sample_data.py --out /tmp/rd-sample --count 500 --video-seconds 6
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

CLASSES = ("longitudinal_crack", "transverse_crack", "pothole", "garbage")


def _asphalt(width: int, height: int, rng: random.Random) -> Image.Image:
    """合成沥青路面底图：基础灰 + 颗粒噪声 + 轻微光照渐变。"""
    base = Image.new("RGB", (width, height), (78, 78, 82))
    pixels = base.load()
    for y in range(height):
        light = 1.0 + 0.25 * (y / height - 0.5)  # 上暗下亮的透视光照
        for x in range(0, width, 2):
            noise = rng.randint(-14, 14)
            value = int(max(0, min(255, (74 + noise) * light)))
            pixels[x, y] = (value, value, min(255, value + 3))
            if x + 1 < width:
                pixels[x + 1, y] = (value, value, min(255, value + 3))
    return base.filter(ImageFilter.GaussianBlur(0.6))


def _draw_defects(image: Image.Image, rng: random.Random, labels: list[dict]) -> None:
    draw = ImageDraw.Draw(image)
    width, height = image.size

    def add(label: str, x1: int, y1: int, x2: int, y2: int) -> None:
        labels.append({
            "class_code": label,
            "bbox": [round(x1 / width, 6), round(y1 / height, 6),
                     round(x2 / width, 6), round(y2 / height, 6)],
        })

    kinds = rng.sample(CLASSES, k=rng.choice([1, 1, 2, 3]))
    for kind in kinds:
        if kind == "longitudinal_crack":
            x = rng.randint(int(width * 0.15), int(width * 0.85))
            y1 = rng.randint(0, int(height * 0.2))
            y2 = rng.randint(int(height * 0.6), height - 1)
            w = rng.randint(2, max(3, width // 90))
            points = [(x + rng.randint(-4, 4), y) for y in range(y1, y2, 6)]
            draw.line(points, fill=(28, 28, 30), width=w)
            add(kind, x - w * 2, y1, x + w * 2 + 2, y2)
        elif kind == "transverse_crack":
            y = rng.randint(int(height * 0.2), int(height * 0.8))
            x1 = rng.randint(0, int(width * 0.2))
            x2 = rng.randint(int(width * 0.6), width - 1)
            w = rng.randint(2, max(3, height // 90))
            points = [(x, y + rng.randint(-3, 3)) for x in range(x1, x2, 6)]
            draw.line(points, fill=(30, 30, 32), width=w)
            add(kind, x1, y - w * 2, x2, y + w * 2 + 2)
        elif kind == "pothole":
            w = rng.randint(int(width * 0.08), int(width * 0.2))
            h = rng.randint(int(w * 0.6), int(w * 1.1))
            cx = rng.randint(w, width - w)
            cy = rng.randint(h, height - h)
            draw.ellipse([cx - w // 2, cy - h // 2, cx + w // 2, cy + h // 2], fill=(45, 43, 45))
            draw.ellipse([cx - w // 3, cy - h // 3, cx + w // 3, cy + h // 3], fill=(28, 27, 29))
            add(kind, cx - w // 2, cy - h // 2, cx + w // 2, cy + h // 2)
        else:  # garbage
            w = rng.randint(int(width * 0.08), int(width * 0.22))
            h = rng.randint(int(width * 0.06), int(width * 0.18))
            x = rng.randint(0, max(1, width - w))
            y = rng.randint(0, max(1, height - h))
            color = rng.choice([(196, 188, 160), (170, 90, 70), (120, 150, 120), (200, 200, 210)])
            draw.rectangle([x, y, x + w, y + h], fill=color)
            draw.line([x, y, x + w, y + h], fill=(90, 90, 90), width=1)
            add(kind, x, y, x + w, y + h)


def _exif_bytes(rng: random.Random, timestamp: str) -> bytes | None:
    """构造含拍摄时间/GPS/设备的最小 EXIF。

    注意：GPS 的度分秒必须写成 IFDRational，直接用 (num, den) 元组会让 Pillow 序列化报错。
    """
    try:
        from PIL import ExifTags
        from PIL.TiffImagePlugin import IFDRational

        exif = Image.Exif()
        exif[ExifTags.Base.Make] = "SyntheticCam"
        exif[ExifTags.Base.Model] = "RoadSim 1.0"
        exif[0x9003] = timestamp  # DateTimeOriginal
        lat = 31.20 + rng.random() * 0.05
        lon = 121.40 + rng.random() * 0.05
        gps = exif.get_ifd(ExifTags.IFD.GPSInfo)
        gps[1], gps[3] = "N", "E"
        gps[2] = (IFDRational(int(lat), 1), IFDRational(int(lat % 1 * 60), 1),
                  IFDRational(int(lat * 3600 % 60), 1))
        gps[4] = (IFDRational(int(lon), 1), IFDRational(int(lon % 1 * 60), 1),
                  IFDRational(int(lon * 3600 % 60), 1))
        return exif.tobytes()
    except Exception as exc:  # 生成器不因 EXIF 失败而中断
        print(f"[make_sample_data] EXIF 生成失败（该图将无 EXIF）: {exc}")
        return None


def generate(out_dir: Path, count: int, *, width: int = 960, height: int = 540,
             seed: int = 7, video_seconds: float = 6.0, video_fps: float = 24.0) -> dict:
    rng = random.Random(seed)
    photos_dir = out_dir / "inbox" / "photos"
    frames_dir = out_dir / "video_frames"
    photos_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)

    labels: list[dict] = []
    for index in range(count):
        image = _asphalt(width, height, rng)
        frame_labels: list[dict] = []
        _draw_defects(image, rng, frame_labels)
        stamp = f"2026:09:{10 + index % 4:02d} {8 + index % 10:02d}:{index % 60:02d}:{index % 60:02d}"
        path = photos_dir / f"sim_{index:05d}.jpg"
        exif = _exif_bytes(rng, stamp)
        save_kwargs = {"format": "JPEG", "quality": 88}
        if exif is not None:
            save_kwargs["exif"] = exif
        image.save(path, **save_kwargs)
        labels.append({"file": path.name, "annotations": frame_labels})

    (out_dir / "labels.json").write_text(json.dumps(labels, ensure_ascii=False, indent=1), encoding="utf-8")

    # 短视频：由连续帧合成（抽帧后可验证 video_frame 导入路径）
    total_frames = max(4, int(video_seconds * video_fps))
    for index in range(total_frames):
        image = _asphalt(width // 2, height // 2, rng)
        _draw_defects(image, rng, [])
        image.save(frames_dir / f"f{index:05d}.jpg", format="JPEG", quality=85)
    video_path = out_dir / "inbox" / "patrol.mp4"
    if shutil.which("ffmpeg"):
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-framerate", str(video_fps),
             "-i", str(frames_dir / "f%05d.jpg"), "-c:v", "libx264", "-pix_fmt", "yuv420p",
             "-t", str(video_seconds), str(video_path)],
            check=True, capture_output=True,
        )
        shutil.rmtree(frames_dir, ignore_errors=True)
    return {"photos": count, "video": str(video_path) if video_path.exists() else None,
            "labels": str(out_dir / "labels.json")}


def main() -> int:
    parser = argparse.ArgumentParser(description="生成合成巡查素材（照片 + 短视频）")
    parser.add_argument("--out", required=True)
    parser.add_argument("--count", type=int, default=500)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--video-seconds", type=float, default=6.0)
    parser.add_argument("--video-fps", type=float, default=24.0)
    args = parser.parse_args()
    out_dir = Path(args.out)
    result = generate(out_dir, args.count, width=args.width, height=args.height, seed=args.seed,
                      video_seconds=args.video_seconds, video_fps=args.video_fps)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
