"""导入管线测试：去重、EXIF、缩略图、任务生成、路径白名单、视频抽帧。"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from rdinspect.config import ConfigError
from rdinspect.core.hashing import find_near_duplicate, hamming_hex, phash_file
from rdinspect.core.images import iter_tiles, quality_metrics, read_meta
from rdinspect.core.ingest import PRIORITY_NEAR_DUPLICATE, import_path

from helpers import make_config, open_repo, synth_images


class TestHashing(unittest.TestCase):
    def test_phash_distance_and_near_duplicate(self) -> None:
        self.assertEqual(hamming_hex("0000000000000000", "0000000000000000"), 0)
        self.assertEqual(hamming_hex("0000000000000000", "0000000000000001"), 1)
        self.assertEqual(hamming_hex("0000000000000000", "ffffffffffffffff"), 64)
        candidates = [(1, "0000000000000000"), (2, "ffffffffffffffff")]
        self.assertEqual(find_near_duplicate("0000000000000001", candidates, threshold=6), 1)
        self.assertIsNone(find_near_duplicate("aaaaaaaaaaaaaaaa", candidates, threshold=6))


class TestImageMeta(unittest.TestCase):
    def test_read_meta_reports_size_and_quality(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config = make_config(tmp_path)
            path = synth_images(config.raw_dir / "probe", count=1, size=(64, 48))[0]
            meta = read_meta(path)
            self.assertEqual((meta.width, meta.height), (64, 48))
            self.assertGreater(meta.bytes, 0)
            metrics = quality_metrics(path)
            self.assertIn("sharpness", metrics)
            self.assertTrue(0.0 <= metrics["brightness"] <= 1.0)

    def test_iter_tiles_yields_windows_with_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config = make_config(tmp_path)
            path = synth_images(config.raw_dir / "tile", count=1, size=(200, 120))[0]
            tiles = list(iter_tiles(path, tile=100, overlap=0.0))
            self.assertEqual(len(tiles), 4)  # 2×2
            first_image, first_meta = tiles[0]
            self.assertEqual((first_image.width, first_image.height), (100, 100))  # 首片为完整 tile
            self.assertEqual(first_meta["origin_width"], 200)


class TestIngest(unittest.TestCase):
    def test_import_creates_images_tasks_thumbs_and_dedups(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config = make_config(tmp_path)
            _, repo = open_repo(config)
            source = config.data_dir / "inbox" / "photos"
            paths = synth_images(source, count=5)

            report = import_path(config, repo, source, kind="photo")
            self.assertEqual(report.added, 5)
            self.assertEqual(report.dup_sha, 0)
            self.assertEqual(repo.count_images(), 5)
            self.assertEqual(len(repo.list_tasks(status="pending", limit=10)), 5)
            for image_id in report.image_ids:
                self.assertTrue((config.thumbs_dir / f"{image_id}.webp").exists(), "缩略图缺失")

            # 再次导入：全部命中 sha256 幂等去重
            again = import_path(config, repo, source, kind="photo")
            self.assertEqual(again.added, 0)
            self.assertEqual(again.dup_sha, 5)
            self.assertEqual(repo.count_images(), 5)
            del paths

    def test_near_duplicate_is_kept_but_deprioritised(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config = make_config(tmp_path)
            _, repo = open_repo(config)
            source = config.data_dir / "inbox" / "dup"
            paths = synth_images(source, count=2, seed=3)

            # 复制第一张（内容完全相同 → sha 去重），再生成一张仅亮度微变的近似图
            from PIL import Image, ImageEnhance

            near = Image.open(paths[0])
            near = ImageEnhance.Brightness(near).enhance(1.02)
            near_path = source / "near.jpg"
            near.save(near_path, quality=90)

            report = import_path(config, repo, source, kind="photo")
            self.assertEqual(report.added, 3)
            self.assertEqual(report.dup_sha, 0)
            # 内容寻址存储不保留原文件名，按 duplicate_of 反查近似重复项
            near_row = repo.conn.execute(
                "SELECT id, duplicate_of FROM images WHERE duplicate_of IS NOT NULL").fetchone()
            self.assertIsNotNone(near_row, "近似重复应标记 duplicate_of")
            task = repo.conn.execute(
                "SELECT priority FROM tasks WHERE image_id = ?", (near_row["id"],)).fetchone()
            self.assertEqual(task["priority"], PRIORITY_NEAR_DUPLICATE)

    def test_source_path_outside_allowlist_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config = make_config(tmp_path)
            _, repo = open_repo(config)
            outside = tmp_path / "outside"
            synth_images(outside, count=1)
            with self.assertRaises(ConfigError):
                import_path(config, repo, outside, kind="photo")

    def test_low_quality_metrics_are_recorded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config = make_config(tmp_path)
            _, repo = open_repo(config)
            source = config.data_dir / "inbox" / "q"
            synth_images(source, count=1)
            report = import_path(config, repo, source, kind="photo")
            row = repo.get_image(report.image_ids[0])
            self.assertIsNotNone(row["quality_json"])
            self.assertIn("brightness", row["quality_json"])

    @unittest.skipIf(shutil.which("ffmpeg") is None, "缺少 ffmpeg")
    def test_video_frames_are_imported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config = make_config(tmp_path)
            _, repo = open_repo(config)
            frames_dir = tmp_path / "frames"
            paths = synth_images(frames_dir, count=8, size=(64, 48))
            video = config.data_dir / "inbox" / "clip.mp4"
            video.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-framerate", "8",
                 "-i", str(paths[0].parent / "img_%03d.jpg"), "-c:v", "libx264",
                 "-pix_fmt", "yuv420p", str(video)], check=True, capture_output=True)

            report = import_path(config, repo, video, kind="video", fps=2, max_frames=4)
            self.assertGreaterEqual(report.added, 1)
            kinds = {repo.get_image(image_id)["source_kind"] for image_id in report.image_ids}
            self.assertEqual(kinds, {"video_frame"})

    def test_phash_file_stable_for_same_content(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config = make_config(tmp_path)
            path = synth_images(config.raw_dir / "ph", count=1)[0]
            self.assertEqual(phash_file(path), phash_file(path))


if __name__ == "__main__":
    unittest.main()
