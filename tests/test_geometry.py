"""几何换算与切片测试（对应 docs/format-samples 的坐标约定）。"""

from __future__ import annotations

import unittest

from rdinspect.core.geometry import (
    GeometryError,
    bbox_area,
    bbox_from_coco,
    bbox_from_labelme,
    bbox_from_xywh_norm,
    bbox_iou,
    bbox_to_coco,
    bbox_to_labelme,
    bbox_to_xywh_norm,
    class_index_map,
    parse_yolo_line,
    tile_to_orig,
    tiled_origins,
    validate_bbox,
    yolo_line,
)

CLASSES = [
    {"code": "longitudinal_crack", "order_index": 0},
    {"code": "transverse_crack", "order_index": 1},
    {"code": "pothole", "order_index": 2},
    {"code": "garbage", "order_index": 3},
    {"code": "alligator_crack", "order_index": 4},
]


class TestGeometry(unittest.TestCase):
    def test_validate_rejects_bad_boxes(self) -> None:
        for bad in [(0.5, 0.1, 0.4, 0.2), (-0.1, 0, 0.2, 0.2), (0.1, 0.1, 1.2, 0.2), (0.1, 0.1, 0.1, 0.2)]:
            with self.assertRaises(GeometryError):
                validate_bbox(*bad)

    def test_yolo_roundtrip_matches_docs_sample(self) -> None:
        # docs/format-samples/sample.yolo.txt 第一行：横向裂缝（索引 1）
        class_index, bbox = parse_yolo_line("1 0.275000 0.230000 0.350000 0.060000")
        self.assertEqual(class_index, 1)
        self.assertAlmostEqual(bbox["x1"], 0.10, places=6)
        self.assertAlmostEqual(bbox["y1"], 0.20, places=6)
        self.assertAlmostEqual(bbox["x2"], 0.45, places=6)
        self.assertAlmostEqual(bbox["y2"], 0.26, places=6)
        self.assertEqual(yolo_line(1, bbox), "1 0.275000 0.230000 0.350000 0.060000")

    def test_coco_and_labelme_roundtrip(self) -> None:
        bbox = validate_bbox(0.10, 0.20, 0.45, 0.26)
        self.assertEqual(bbox_to_coco(bbox, 1920, 1080), [192.0, 216.0, 672.0, 64.8])
        back = bbox_from_coco([192.0, 216.0, 672.0, 64.8], 1920, 1080)
        for key in ("x1", "y1", "x2", "y2"):
            self.assertAlmostEqual(back[key], bbox[key], places=6)
        self.assertEqual(bbox_to_labelme(bbox, 1920, 1080), [[192.0, 216.0], [864.0, 280.8]])
        back_lm = bbox_from_labelme([[864.0, 280.8], [192.0, 216.0]], 1920, 1080)  # 顺序颠倒也应正常
        for key in ("x1", "y1", "x2", "y2"):
            self.assertAlmostEqual(back_lm[key], bbox[key], places=6)

    def test_xywh_norm_roundtrip(self) -> None:
        bbox = validate_bbox(0.2, 0.3, 0.6, 0.7)
        cx, cy, w, h = bbox_to_xywh_norm(bbox)
        self.assertAlmostEqual(cx, 0.4)
        self.assertAlmostEqual(cy, 0.5)
        self.assertAlmostEqual(w, 0.4)
        self.assertAlmostEqual(h, 0.4)
        back = bbox_from_xywh_norm(cx, cy, w, h)
        for key in ("x1", "y1", "x2", "y2"):
            self.assertAlmostEqual(back[key], bbox[key], places=12)

    def test_iou_and_area(self) -> None:
        a = validate_bbox(0.0, 0.0, 0.5, 0.5)
        b = validate_bbox(0.25, 0.25, 0.75, 0.75)
        self.assertAlmostEqual(bbox_area(a), 0.25)
        self.assertAlmostEqual(bbox_iou(a, b), (0.25 ** 2) / (0.25 + 0.25 - 0.0625))
        self.assertEqual(bbox_iou(a, validate_bbox(0.6, 0.6, 0.9, 0.9)), 0.0)

    def test_class_index_follows_order_index(self) -> None:
        mapping = class_index_map(CLASSES)
        self.assertEqual(mapping["longitudinal_crack"], 0)
        self.assertEqual(mapping["transverse_crack"], 1)
        self.assertEqual(mapping["garbage"], 3)

    def test_tiling_covers_image_without_overflow(self) -> None:
        windows = tiled_origins(2000, 1200, tile=1024, overlap=0.2)
        self.assertGreater(len(windows), 1)
        for (x, y, w, h) in windows:
            self.assertGreaterEqual(x, 0)
            self.assertGreaterEqual(y, 0)
            self.assertLessEqual(x + w, 2000)
            self.assertLessEqual(y + h, 1200)
        # 覆盖右边界与下边界
        self.assertTrue(any(x + w == 2000 for x, y, w, h in windows))
        self.assertTrue(any(y + h == 1200 for x, y, w, h in windows))

    def test_tile_to_orig_maps_back(self) -> None:
        window = (1024, 0, 976, 800)
        tile_bbox = validate_bbox(0.0, 0.0, 1.0, 1.0)
        back = tile_to_orig(tile_bbox, window, 2000, 1200)
        self.assertAlmostEqual(back["x1"], 1024 / 2000, places=4)
        self.assertAlmostEqual(back["x2"], 1.0, places=4)
        self.assertAlmostEqual(back["y2"], 800 / 1200, places=4)


if __name__ == "__main__":
    unittest.main()
