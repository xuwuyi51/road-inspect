"""三格式导出/导入与往返零误差（M1 验收项之一）。"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from rdinspect.core.formats import (
    export_coco,
    export_yolo,
    import_coco,
    import_labelme,
    parse_yolo_file,
    roundtrip_check,
)
from rdinspect.core.geometry import validate_bbox

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLES = PROJECT_ROOT / "docs" / "format-samples"

CLASSES = [
    {"code": "longitudinal_crack", "name_zh": "纵向裂缝", "order_index": 0, "is_crack": True},
    {"code": "transverse_crack", "name_zh": "横向裂缝", "order_index": 1, "is_crack": True},
    {"code": "pothole", "name_zh": "坑洞", "order_index": 2, "is_crack": False},
    {"code": "garbage", "name_zh": "垃圾", "order_index": 3, "is_crack": False},
]


def make_items(count: int = 3) -> list[dict]:
    items = []
    for index in range(count):
        items.append({
            "split": "train" if index < count - 1 else "val",
            "image": {"id": index + 1, "path": f"raw/2026/img_{index}.jpg", "width": 1920, "height": 1080,
                      "sha256": f"hash{index}", "captured_at": None, "gps_lat": None, "gps_lon": None},
            "annotations": [
                {"class_code": "transverse_crack", "bbox": validate_bbox(0.10, 0.20, 0.45, 0.26),
                 "source": "model_edited", "score": 0.71},
                {"class_code": "pothole", "bbox": validate_bbox(0.70, 0.60, 0.88, 0.78),
                 "source": "human", "score": None},
                {"class_code": "garbage", "bbox": validate_bbox(0.20, 0.65, 0.38, 0.85),
                 "source": "human", "score": None},
            ],
        })
    return items


class TestFormats(unittest.TestCase):
    def test_roundtrip_is_exact_for_all_three_formats(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = roundtrip_check(make_items(), CLASSES, Path(tmp))
        self.assertTrue(result["ok"], msg=f"往返存在误差: {result['mismatches'][:3]}")
        self.assertEqual(result["count"], 0)

    def test_yolo_export_writes_data_yaml_and_labels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            stats = export_yolo(make_items(2), CLASSES, Path(tmp), copy_images=False)
            self.assertEqual(stats["images"], {"train": 1, "val": 1, "test": 0})
            self.assertEqual(stats["labels"], 6)
            yaml_text = (Path(tmp) / "data.yaml").read_text(encoding="utf-8")
            self.assertIn("nc: 4", yaml_text)
            self.assertIn("transverse_crack", yaml_text)
            label = Path(tmp) / "labels" / "train" / "img_0.txt"
            self.assertTrue(label.exists())
            rows = parse_yolo_file(label.read_text(encoding="utf-8"))
            self.assertEqual([row[0] for row in rows], ["1", "2", "3"])

    def test_coco_export_and_import_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "coco.json"
            stats = export_coco(make_items(2), CLASSES, out, copy_images=False)
            self.assertEqual(stats["images"], 2)
            data = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual([c["name"] for c in data["categories"]], [c["code"] for c in CLASSES])
            rows = import_coco(data, CLASSES)
            self.assertEqual(len(rows), 2)
            first = rows[0]["annotations"][0]
            self.assertEqual(first["class_code"], "transverse_crack")
            self.assertAlmostEqual(first["bbox"]["x1"], 0.10, places=6)

    def test_docs_sample_files_parse_with_our_parsers(self) -> None:
        """docs/format-samples 的三份样例必须能被实现解析（文档与代码同源校验）。"""
        classes = SAMPLES.joinpath("classes.txt").read_text(encoding="utf-8").split()
        yolo_rows = parse_yolo_file(SAMPLES.joinpath("sample.yolo.txt").read_text(encoding="utf-8"))
        coco = json.loads(SAMPLES.joinpath("sample.coco.json").read_text(encoding="utf-8"))
        labelme = json.loads(SAMPLES.joinpath("sample.labelme.json").read_text(encoding="utf-8"))

        coco_classes = [
            {"code": c["name"], "order_index": classes.index(c["name"]), "is_crack": False, "name_zh": c["name"]}
            for c in coco["categories"]
        ]
        self.assertEqual(len(yolo_rows), 4)
        self.assertEqual(len(import_coco(coco, coco_classes)), 1)
        parsed_labelme = import_labelme(labelme, coco_classes)
        self.assertEqual(len(parsed_labelme["annotations"]), 4)

        # YOLO 索引 → 类别名 → 与 COCO 同名标注坐标一致（零误差）
        by_index = {row[0]: row[1] for row in yolo_rows}
        for ann in import_coco(coco, coco_classes)[0]["annotations"]:
            index = str(classes.index(ann["class_code"]))
            self.assertIn(index, by_index)
            for key in ("x1", "y1", "x2", "y2"):
                self.assertAlmostEqual(by_index[index][key], ann["bbox"][key], places=6)


if __name__ == "__main__":
    unittest.main()
