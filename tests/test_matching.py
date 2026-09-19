"""检测匹配指标测试（纯标准库 unittest）。

运行方式（仓库根目录）：
    PYTHONPATH=src:tests .venv/bin/python -m unittest tests.test_matching -v
    PYTHONPATH=src:tests .venv/bin/python -m unittest discover -s tests -t .
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import unittest
from pathlib import Path

from rdinspect.train import matching

POTHOLE = "pothole"
GARBAGE = "garbage"
ORDER = [POTHOLE, GARBAGE]

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
TESTS_PATH = REPO_ROOT / "tests"


def box(x1: float, y1: float, x2: float, y2: float) -> dict[str, float]:
    """构造归一化 bbox。"""
    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}


def gt(code: str, bbox: dict[str, float]) -> dict:
    """构造真值项。"""
    return {"class_code": code, "bbox": bbox}


def pred(code: str, bbox: dict[str, float], score: float) -> dict:
    """构造预测项。"""
    return {"class_code": code, "bbox": bbox, "score": score}


def image(image_id: int, gts: list[dict], preds: list[dict], *,
          width: int | None = None, height: int | None = None) -> dict:
    """构造 per_image 项（size_bucket_recall 需要 width/height）。"""
    item: dict = {"image_id": image_id, "gts": gts, "preds": preds}
    if width is not None:
        item["width"] = width
    if height is not None:
        item["height"] = height
    return item


class TestIoU(unittest.TestCase):
    """iou：交并比与非法框处理。"""

    def test_identical_boxes_return_exactly_one(self) -> None:
        for candidate in (box(0.1, 0.2, 0.4, 0.6), box(0.0, 0.0, 1.0, 1.0), box(0.31, 0.07, 0.32, 0.93)):
            with self.subTest(bbox=candidate):
                self.assertEqual(matching.iou(candidate, dict(candidate)), 1.0)
                self.assertEqual(matching.iou(candidate, candidate), 1.0)

    def test_disjoint_boxes_return_zero(self) -> None:
        a = box(0.0, 0.0, 0.2, 0.2)
        b = box(0.5, 0.5, 0.7, 0.7)
        self.assertEqual(matching.iou(a, b), 0.0)
        self.assertEqual(matching.iou(b, a), 0.0)

    def test_touching_boxes_return_zero(self) -> None:
        # 角相切、左右相切、上下相切：交集面积为 0，按「仅相切 = 0.0」处理
        corner = (box(0.0, 0.0, 0.5, 0.5), box(0.5, 0.5, 1.0, 1.0))
        vertical = (box(0.0, 0.0, 0.5, 1.0), box(0.5, 0.0, 1.0, 1.0))
        horizontal = (box(0.0, 0.0, 0.5, 0.5), box(0.0, 0.5, 0.5, 1.0))
        for a, b in (corner, vertical, horizontal):
            with self.subTest(a=a, b=b):
                self.assertEqual(matching.iou(a, b), 0.0)

    def test_half_overlap_returns_half(self) -> None:
        # 0.8x0.8 与外框相同、高度减半的框：交 0.8*0.4，并 0.8*0.8 → 0.5
        a = box(0.0, 0.0, 0.8, 0.8)
        b = box(0.0, 0.0, 0.8, 0.4)
        self.assertAlmostEqual(matching.iou(a, b), 0.5, delta=1e-9)
        self.assertAlmostEqual(matching.iou(b, a), 0.5, delta=1e-9)

    def test_quarter_shift_overlap_matches_hand_computed_value(self) -> None:
        # 0.5x0.5 与右下平移 0.25 的同尺寸框：交 0.0625，并 0.4375 → 1/7
        a = box(0.0, 0.0, 0.5, 0.5)
        b = box(0.25, 0.25, 0.75, 0.75)
        self.assertAlmostEqual(matching.iou(a, b), 1 / 7, delta=1e-9)
        self.assertAlmostEqual(matching.iou(b, a), 1 / 7, delta=1e-9)
        # 0.4x0.4 与右侧平移 0.2 的同尺寸框：交 0.08，并 0.24 → 1/3
        self.assertAlmostEqual(matching.iou(box(0.0, 0.0, 0.4, 0.4), box(0.2, 0.0, 0.6, 0.4)),
                               1 / 3, delta=1e-9)

    def test_invalid_boxes_return_zero(self) -> None:
        valid = box(0.0, 0.0, 0.5, 0.5)
        invalid = [
            box(0.2, 0.2, 0.2, 0.5),                                  # 零宽（零面积）
            box(0.2, 0.2, 0.5, 0.2),                                  # 零高（零面积）
            box(0.5, 0.1, 0.2, 0.3),                                  # 坐标倒置（负宽）
            box(0.1, 0.6, 0.5, 0.2),                                  # 坐标倒置（负高）
            {"x1": 0.1, "y1": 0.1, "x2": 0.5},                        # 缺键
            {"x1": "甲", "y1": 0.1, "x2": 0.5, "y2": 0.5},            # 非数字
            {"x1": 0.1, "y1": 0.1, "x2": float("nan"), "y2": 0.5},    # NaN
            {"x1": 0.1, "y1": 0.1, "x2": float("inf"), "y2": 0.5},    # Inf
            {"x1": None, "y1": 0.1, "x2": 0.5, "y2": 0.5},            # None
            {},                                                       # 空字典
        ]
        for bad in invalid:
            with self.subTest(bbox=bad):
                self.assertEqual(matching.iou(bad, valid), 0.0)
                self.assertEqual(matching.iou(valid, bad), 0.0)
                self.assertEqual(matching.iou(bad, bad), 0.0)


class TestBoxesToPx(unittest.TestCase):
    """boxes_to_px：归一化 → 像素。"""

    def test_scales_half_box_exactly(self) -> None:
        self.assertEqual(matching.boxes_to_px(box(0.5, 0.5, 1.0, 1.0), 640, 480),
                         (320.0, 240.0, 640.0, 480.0))

    def test_scales_normalized_box(self) -> None:
        pixels = matching.boxes_to_px(box(0.10, 0.20, 0.45, 0.26), 1920, 1080)
        for value, expected in zip(pixels, (192.0, 216.0, 864.0, 280.8)):
            self.assertAlmostEqual(value, expected, delta=1e-9)
        self.assertLess(pixels[0], pixels[2])
        self.assertLess(pixels[1], pixels[3])


class TestMatchDetections(unittest.TestCase):
    """match_detections：贪心顺序、一对一占用、类别感知与混淆。"""

    def test_greedy_follows_score_order_not_input_order(self) -> None:
        gts = [gt(POTHOLE, box(0.0, 0.0, 0.2, 0.2))]
        preds = [pred(POTHOLE, box(0.0, 0.0, 0.2, 0.2), 0.5),   # 低分，IoU 1.0
                 pred(POTHOLE, box(0.0, 0.0, 0.2, 0.2), 0.9)]   # 高分，IoU 1.0
        result = matching.match_detections(preds, gts)
        self.assertEqual(result["pairs"], [{"pred_index": 1, "gt_index": 0, "iou": 1.0}])
        self.assertEqual(result["extra"], [0])
        self.assertEqual(result["missed"], [])

    def test_higher_score_claims_shared_gt_even_with_lower_iou(self) -> None:
        # 高分框（IoU 0.8223）先认领 GT，低分但完全重合的框只能算误检
        gts = [gt(POTHOLE, box(0.0, 0.0, 0.2, 0.2))]
        preds = [pred(POTHOLE, box(0.0, 0.0, 0.2, 0.2), 0.5),
                 pred(POTHOLE, box(0.01, 0.01, 0.21, 0.21), 0.9)]
        result = matching.match_detections(preds, gts)
        self.assertEqual(len(result["pairs"]), 1)
        self.assertEqual(result["pairs"][0]["pred_index"], 1)
        self.assertEqual(result["pairs"][0]["gt_index"], 0)
        self.assertAlmostEqual(result["pairs"][0]["iou"], 0.0361 / 0.0439, delta=1e-9)
        self.assertEqual(result["extra"], [0])

    def test_each_pred_claims_at_most_one_gt(self) -> None:
        # 一个预测与两个 GT 都达标时只认领 IoU 最大的，另一个 GT 算漏检
        gts = [gt(POTHOLE, box(0.0, 0.0, 0.2, 0.2)), gt(POTHOLE, box(0.01, 0.01, 0.21, 0.21))]
        preds = [pred(POTHOLE, box(0.0, 0.0, 0.2, 0.2), 0.9)]
        result = matching.match_detections(preds, gts)
        self.assertEqual(result["pairs"], [{"pred_index": 0, "gt_index": 0, "iou": 1.0}])
        self.assertEqual(result["missed"], [1])
        self.assertEqual(result["extra"], [])

    def test_one_gt_is_never_claimed_twice(self) -> None:
        gts = [gt(POTHOLE, box(0.0, 0.0, 0.2, 0.2))]
        preds = [pred(POTHOLE, box(0.0, 0.0, 0.2, 0.2), 0.9),
                 pred(POTHOLE, box(0.0, 0.0, 0.2, 0.2), 0.8)]
        result = matching.match_detections(preds, gts)
        self.assertEqual(result["matched_gt"], 1)
        self.assertEqual(len(result["pairs"]), 1)
        self.assertEqual(result["extra"], [1])
        self.assertEqual(result["missed"], [])
        claimed = [pair["gt_index"] for pair in result["pairs"]]
        self.assertEqual(len(claimed), len(set(claimed)))

    def test_class_aware_skips_cross_class_pairs(self) -> None:
        gts = [gt(POTHOLE, box(0.0, 0.0, 0.2, 0.2))]
        preds = [pred(GARBAGE, box(0.0, 0.0, 0.2, 0.2), 0.9),   # 高分但错类
                 pred(POTHOLE, box(0.0, 0.0, 0.2, 0.2), 0.8)]   # 低分但同类
        result = matching.match_detections(preds, gts, class_aware=True)
        self.assertEqual(result["pairs"], [{"pred_index": 1, "gt_index": 0, "iou": 1.0}])
        self.assertEqual(result["matched_gt"], 1)
        self.assertEqual(result["missed"], [])
        self.assertEqual(result["extra"], [0])
        self.assertEqual(len(result["class_confusions"]), 1)
        confusion = result["class_confusions"][0]
        self.assertEqual((confusion["pred_index"], confusion["gt_index"]), (0, 0))
        self.assertEqual(confusion["pred_class"], GARBAGE)
        self.assertEqual(confusion["gt_class"], POTHOLE)
        self.assertEqual(confusion["iou"], 1.0)

    def test_class_agnostic_differs_from_class_aware(self) -> None:
        gts = [gt(POTHOLE, box(0.0, 0.0, 0.2, 0.2))]
        preds = [pred(GARBAGE, box(0.0, 0.0, 0.2, 0.2), 0.9),
                 pred(POTHOLE, box(0.0, 0.0, 0.2, 0.2), 0.8)]
        aware = matching.match_detections(preds, gts, class_aware=True)
        agnostic = matching.match_detections(preds, gts, class_aware=False)
        # 忽略类别时高分框抢走唯一的 GT → 没有任何「同类」匹配对
        self.assertEqual(agnostic["pairs"], [])
        self.assertEqual(agnostic["matched_gt"], 0)
        self.assertEqual(agnostic["missed"], [0])
        self.assertEqual(agnostic["extra"], [0, 1])
        self.assertEqual(agnostic["class_aware"], False)
        self.assertEqual(aware["class_aware"], True)
        self.assertEqual(aware["matched_gt"], 1)
        self.assertNotEqual(aware["pairs"], agnostic["pairs"])

    def test_confusion_pair_counts_as_missed_and_extra(self) -> None:
        gts = [gt(POTHOLE, box(0.0, 0.0, 0.2, 0.2))]
        preds = [pred(GARBAGE, box(0.0, 0.0, 0.2, 0.2), 0.9)]
        result = matching.match_detections(preds, gts, class_aware=False)
        self.assertEqual(len(result["class_confusions"]), 1)
        self.assertIn(0, result["missed"])          # 混淆的 GT 算漏检
        self.assertIn(0, result["extra"])           # 混淆的 pred 算误检
        self.assertEqual(result["pairs"], [])
        self.assertEqual(result["matched_gt"], 0)

    def test_pair_and_confusion_never_share_the_same_match(self) -> None:
        gts = [gt(POTHOLE, box(0.0, 0.0, 0.2, 0.2)), gt(GARBAGE, box(0.4, 0.4, 0.6, 0.6))]
        preds = [pred(POTHOLE, box(0.0, 0.0, 0.2, 0.2), 0.9), pred(GARBAGE, box(0.4, 0.4, 0.6, 0.6), 0.8)]
        result = matching.match_detections(preds, gts)
        pairs = {(pair["pred_index"], pair["gt_index"]) for pair in result["pairs"]}
        confusions = {(item["pred_index"], item["gt_index"]) for item in result["class_confusions"]}
        self.assertEqual(pairs, {(0, 0), (1, 1)})
        self.assertEqual(confusions, set())
        self.assertEqual(pairs & confusions, set())
        for item in result["class_confusions"]:
            self.assertNotEqual(item["pred_class"], item["gt_class"])

    def test_iou_threshold_is_inclusive(self) -> None:
        # 交 0.8*0.4、并 0.8*0.8 → IoU 恰为 0.5
        gts = [gt(POTHOLE, box(0.0, 0.0, 0.8, 0.8))]
        preds = [pred(POTHOLE, box(0.0, 0.0, 0.8, 0.4), 0.9)]
        self.assertEqual(matching.match_detections(preds, gts, iou_thr=0.5)["matched_gt"], 1)
        self.assertEqual(matching.match_detections(preds, gts, iou_thr=0.51)["matched_gt"], 0)
        self.assertEqual(matching.match_detections(preds, gts, iou_thr=0.51)["missed"], [0])

    def test_counters_and_empty_inputs(self) -> None:
        empty = matching.match_detections([], [])
        self.assertEqual(empty["pairs"], [])
        self.assertEqual(empty["missed"], [])
        self.assertEqual(empty["extra"], [])
        self.assertEqual(empty["class_confusions"], [])
        self.assertEqual((empty["matched_gt"], empty["total_gt"], empty["total_pred"]), (0, 0, 0))
        self.assertEqual(empty["iou_thr"], 0.5)
        self.assertIs(empty["class_aware"], True)

        gts = [gt(POTHOLE, box(0.0, 0.0, 0.2, 0.2)), gt(GARBAGE, box(0.4, 0.4, 0.6, 0.6))]
        preds = [pred(POTHOLE, box(0.0, 0.0, 0.2, 0.2), 0.9),
                 pred(GARBAGE, box(0.7, 0.7, 0.9, 0.9), 0.7)]     # 位置错 → 误检
        result = matching.match_detections(preds, gts)
        self.assertEqual((result["matched_gt"], result["total_gt"], result["total_pred"]), (1, 2, 2))
        self.assertEqual(result["missed"], [1])
        self.assertEqual(result["extra"], [1])

    def test_missing_bbox_is_treated_as_invalid(self) -> None:
        gts = [gt(POTHOLE, box(0.0, 0.0, 0.2, 0.2))]
        preds = [{"class_code": POTHOLE, "score": 0.9}]          # 没有 bbox 字段
        result = matching.match_detections(preds, gts)
        self.assertEqual(result["matched_gt"], 0)
        self.assertEqual(result["missed"], [0])
        self.assertEqual(result["extra"], [0])


class TestConfusionMatrix(unittest.TestCase):
    """confusion_matrix：行 = 真值、列 = 预测，最后一行/列是背景。"""

    def test_labels_and_shape_include_background(self) -> None:
        result = matching.confusion_matrix([], ORDER)
        self.assertEqual(result["labels"], [POTHOLE, GARBAGE, "__background__"])
        self.assertEqual(len(result["matrix"]), len(ORDER) + 1)
        for row in result["matrix"]:
            self.assertEqual(len(row), len(ORDER) + 1)
        self.assertEqual(result["iou_thr"], 0.5)
        self.assertEqual(result["score_thr"], 0.0)
        self.assertEqual((result["total_gt"], result["total_pred"]), (0, 0))

    def test_diagonal_counts_correct_classification(self) -> None:
        images = [image(1,
                        [gt(POTHOLE, box(0.0, 0.0, 0.2, 0.2)), gt(GARBAGE, box(0.4, 0.4, 0.6, 0.6))],
                        [pred(POTHOLE, box(0.0, 0.0, 0.2, 0.2), 0.9),
                         pred(GARBAGE, box(0.4, 0.4, 0.6, 0.6), 0.8)])]
        result = matching.confusion_matrix(images, ORDER)
        self.assertEqual(result["matrix"], [[1, 0, 0], [0, 1, 0], [0, 0, 0]])
        self.assertEqual(result["normalized"], [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
        self.assertEqual((result["total_gt"], result["total_pred"]), (2, 2))

    def test_missed_gt_lands_in_last_column(self) -> None:
        images = [image(1,
                        [gt(POTHOLE, box(0.0, 0.0, 0.2, 0.2)), gt(GARBAGE, box(0.4, 0.4, 0.6, 0.6))],
                        [pred(POTHOLE, box(0.0, 0.0, 0.2, 0.2), 0.9)])]   # garbage 无预测 → 漏检
        matrix = matching.confusion_matrix(images, ORDER)["matrix"]
        background = len(ORDER)
        self.assertEqual(matrix[1][background], 1)      # 真值 garbage、预测为背景
        self.assertEqual([row[background] for row in matrix], [0, 1, 0])

    def test_extra_pred_lands_in_last_row(self) -> None:
        images = [image(1,
                        [gt(POTHOLE, box(0.0, 0.0, 0.2, 0.2))],
                        [pred(POTHOLE, box(0.0, 0.0, 0.2, 0.2), 0.9),
                         pred(GARBAGE, box(0.7, 0.7, 0.9, 0.9), 0.8)])]   # 多余 garbage → 误检
        matrix = matching.confusion_matrix(images, ORDER)["matrix"]
        background = len(ORDER)
        self.assertEqual(matrix[background], [0, 1, 0])   # 真值为背景、预测为 garbage
        self.assertEqual(matrix[0], [1, 0, 0])

    def test_cross_class_confusion_lands_in_cell_without_double_counting(self) -> None:
        images = [image(1,
                        [gt(POTHOLE, box(0.0, 0.0, 0.2, 0.2))],
                        [pred(GARBAGE, box(0.0, 0.0, 0.2, 0.2), 0.9)])]   # 完全重合但错类
        result = matching.confusion_matrix(images, ORDER)
        self.assertEqual(result["matrix"], [[0, 1, 0], [0, 0, 0], [0, 0, 0]])
        # 混淆对已落在 (pothole, garbage) 单元格，不得再计入背景列（否则行和会 > 1）
        self.assertEqual([sum(row) for row in result["matrix"]], [1, 0, 0])
        self.assertEqual(result["normalized"][0], [0.0, 1.0, 0.0])
        self.assertEqual((result["total_gt"], result["total_pred"]), (1, 1))

    def test_normalized_rows_sum_to_one_or_all_zero(self) -> None:
        images = [image(1,
                        [gt(POTHOLE, box(0.0, 0.0, 0.2, 0.2)), gt(GARBAGE, box(0.4, 0.4, 0.6, 0.6))],
                        [pred(POTHOLE, box(0.0, 0.0, 0.2, 0.2), 0.9),
                         pred(GARBAGE, box(0.7, 0.7, 0.9, 0.9), 0.8)])]
        result = matching.confusion_matrix(images, ORDER)
        self.assertEqual(result["matrix"], [[1, 0, 0], [0, 0, 1], [0, 1, 0]])
        for index, row in enumerate(result["normalized"]):
            self.assertAlmostEqual(sum(row), 1.0, delta=1e-12)
            self.assertEqual(len(row), len(result["labels"]))
            self.assertEqual(row, [value / sum(result["matrix"][index]) for value in result["matrix"][index]])
        # 全 0 行的归一化结果必须整行 0.0
        zero = matching.confusion_matrix([], ORDER)
        self.assertEqual(zero["normalized"], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        for row in zero["normalized"]:
            self.assertEqual(sum(row), 0.0)

    def test_score_thr_filters_predictions(self) -> None:
        images = [image(1,
                        [gt(POTHOLE, box(0.0, 0.0, 0.2, 0.2))],
                        [pred(POTHOLE, box(0.0, 0.0, 0.2, 0.2), 0.7),
                         pred(POTHOLE, box(0.0, 0.0, 0.2, 0.2), 0.3)])]
        unfiltered = matching.confusion_matrix(images, ORDER, score_thr=0.0)
        filtered = matching.confusion_matrix(images, ORDER, score_thr=0.7)
        self.assertEqual(unfiltered["total_pred"], 2)
        self.assertEqual(unfiltered["matrix"], [[1, 0, 0], [0, 0, 0], [1, 0, 0]])   # 低分框成误检
        self.assertEqual(filtered["total_pred"], 1)
        self.assertEqual(filtered["matrix"], [[1, 0, 0], [0, 0, 0], [0, 0, 0]])
        self.assertEqual(filtered["score_thr"], 0.7)

    def test_counts_accumulate_over_images(self) -> None:
        images = [image(1, [gt(POTHOLE, box(0.0, 0.0, 0.2, 0.2))],
                        [pred(GARBAGE, box(0.0, 0.0, 0.2, 0.2), 0.9)]),
                  image(2, [gt(POTHOLE, box(0.0, 0.0, 0.2, 0.2))],
                        [pred(POTHOLE, box(0.0, 0.0, 0.2, 0.2), 0.9)])]
        result = matching.confusion_matrix(images, ORDER)
        self.assertEqual(result["matrix"], [[1, 1, 0], [0, 0, 0], [0, 0, 0]])
        self.assertEqual((result["total_gt"], result["total_pred"]), (2, 2))
        self.assertEqual(result["normalized"][0], [0.5, 0.5, 0.0])

    def test_unknown_or_duplicate_class_raises(self) -> None:
        unknown = [image(1, [gt("crack_x", box(0.0, 0.0, 0.2, 0.2))], [])]
        with self.assertRaises(ValueError):
            matching.confusion_matrix(unknown, ORDER)
        with self.assertRaises(ValueError):
            matching.confusion_matrix([], [POTHOLE, POTHOLE])


class TestSizeBucketRecall(unittest.TestCase):
    """size_bucket_recall：COCO 口径的大小分桶召回。"""

    def _boundary_images(self) -> list[dict]:
        # 31.9px / 95.9px / 96.1px 用 100x100 图，32.0px 用 64x64 图（0.5*64 在二进制下精确）
        tiny = {"class_code": POTHOLE, "bbox": box(0.0, 0.0, 0.319, 0.319)}
        medium = {"class_code": POTHOLE, "bbox": box(0.0, 0.0, 0.959, 0.959)}
        large = {"class_code": POTHOLE, "bbox": box(0.0, 0.0, 0.961, 0.961)}
        exact32 = {"class_code": POTHOLE, "bbox": box(0.0, 0.0, 0.5, 0.5)}
        return [
            image(1, [tiny, medium, large], [pred(POTHOLE, item["bbox"], 0.9) for item in
                                             (tiny, medium, large)], width=100, height=100),
            image(2, [exact32], [pred(POTHOLE, exact32["bbox"], 0.9)], width=64, height=64),
        ]

    def test_bucket_boundaries_follow_coco_edges(self) -> None:
        result = matching.size_bucket_recall(self._boundary_images())
        buckets = result["buckets"]
        self.assertEqual(buckets["small"]["gt"], 1)      # 31.9px < 32
        self.assertEqual(buckets["medium"]["gt"], 2)     # 32.0px（左闭）与 95.9px
        self.assertEqual(buckets["large"]["gt"], 1)      # 96.1px
        self.assertEqual(result["edges"], [0.0, 32.0, 96.0])
        self.assertEqual(result["iou_thr"], 0.5)
        self.assertEqual(result["total_gt"], 4)
        self.assertEqual(list(buckets), ["small", "medium", "large"])
        self.assertEqual(buckets["small"]["recall"], 1.0)
        self.assertEqual(buckets["medium"]["recall"], 1.0)
        self.assertEqual(buckets["large"]["recall"], 1.0)

    def test_exact_96px_falls_into_large(self) -> None:
        # 96x96 图上铺满整图 → 等效边长恰好 96.0，属于 large（左闭右开）
        item = {"class_code": POTHOLE, "bbox": box(0.0, 0.0, 1.0, 1.0)}
        result = matching.size_bucket_recall(
            [image(1, [item], [pred(POTHOLE, item["bbox"], 0.9)], width=96, height=96)])
        self.assertEqual(result["buckets"]["large"]["gt"], 1)
        self.assertEqual(result["buckets"]["medium"]["gt"], 0)
        self.assertEqual(result["buckets"]["large"]["recall"], 1.0)

    def test_empty_buckets_have_none_recall(self) -> None:
        item = {"class_code": POTHOLE, "bbox": box(0.0, 0.0, 0.319, 0.319)}
        result = matching.size_bucket_recall(
            [image(1, [item], [pred(POTHOLE, item["bbox"], 0.9)], width=100, height=100)])
        self.assertEqual(result["buckets"]["small"], {"gt": 1, "matched": 1, "recall": 1.0})
        for name in ("medium", "large"):
            with self.subTest(bucket=name):
                self.assertEqual(result["buckets"][name]["gt"], 0)
                self.assertEqual(result["buckets"][name]["matched"], 0)
                self.assertIsNone(result["buckets"][name]["recall"])

    def test_partial_match_gives_fractional_recall(self) -> None:
        first = {"class_code": POTHOLE, "bbox": box(0.0, 0.0, 0.2, 0.2)}
        second = {"class_code": POTHOLE, "bbox": box(0.4, 0.4, 0.6, 0.6)}
        result = matching.size_bucket_recall(
            [image(1, [first, second], [pred(POTHOLE, first["bbox"], 0.9)],
                   width=100, height=100)])
        self.assertEqual(result["buckets"]["small"]["gt"], 2)
        self.assertEqual(result["buckets"]["small"]["matched"], 1)
        self.assertAlmostEqual(result["buckets"]["small"]["recall"], 0.5, delta=1e-12)

    def test_matching_is_class_aware(self) -> None:
        # 完全重合但类别不同的预测不算命中
        item = {"class_code": POTHOLE, "bbox": box(0.0, 0.0, 0.2, 0.2)}
        result = matching.size_bucket_recall(
            [image(1, [item], [pred(GARBAGE, item["bbox"], 0.9)], width=100, height=100)])
        self.assertEqual(result["buckets"]["small"]["matched"], 0)
        self.assertEqual(result["buckets"]["small"]["recall"], 0.0)

    def test_missing_image_size_raises(self) -> None:
        item = {"class_code": POTHOLE, "bbox": box(0.0, 0.0, 0.2, 0.2)}
        with self.assertRaises(ValueError):
            matching.size_bucket_recall([image(1, [item], [])])


class TestAveragePrecision(unittest.TestCase):
    """average_precision：VOC2010 全点插值 AP 与 mAP50。"""

    # 完全重合、位置很远的两个框，便于构造 TP / FP
    ON_TARGET = box(0.0, 0.0, 0.2, 0.2)
    OFF_TARGET = box(0.7, 0.7, 0.9, 0.9)

    def test_perfect_predictions_give_ap_one(self) -> None:
        images = [image(1, [gt(POTHOLE, self.ON_TARGET)], [pred(POTHOLE, self.ON_TARGET, 0.9)])]
        result = matching.average_precision(images, [POTHOLE])
        info = result["per_class"][POTHOLE]
        self.assertEqual(info["ap"], 1.0)
        self.assertEqual(info["precision"], [1.0])
        self.assertEqual(info["recall"], [1.0])
        self.assertEqual((info["n_gt"], info["n_pred"]), (1, 1))
        self.assertNotIn("no_gt", info)
        self.assertEqual(result["map50"], 1.0)
        self.assertEqual(result["iou_thr"], 0.5)

    def test_no_predictions_give_ap_zero(self) -> None:
        images = [image(1, [gt(POTHOLE, self.ON_TARGET)], [])]
        info = matching.average_precision(images, [POTHOLE])["per_class"][POTHOLE]
        self.assertEqual(info["ap"], 0.0)
        self.assertEqual(info["precision"], [])
        self.assertEqual(info["recall"], [])
        self.assertEqual((info["n_gt"], info["n_pred"]), (1, 0))

    def test_wrong_class_prediction_gives_ap_zero(self) -> None:
        images = [image(1, [gt(POTHOLE, self.ON_TARGET)], [pred(GARBAGE, self.ON_TARGET, 0.9)])]
        result = matching.average_precision(images, ORDER)
        self.assertEqual(result["per_class"][POTHOLE]["ap"], 0.0)
        self.assertEqual(result["per_class"][POTHOLE]["n_pred"], 0)
        self.assertIs(result["per_class"][GARBAGE]["no_gt"], True)
        self.assertEqual(result["map50"], 0.0)

    def test_class_without_gt_is_flagged_and_map_averages_over_all_classes(self) -> None:
        images = [image(1,
                        [gt(POTHOLE, self.ON_TARGET), gt(GARBAGE, box(0.4, 0.4, 0.6, 0.6))],
                        [pred(POTHOLE, self.ON_TARGET, 0.9), pred(GARBAGE, self.OFF_TARGET, 0.8)])]
        result = matching.average_precision(images, [POTHOLE, GARBAGE, "alligator_crack"])
        self.assertEqual(result["per_class"][POTHOLE]["ap"], 1.0)
        self.assertEqual(result["per_class"][GARBAGE]["ap"], 0.0)
        self.assertEqual(result["per_class"]["alligator_crack"], {
            "ap": 0.0, "precision": [], "recall": [], "n_gt": 0, "n_pred": 0, "no_gt": True})
        # AP=0 的类（含无 GT 类）同样进入平均 → 1/3
        self.assertAlmostEqual(result["map50"], 1 / 3, delta=1e-12)

    def test_empty_class_order_gives_map_zero(self) -> None:
        images = [image(1, [gt(POTHOLE, self.ON_TARGET)], [pred(POTHOLE, self.ON_TARGET, 0.9)])]
        result = matching.average_precision(images, [])
        self.assertEqual(result["map50"], 0.0)
        self.assertEqual(result["per_class"], {})

    def test_high_score_false_positive_lowers_ap(self) -> None:
        good = [image(1, [gt(POTHOLE, self.ON_TARGET)],
                      [pred(POTHOLE, self.ON_TARGET, 0.9), pred(POTHOLE, self.OFF_TARGET, 0.8)])]
        bad = [image(1, [gt(POTHOLE, self.ON_TARGET)],
                     [pred(POTHOLE, self.OFF_TARGET, 0.9), pred(POTHOLE, self.ON_TARGET, 0.8)])]
        good_ap = matching.average_precision(good, [POTHOLE])["per_class"][POTHOLE]["ap"]
        bad_info = matching.average_precision(bad, [POTHOLE])["per_class"][POTHOLE]
        self.assertEqual(good_ap, 1.0)
        self.assertLess(bad_info["ap"], good_ap)
        # 高分为 FP 时：P = [0, 0.5]、R = [0, 1] → AP = 0.5
        self.assertEqual(bad_info["precision"], [0.0, 0.5])
        self.assertEqual(bad_info["recall"], [0.0, 1.0])
        self.assertAlmostEqual(bad_info["ap"], 0.5, delta=1e-12)

    def test_all_point_interpolation_matches_hand_computed_ap(self) -> None:
        # 经典 VOC 例：TP, FP, TP（2 个 GT）→ P = [1, 0.5, 2/3]、R = [0.5, 0.5, 1.0]
        # 全点插值：0.5*1 + 0*2/3 + 0.5*2/3 = 5/6
        images = [image(1,
                        [gt(POTHOLE, box(0.0, 0.0, 0.2, 0.2)), gt(POTHOLE, box(0.4, 0.4, 0.6, 0.6))],
                        [pred(POTHOLE, box(0.0, 0.0, 0.2, 0.2), 0.9),
                         pred(POTHOLE, self.OFF_TARGET, 0.8),
                         pred(POTHOLE, box(0.4, 0.4, 0.6, 0.6), 0.7)])]
        info = matching.average_precision(images, [POTHOLE])["per_class"][POTHOLE]
        self.assertEqual(info["precision"], [1.0, 0.5, 2 / 3])
        self.assertEqual(info["recall"], [0.5, 0.5, 1.0])
        self.assertAlmostEqual(info["ap"], 5 / 6, delta=1e-9)

    def test_each_gt_is_claimed_once_across_predictions(self) -> None:
        # 同一 GT 被两个预测重复命中时，只有高分那条算 TP（第二条计 FP，但排在 TP 之后不影响 AP）
        images = [image(1, [gt(POTHOLE, self.ON_TARGET)],
                        [pred(POTHOLE, self.ON_TARGET, 0.9), pred(POTHOLE, self.ON_TARGET, 0.8)])]
        info = matching.average_precision(images, [POTHOLE])["per_class"][POTHOLE]
        self.assertEqual(info["n_pred"], 2)
        self.assertEqual(info["precision"], [1.0, 0.5])
        self.assertEqual(info["recall"], [1.0, 1.0])
        self.assertEqual(info["ap"], 1.0)

    def test_unknown_class_code_raises(self) -> None:
        images = [image(1, [gt("crack_x", self.ON_TARGET)], [])]
        with self.assertRaises(ValueError):
            matching.average_precision(images, ORDER)


class TestPrecisionRecallAt(unittest.TestCase):
    """precision_recall_at：固定置信度阈值下的 P/R/F1。"""

    ON_TARGET = box(0.0, 0.0, 0.2, 0.2)
    OFF_TARGET = box(0.7, 0.7, 0.9, 0.9)

    def test_all_correct_gives_perfect_scores(self) -> None:
        images = [image(1, [gt(POTHOLE, self.ON_TARGET)], [pred(POTHOLE, self.ON_TARGET, 0.9)])]
        result = matching.precision_recall_at(images, ORDER)
        self.assertEqual(result["overall"],
                         {"tp": 1, "fp": 0, "fn": 0, "precision": 1.0, "recall": 1.0, "f1": 1.0})
        self.assertEqual(result["conf"], 0.25)
        self.assertEqual(result["iou_thr"], 0.5)
        self.assertEqual(result["per_class"][POTHOLE],
                         {"tp": 1, "fp": 0, "fn": 0, "precision": 1.0, "recall": 1.0})

    def test_all_wrong_gives_zero_precision_and_recall(self) -> None:
        images = [image(1, [gt(POTHOLE, self.ON_TARGET)], [pred(GARBAGE, self.ON_TARGET, 0.9)])]
        overall = matching.precision_recall_at(images, ORDER)["overall"]
        self.assertEqual((overall["tp"], overall["fp"], overall["fn"]), (0, 1, 1))
        self.assertEqual(overall["precision"], 0.0)
        self.assertEqual(overall["recall"], 0.0)
        self.assertEqual(overall["f1"], 0.0)

    def test_no_predictions_gives_none_precision(self) -> None:
        images = [image(1, [gt(POTHOLE, self.ON_TARGET)], [])]
        overall = matching.precision_recall_at(images, ORDER)["overall"]
        self.assertEqual((overall["tp"], overall["fp"], overall["fn"]), (0, 0, 1))
        self.assertIsNone(overall["precision"])
        self.assertEqual(overall["recall"], 0.0)
        self.assertIsNone(overall["f1"])
        self.assertIsNone(matching.precision_recall_at(images, ORDER)["per_class"][GARBAGE]["precision"])

    def test_per_class_counts(self) -> None:
        images = [image(1,
                        [gt(POTHOLE, self.ON_TARGET), gt(GARBAGE, box(0.4, 0.4, 0.6, 0.6))],
                        [pred(POTHOLE, self.ON_TARGET, 0.9), pred(GARBAGE, self.OFF_TARGET, 0.8)])]
        result = matching.precision_recall_at(images, ORDER)
        self.assertEqual(result["per_class"][POTHOLE],
                         {"tp": 1, "fp": 0, "fn": 0, "precision": 1.0, "recall": 1.0})
        self.assertEqual(result["per_class"][GARBAGE],
                         {"tp": 0, "fp": 1, "fn": 1, "precision": 0.0, "recall": 0.0})
        self.assertEqual(result["overall"],
                         {"tp": 1, "fp": 1, "fn": 1, "precision": 0.5, "recall": 0.5, "f1": 0.5})

    def test_conf_threshold_filters_low_score_predictions(self) -> None:
        images = [image(1, [gt(POTHOLE, self.ON_TARGET)], [pred(POTHOLE, self.ON_TARGET, 0.1)])]
        filtered = matching.precision_recall_at(images, ORDER, conf=0.25)
        self.assertEqual((filtered["overall"]["tp"], filtered["overall"]["fp"]), (0, 0))
        self.assertIsNone(filtered["overall"]["precision"])
        self.assertEqual(filtered["overall"]["fn"], 1)
        kept = matching.precision_recall_at(images, ORDER, conf=0.05)
        self.assertEqual(kept["overall"]["tp"], 1)
        self.assertEqual(kept["overall"]["f1"], 1.0)

    def test_conf_threshold_is_inclusive(self) -> None:
        images = [image(1, [gt(POTHOLE, self.ON_TARGET)], [pred(POTHOLE, self.ON_TARGET, 0.25)])]
        result = matching.precision_recall_at(images, ORDER, conf=0.25)
        self.assertEqual(result["overall"]["tp"], 1)
        self.assertEqual(result["conf"], 0.25)

    def test_empty_inputs_give_none_metrics(self) -> None:
        overall = matching.precision_recall_at([], ORDER)["overall"]
        self.assertEqual((overall["tp"], overall["fp"], overall["fn"]), (0, 0, 0))
        self.assertIsNone(overall["precision"])
        self.assertIsNone(overall["recall"])
        self.assertIsNone(overall["f1"])


class TestStdlibOnly(unittest.TestCase):
    """守卫测试：matching 必须保持纯标准库（不得依赖 numpy/torch/cv2）。"""

    def test_import_pulls_in_no_ml_packages(self) -> None:
        """在全新解释器里导入模块后，sys.modules 不含 torch/numpy/cv2。

        必须放到子进程里断言：同一个 unittest 进程里先执行的 test_api 会经
        ``rdinspect.api.app`` 把 numpy 装进 ``sys.modules``，进程内断言会假阳性。
        """
        probe = (
            "import json, sys\n"
            "import rdinspect.train.matching\n"
            "leaked = sorted(name for name in ('torch', 'numpy', 'cv2') if name in sys.modules)\n"
            "print(json.dumps(leaked))\n"
        )
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join([str(SRC_PATH), str(TESTS_PATH)])
        completed = subprocess.run([sys.executable, "-c", probe], cwd=str(REPO_ROOT), env=env,
                                   capture_output=True, text=True, check=False)
        self.assertEqual(completed.returncode, 0, msg=completed.stderr)
        leaked = json.loads(completed.stdout.strip().splitlines()[-1])
        self.assertEqual(leaked, [], msg=f"matching 模块引入了 ML 依赖: {leaked}")

    def test_source_only_imports_stdlib(self) -> None:
        source = Path(matching.__file__).read_text(encoding="utf-8")
        for name in ("numpy", "torch", "cv2", "PIL", "yaml", "fastapi"):
            with self.subTest(module=name):
                self.assertIsNone(re.search(rf"^\s*(?:import|from)\s+{name}\b", source, re.MULTILINE))
        imported = set(re.findall(r"^(?:import|from)\s+([A-Za-z_][\w.]*)", source, re.MULTILINE))
        self.assertEqual(imported, {"__future__", "math", "typing"})


if __name__ == "__main__":
    unittest.main()
