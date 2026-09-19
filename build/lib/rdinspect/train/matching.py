"""检测匹配指标（纯标准库，不依赖 numpy/torch/cv2）：IoU、贪心匹配、混淆矩阵、大小桶召回、AP 与 P/R。

坐标与数据约定（与 :mod:`rdinspect.core.geometry` 一致）：
  * 内部几何为归一化 bbox ``{"x1": float, "y1": float, "x2": float, "y2": float}``，取值 (0, 1)；
  * 真值项 ``{"class_code": str, "bbox": {...}}``，预测项 ``{"class_code": str, "bbox": {...}, "score": float}``；
  * 类别一律用字符串 code（如 ``longitudinal_crack`` / ``pothole``），本模块不依赖任何类别索引映射。

匹配规则（各指标共用，见 :func:`match_detections`）：
  * 「按 score 降序的贪心匹配」：预测按分数从高到低依次认领 GT，每个 GT 至多被认领一次，
    每个预测至多认领一个 GT；候选 GT 取 IoU 最大者，IoU 相同时取下标较小者（结果确定）；
  * IoU 判定为闭区间（``IoU >= iou_thr`` 即命中），默认阈值 0.5；
  * ``class_aware=True`` 只允许同类别配对；``class_aware=False`` 忽略类别，用于诊断「类别混淆」。

术语：命中（TP）= 同类且 IoU 达标的匹配对；漏检（FN）= 未被同类预测命中的 GT；误检（FP）= 未命中同类 GT 的预测。
"""

from __future__ import annotations

import math
from typing import Any, Sequence

__all__ = [
    "BACKGROUND_LABEL",
    "BUCKET_EDGES",
    "average_precision",
    "boxes_to_px",
    "confusion_matrix",
    "iou",
    "match_detections",
    "precision_recall_at",
    "size_bucket_recall",
]

BBox = dict[str, float]

#: 混淆矩阵中代表「无对应目标」的标签：最后一列 = 漏检的 GT（真值为某类、预测为背景），
#: 最后一行 = 误检的预测（真值为背景、预测为某类）。
BACKGROUND_LABEL = "__background__"

#: COCO 口径的大小桶边界（GT 像素面积的等效边长 sqrt(w*h)）：[0,32) small、[32,96) medium、[96,inf) large。
BUCKET_EDGES: tuple[float, float, float] = (0.0, 32.0, 96.0)

_BUCKET_NAMES: tuple[str, str, str] = ("small", "medium", "large")

_MISSING = (KeyError, TypeError, ValueError)


def _coords(bbox: Any) -> tuple[float, float, float, float] | None:
    """取出 bbox 的四个坐标并校验；非法（非字典/缺键/非数字/非有限值/倒置/零面积）返回 None。"""
    if not isinstance(bbox, dict):
        return None
    try:
        x1, y1 = float(bbox["x1"]), float(bbox["y1"])
        x2, y2 = float(bbox["x2"]), float(bbox["y2"])
    except _MISSING:
        return None
    if not all(math.isfinite(value) for value in (x1, y1, x2, y2)):
        return None
    if x1 >= x2 or y1 >= y2:  # 坐标倒置 / 零面积 / 负宽高
        return None
    return x1, y1, x2, y2


def _score_of(pred: Any) -> float:
    """预测分数；缺字段或不可转 float 时按 0.0 处理。"""
    if not isinstance(pred, dict):
        return 0.0
    try:
        return float(pred.get("score", 0.0))
    except _MISSING:
        return 0.0


def _class_of(item: Any) -> str:
    """类别 code；缺字段时按空串处理（空串永远不会与 class_order 中的 code 相等）。"""
    if not isinstance(item, dict):
        return ""
    code = item.get("class_code", "")
    return code if isinstance(code, str) else str(code)


def _bbox_of(item: Any) -> BBox:
    """取 bbox 字段；缺失或非法时返回空字典（后续按非法框处理，IoU 记 0.0）。"""
    if isinstance(item, dict):
        bbox = item.get("bbox")
        if isinstance(bbox, dict):
            return bbox
    return {}


def _class_index(class_order: Sequence[str]) -> dict[str, int]:
    """类别 code → 位置下标；重复 code 直接报错（否则统计会静默串行）。"""
    index_of: dict[str, int] = {}
    for position, code in enumerate(class_order):
        if code in index_of:
            raise ValueError(f"class_order 存在重复类别 code: {code!r}")
        index_of[code] = position
    return index_of


def _known_index(code: str, index_of: dict[str, int]) -> int:
    """code 的下标；不在 class_order 中时报错（避免未知类别被静默丢弃、指标偏乐观）。"""
    if code not in index_of:
        raise ValueError(f"出现未知类别 code: {code!r}（不在 class_order 中）")
    return index_of[code]


def iou(a: dict[str, float], b: dict[str, float]) -> float:
    """两个框的交并比（IoU）。

    参数：
        a: 框 ``{"x1","y1","x2","y2"}``，归一化（本函数不校验取值范围，像素框同样可用）。
        b: 同上。

    返回：
        IoU ∈ [0.0, 1.0]；完全重合为 1.0，仅相切（交集面积为 0）为 0.0。

    非法框（缺键、非数字、NaN/Inf、坐标倒置、零面积、负宽高）一律返回 0.0，不抛异常，
    以便脏标注/脏预测不会让整轮评估崩掉。
    """
    box_a, box_b = _coords(a), _coords(b)
    if box_a is None or box_b is None:
        return 0.0
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inner_w = min(ax2, bx2) - max(ax1, bx1)
    inner_h = min(ay2, by2) - max(ay1, by1)
    if inner_w <= 0.0 or inner_h <= 0.0:
        return 0.0
    inter = inner_w * inner_h
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def boxes_to_px(bbox: dict[str, float], width: int, height: int) -> tuple[float, float, float, float]:
    """归一化框 → 像素 ``(x1, y1, x2, y2)``。

    参数：
        bbox: 归一化框。
        width: 图像宽（像素）。
        height: 图像高（像素）。

    说明：纯换算，不做合法性校验（缺键/非数字会直接抛出 KeyError/TypeError）。
    """
    return (
        float(bbox["x1"]) * width,
        float(bbox["y1"]) * height,
        float(bbox["x2"]) * width,
        float(bbox["y2"]) * height,
    )


def _greedy_pairs(preds: Sequence[Any], gts: Sequence[Any], *, iou_thr: float,
                  class_aware: bool) -> list[tuple[int, int, float]]:
    """按 score 降序的贪心匹配，返回 ``(pred_index, gt_index, iou)`` 列表（贪心接受顺序）。"""
    order = sorted(range(len(preds)), key=lambda index: (-_score_of(preds[index]), index))
    used_gt: set[int] = set()
    accepted: list[tuple[int, int, float]] = []
    for pred_index in order:
        pred_class = _class_of(preds[pred_index])
        pred_bbox = _bbox_of(preds[pred_index])
        best_gt, best_iou = -1, 0.0
        for gt_index, gt in enumerate(gts):
            if gt_index in used_gt:
                continue
            if class_aware and _class_of(gt) != pred_class:
                continue
            overlap = iou(pred_bbox, _bbox_of(gt))
            if overlap < iou_thr:
                continue
            if best_gt < 0 or overlap > best_iou:
                best_gt, best_iou = gt_index, overlap
        if best_gt >= 0:
            used_gt.add(best_gt)
            accepted.append((pred_index, best_gt, best_iou))
    return accepted


def match_detections(preds: Sequence[dict], gts: Sequence[dict], *, iou_thr: float = 0.5,
                     class_aware: bool = True) -> dict[str, Any]:
    """单图检测匹配：按 score 降序贪心，一对一占用，可区分「类别混淆」。

    参数：
        preds: 预测项列表 ``{"class_code","bbox","score"}``（顺序任意，内部按 score 降序）。
        gts: 真值项列表 ``{"class_code","bbox"}``。
        iou_thr: IoU 命中阈值（闭区间，``IoU >= iou_thr``）。
        class_aware: True 时只有类别相同的 pred/gt 才能配对；False 时忽略类别（用于发现类别混淆）。

    返回：
        ``{"iou_thr", "class_aware", "pairs", "missed", "extra", "class_confusions",
        "matched_gt", "total_gt", "total_pred"}``

      * ``pairs``: 匹配成功的**同类**对，按贪心接受顺序（即 score 降序）排列；
      * ``missed``: 未匹配的 gt 下标（漏检），升序；
      * ``extra``: 未匹配的 pred 下标（误检），升序；
      * ``class_confusions``: IoU 达标但类别不同的对（``pred_class != gt_class``），恒基于
        class-agnostic 匹配计算，因此两种 ``class_aware`` 取值下都会返回，便于诊断混淆；
      * ``matched_gt``/``total_gt``/``total_pred``: 命中数（= ``len(pairs)``）与总数。

    一致性保证：同一对不会同时出现在 ``pairs`` 与 ``class_confusions`` 中（前者同类、后者异类）；
    被计入 ``class_confusions`` 的 gt 计漏检、pred 计误检（它们不是有效命中）。
    """
    agnostic = _greedy_pairs(preds, gts, iou_thr=iou_thr, class_aware=False)
    accepted = _greedy_pairs(preds, gts, iou_thr=iou_thr, class_aware=True) if class_aware else agnostic

    pairs: list[dict[str, Any]] = []
    for pred_index, gt_index, overlap in accepted:
        if _class_of(preds[pred_index]) != _class_of(gts[gt_index]):
            continue  # class_aware=True 时不会出现；此处兜底保证 pairs 只含同类对
        pairs.append({"pred_index": pred_index, "gt_index": gt_index, "iou": overlap})

    confusions: list[dict[str, Any]] = []
    for pred_index, gt_index, overlap in agnostic:
        pred_class, gt_class = _class_of(preds[pred_index]), _class_of(gts[gt_index])
        if pred_class == gt_class:
            continue
        confusions.append({"pred_index": pred_index, "gt_index": gt_index, "iou": overlap,
                           "pred_class": pred_class, "gt_class": gt_class})

    matched_gt = {pair["gt_index"] for pair in pairs}
    matched_pred = {pair["pred_index"] for pair in pairs}
    return {
        "iou_thr": float(iou_thr),
        "class_aware": bool(class_aware),
        "pairs": pairs,
        "missed": [index for index in range(len(gts)) if index not in matched_gt],
        "extra": [index for index in range(len(preds)) if index not in matched_pred],
        "class_confusions": confusions,
        "matched_gt": len(pairs),
        "total_gt": len(gts),
        "total_pred": len(preds),
    }


def confusion_matrix(per_image: Sequence[dict], class_order: Sequence[str], *,
                     iou_thr: float = 0.5, score_thr: float = 0.0) -> dict[str, Any]:
    """多图混淆矩阵：**行 = 真值，列 = 预测**。

    参数：
        per_image: 每项 ``{"image_id": int, "gts": [...], "preds": [...]}``；``score < score_thr``
            的预测被忽略（阈值闭区间，``score == score_thr`` 保留）。
        class_order: 类别 code 顺序（不允许重复）。
        iou_thr: 匹配 IoU 阈值。
        score_thr: 置信度下限。

    返回：
        ``{"labels", "matrix", "normalized", "iou_thr", "score_thr", "total_gt", "total_pred"}``

    标签顺序为 ``list(class_order) + ["__background__"]``，矩阵形状 ``(n+1) x (n+1)``：
      * 对角线 = 分类正确；非对角 ``matrix[i][j]`` = 真值 i 被判成 j（类别混淆）；
      * ``matrix[gt][__background__]``（最后一列）= 漏检的 GT；
      * ``matrix[__background__][pred]``（最后一行）= 误检的预测。
      每张图先用 class-agnostic 贪心匹配，因此一个混淆对只落在它自己的 (gt, pred) 单元格，
      不会再额外计入背景行/列（不会被重复计数）。
      * ``normalized`` 按行归一（行和为 1.0；该行全 0 时整行 0.0）。
      * ``total_pred`` 统计的是 ``score_thr`` 过滤后的预测数。

    出现 ``class_order`` 之外的类别 code 时抛 ``ValueError``（不静默丢弃数据）。
    """
    index_of = _class_index(class_order)
    labels = [*class_order, BACKGROUND_LABEL]
    background = len(class_order)
    matrix = [[0] * len(labels) for _ in labels]
    total_gt = total_pred = 0

    for image in per_image:
        gts = list(image.get("gts") or [])
        preds = [pred for pred in (image.get("preds") or []) if _score_of(pred) >= score_thr]
        total_gt += len(gts)
        total_pred += len(preds)
        match = match_detections(preds, gts, iou_thr=iou_thr, class_aware=False)
        # class-agnostic 匹配结果被拆成同类对（pairs）与异类对（class_confusions），二者互斥
        paired_gt: set[int] = set()
        paired_pred: set[int] = set()
        for hit in [*match["pairs"], *match["class_confusions"]]:
            row = _known_index(_class_of(gts[hit["gt_index"]]), index_of)
            column = _known_index(_class_of(preds[hit["pred_index"]]), index_of)
            matrix[row][column] += 1
            paired_gt.add(hit["gt_index"])
            paired_pred.add(hit["pred_index"])
        # 注意：class-agnostic 下混淆对的 gt/pred 也算「漏检/误检」，但它们已经落在 (gt, pred) 单元格里，
        # 这里只补计真正没有配对的 gt/pred，避免同一个目标被统计两次（否则行和会大于 1）。
        for gt_index in range(len(gts)):
            if gt_index not in paired_gt:
                matrix[_known_index(_class_of(gts[gt_index]), index_of)][background] += 1
        for pred_index in range(len(preds)):
            if pred_index not in paired_pred:
                matrix[background][_known_index(_class_of(preds[pred_index]), index_of)] += 1

    normalized: list[list[float]] = []
    for row in matrix:
        row_total = sum(row)
        normalized.append([value / row_total for value in row] if row_total else [0.0] * len(labels))

    return {
        "labels": labels,
        "matrix": matrix,
        "normalized": normalized,
        "iou_thr": float(iou_thr),
        "score_thr": float(score_thr),
        "total_gt": total_gt,
        "total_pred": total_pred,
    }


def _equivalent_side(gt: dict, width: int, height: int) -> float:
    """GT 像素面积的等效边长 ``sqrt(w*h)``。"""
    coords = _coords(_bbox_of(gt))
    if coords is None:
        raise ValueError(f"GT bbox 非法，无法计算像素面积: {_bbox_of(gt)!r}")
    x1, y1, x2, y2 = boxes_to_px(
        {"x1": coords[0], "y1": coords[1], "x2": coords[2], "y2": coords[3]}, width, height)
    return math.sqrt(max(0.0, x2 - x1) * max(0.0, y2 - y1))


def _bucket_of(side: float) -> str:
    """等效边长 → 桶名（左闭右开：``[0,32)`` / ``[32,96)`` / ``[96, inf)``）。"""
    if side < BUCKET_EDGES[1]:
        return _BUCKET_NAMES[0]
    if side < BUCKET_EDGES[2]:
        return _BUCKET_NAMES[1]
    return _BUCKET_NAMES[2]


def size_bucket_recall(per_image: Sequence[dict], *, iou_thr: float = 0.5) -> dict[str, Any]:
    """按 GT 大小分桶的召回率（COCO 口径：等效边长 ``sqrt(w*h)``）。

    参数：
        per_image: 每项 ``{"image_id": int, "width": int, "height": int, "gts": [...], "preds": [...]}``；
            分桶需要像素尺寸，故 ``width``/``height`` 为必需字段（缺失或非正数抛 ``ValueError``）。
        iou_thr: 匹配 IoU 阈值；匹配使用 class-aware 贪心。

    返回：
        ``{"buckets": {"small": {"gt", "matched", "recall"}, "medium": ..., "large": ...},
        "iou_thr", "total_gt", "edges"}``

    桶边界：``[0,32)`` small、``[32,96)`` medium、``[96,inf)`` large（``edges`` = ``[0.0, 32.0, 96.0]``）。
    某桶 ``gt == 0`` 时 ``recall`` 为 ``None``（不写 0.0/1.0 假装有数据）。
    """
    stats = {name: {"gt": 0, "matched": 0} for name in _BUCKET_NAMES}
    total_gt = 0

    for image in per_image:
        width, height = image.get("width"), image.get("height")
        if not isinstance(width, (int, float)) or not isinstance(height, (int, float)) \
                or isinstance(width, bool) or isinstance(height, bool) or width <= 0 or height <= 0:
            raise ValueError(f"per_image 项缺少有效的 width/height（像素尺寸）: {width!r}x{height!r}")
        gts = list(image.get("gts") or [])
        preds = list(image.get("preds") or [])
        match = match_detections(preds, gts, iou_thr=iou_thr, class_aware=True)
        matched = {pair["gt_index"] for pair in match["pairs"]}
        for gt_index, gt in enumerate(gts):
            total_gt += 1
            bucket = stats[_bucket_of(_equivalent_side(gt, width, height))]
            bucket["gt"] += 1
            if gt_index in matched:
                bucket["matched"] += 1

    buckets = {
        name: {"gt": bucket["gt"], "matched": bucket["matched"],
               "recall": (bucket["matched"] / bucket["gt"]) if bucket["gt"] else None}
        for name, bucket in stats.items()
    }
    return {"buckets": buckets, "iou_thr": float(iou_thr), "total_gt": total_gt,
            "edges": [float(edge) for edge in BUCKET_EDGES]}


def _voc2010_ap(precisions: Sequence[float], recalls: Sequence[float]) -> float:
    """VOC2010 全点插值 AP：``AP = Σ_k (R_k - R_{k-1}) * P_interp(R_k)``。

    ``P_interp(R) = max_{R' >= R} P(R')``（对精确率做「从后往前取最大值」的包络），
    ``R_0 = 0``。返回未做舍入的原始浮点值；``precisions`` 为空（无预测）时返回 0.0。
    """
    if not precisions:
        return 0.0
    interpolated = [0.0] * len(precisions)
    best = 0.0
    for index in range(len(precisions) - 1, -1, -1):
        best = max(best, precisions[index])
        interpolated[index] = best
    average = 0.0
    previous_recall = 0.0
    for index, recall in enumerate(recalls):
        average += (recall - previous_recall) * interpolated[index]
        previous_recall = recall
    return average


def average_precision(per_image: Sequence[dict], class_order: Sequence[str], *,
                      iou_thr: float = 0.5) -> dict[str, Any]:
    """各类别的 VOC2010 全点插值 AP 与 mAP@IoU。

    参数：
        per_image: 每项 ``{"image_id": int, "gts": [...], "preds": [...]}``。
        class_order: 类别 code 顺序（不允许重复）。
        iou_thr: 匹配 IoU 阈值。

    算法：
        * 跨图汇总该类所有预测，按 score 降序逐条判断：命中「同图、同类、IoU >= 阈值且未被占用」
          的 GT 记 TP（多候选取 IoU 最大者），否则记 FP（class-aware 贪心，GT 一对一占用）；
        * **插值方式：VOC2010 全点插值** ``AP = Σ_k (R_k - R_{k-1}) * P_interp(R_k)``，
          其中 ``P_interp(R) = max_{R' >= R} P(R')``、``R_0 = 0``（不是 11 点插值）。
        * ``precision``/``recall`` 为按 score 降序的累计曲线（长度 = 该类预测数）。

    返回：
        ``{"iou_thr", "per_class": {code: {"ap", "precision", "recall", "n_gt", "n_pred"}},
        "map50"}``

    保守处理：某类**没有 GT** 时无法定义召回，``ap`` 记 0.0 并附 ``"no_gt": True``，
    ``precision``/``recall`` 返回空列表（不伪造曲线）；``map50`` 仍按 ``class_order`` 的类数
    算术平均（含 AP=0 的类），``class_order`` 为空时直接返回空的 ``per_class`` 与 ``map50 = 0.0``。
    非空 ``class_order`` 之外的类别 code 一律抛 ``ValueError``（不静默丢弃，避免 mAP 虚高）。
    """
    codes = list(class_order)
    if not codes:  # 没有要求任何类别：直接给空结果，data 里出现什么类别都无所谓
        return {"iou_thr": float(iou_thr), "per_class": {}, "map50": 0.0}
    index_of = _class_index(codes)
    per_class: dict[str, dict[str, Any]] = {
        code: {"ap": 0.0, "precision": [], "recall": [], "n_gt": 0, "n_pred": 0} for code in codes
    }

    image_gts: list[list[dict]] = []
    image_preds: list[list[dict]] = []
    used: list[list[bool]] = []
    scores: list[tuple[float, int, int, str]] = []
    for image_index, image in enumerate(per_image):
        gts = list(image.get("gts") or [])
        preds = list(image.get("preds") or [])
        image_gts.append(gts)
        image_preds.append(preds)
        used.append([False] * len(gts))
        for gt in gts:
            gt_code = _class_of(gt)
            _known_index(gt_code, index_of)
            per_class[gt_code]["n_gt"] += 1
        for pred_index, pred in enumerate(preds):
            code = _class_of(pred)
            _known_index(code, index_of)
            per_class[code]["n_pred"] += 1
            scores.append((_score_of(pred), image_index, pred_index, code))

    scores.sort(key=lambda entry: (-entry[0], entry[1], entry[2]))

    for code in codes:
        info = per_class[code]
        n_gt = info["n_gt"]
        if n_gt == 0:
            info["no_gt"] = True
            continue  # ap 保持 0.0；没有 GT 就没有 PR 曲线
        true_positive = 0
        precisions: list[float] = []
        recalls: list[float] = []
        for _, image_index, pred_index, entry_code in scores:
            if entry_code != code:
                continue
            pred = image_preds[image_index][pred_index]
            best_gt, best_iou = -1, 0.0
            for gt_index, gt in enumerate(image_gts[image_index]):
                if used[image_index][gt_index] or _class_of(gt) != code:
                    continue
                overlap = iou(_bbox_of(pred), _bbox_of(gt))
                if overlap < iou_thr:
                    continue
                if best_gt < 0 or overlap > best_iou:
                    best_gt, best_iou = gt_index, overlap
            if best_gt >= 0:
                used[image_index][best_gt] = True
                true_positive += 1
            precisions.append(true_positive / (len(precisions) + 1))
            recalls.append(true_positive / n_gt)
        info["precision"] = precisions
        info["recall"] = recalls
        info["ap"] = _voc2010_ap(precisions, recalls)

    mean_ap = sum(per_class[code]["ap"] for code in codes) / len(codes) if codes else 0.0
    return {"iou_thr": float(iou_thr), "per_class": per_class, "map50": mean_ap}


def _ratio(numerator: int, denominator: int) -> float | None:
    """分母为 0 时返回 None（不写 0.0 假装有数据）。"""
    return numerator / denominator if denominator else None


def precision_recall_at(per_image: Sequence[dict], class_order: Sequence[str], *,
                        conf: float = 0.25, iou_thr: float = 0.5) -> dict[str, Any]:
    """固定置信度阈值下的整体与分类别 P/R（class-aware 贪心匹配）。

    参数：
        per_image: 每项 ``{"image_id": int, "gts": [...], "preds": [...]}``。
        class_order: 类别 code 顺序（不允许重复）。
        conf: 置信度下限（``score < conf`` 的预测忽略，闭区间）。
        iou_thr: 匹配 IoU 阈值。

    返回：
        ``{"conf", "iou_thr", "overall": {"tp", "fp", "fn", "precision", "recall", "f1"},
        "per_class": {code: {"tp", "fp", "fn", "precision", "recall"}}}``

    约定：分母为 0 时对应指标为 ``None``（例如无任何预测时 ``precision`` 为 ``None``）；
    ``f1`` 在 P 或 R 为 ``None`` 时为 ``None``，P、R 均为 0.0（全错）时 ``f1`` 记 0.0。
    """
    index_of = _class_index(class_order)
    counts = {code: {"tp": 0, "fp": 0, "fn": 0} for code in class_order}

    for image in per_image:
        gts = list(image.get("gts") or [])
        preds = [pred for pred in (image.get("preds") or []) if _score_of(pred) >= conf]
        for gt in gts:
            gt_code = _class_of(gt)
            _known_index(gt_code, index_of)
            counts[gt_code]["fn"] += 1
        for pred in preds:
            pred_code = _class_of(pred)
            _known_index(pred_code, index_of)
            counts[pred_code]["fp"] += 1
        match = match_detections(preds, gts, iou_thr=iou_thr, class_aware=True)
        for pair in match["pairs"]:
            code = _class_of(gts[pair["gt_index"]])
            _known_index(code, index_of)
            counts[code]["tp"] += 1
            counts[code]["fp"] -= 1
            counts[code]["fn"] -= 1

    true_positive = sum(item["tp"] for item in counts.values())
    false_positive = sum(item["fp"] for item in counts.values())
    false_negative = sum(item["fn"] for item in counts.values())
    precision = _ratio(true_positive, true_positive + false_positive)
    recall = _ratio(true_positive, true_positive + false_negative)
    if precision is None or recall is None:
        f1 = None
    elif precision + recall == 0.0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    overall = {"tp": true_positive, "fp": false_positive, "fn": false_negative,
               "precision": precision, "recall": recall, "f1": f1}
    per_class = {
        code: {"tp": item["tp"], "fp": item["fp"], "fn": item["fn"],
               "precision": _ratio(item["tp"], item["tp"] + item["fp"]),
               "recall": _ratio(item["tp"], item["tp"] + item["fn"])}
        for code, item in counts.items()
    }
    return {"conf": float(conf), "iou_thr": float(iou_thr), "overall": overall, "per_class": per_class}
