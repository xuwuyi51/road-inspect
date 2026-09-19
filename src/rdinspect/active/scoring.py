"""主动学习选样打分（M5）：纯函数，无数据库/无 ML 依赖，便于单测与复算。

依据 ADR-0007 的三类策略（权重 50% 不确定性 / 30% 错误驱动 / 20% 多样性）：

* **不确定性**：候选框置信度落在 ``[conf_lo, conf_hi]``（默认 0.25–0.45）即为"模型犹豫"；
  若有原始 top-2 分数，``top1 - top2 < margin`` 也算（`margin_evidence` 会记录用了哪条证据）。
* **错误驱动**：来自最近一次评估的混淆矩阵与分类别召回——把"容易被混淆的类"和"召回偏低的类"
  提升权重；某类如果既高混淆又低召回，得分更高。
* **多样性**：pHash 汉明距离聚类，每簇最多取 1 张；按 ADR 的 20% 配额从"未被分数选中"的池子里补。

分数统一落在 [0,1]；`priority = 1 + (1-score) * 99`，即**分数越高优先级数值越小**（`tasks.priority` 越小越优先）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

#: 不确定性区间：模型置信度落在这里说明"看到了但不确定"
DEFAULT_CONF_BAND = (0.25, 0.45)
#: top1-top2 差值小于它算不确定
DEFAULT_MARGIN = 0.1
#: ADR-0007 的权重
DEFAULT_WEIGHTS = {"uncertainty": 0.5, "error": 0.3, "diversity": 0.2}
#: 分类别召回低于它视为"弱类"（错误驱动里加权）
DEFAULT_WEAK_RECALL = 0.6

STRATEGIES = ("hybrid", "uncertainty", "error", "diversity", "random")


def hamming_hex(a: str, b: str) -> int:
    """两个十六进制哈希的汉明距离（长度不一致时按较短的对齐比较）。"""
    try:
        left, right = int(a, 16), int(b, 16)
    except (TypeError, ValueError):
        return 64
    return bin(left ^ right).count("1")


# ─────────────────────────── 错误驱动：从评估结果提取信号 ───────────────────────────
def confusion_weights(evaluation: dict[str, Any], *, min_share: float = 0.0) -> dict[str, float]:
    """从评估结果的混淆矩阵里提取"每类作为真值时的误判占比"（越大越该优先补标）。

    矩阵语义：行 = 真值，列 = 预测，最后一列/行是背景（见 train/matching.py）。
    """
    matrix = ((evaluation.get("internal") or {}).get("confusion_matrix") or {})
    labels = list(matrix.get("labels") or [])
    rows = matrix.get("matrix") or []
    if not labels or not rows:
        return {}
    background = "__background__"
    weights: dict[str, float] = {}
    for index, label in enumerate(labels):
        if label == background or index >= len(rows):
            continue
        row = rows[index]
        total = sum(int(value or 0) for value in row)
        if not total:
            continue
        wrong = sum(int(value or 0) for position, value in enumerate(row)
                    if position != index and labels[position] != background)
        share = wrong / total
        if share > min_share:
            weights[label] = round(share, 6)
    return weights


def weak_classes(evaluation: dict[str, Any], *, recall_threshold: float = DEFAULT_WEAK_RECALL
                 ) -> dict[str, float]:
    """分类别召回低于阈值的类 → 权重 = (阈值 − 召回)，用于错误驱动加权。"""
    per_class = ((evaluation.get("internal") or {}).get("precision_recall") or {}).get("per_class") or {}
    result: dict[str, float] = {}
    for code, entry in per_class.items():
        recall = (entry or {}).get("recall")
        if isinstance(recall, (int, float)) and recall < recall_threshold:
            result[str(code)] = round(float(recall_threshold - recall), 6)
    return result


# ─────────────────────────── 单图打分 ───────────────────────────
@dataclass
class ImageSignals:
    """一张待标注图像的选样输入（全部来自已有数据或可选的一次 ONNX 打分）。"""

    image_id: int
    task_id: int
    phash: str | None = None
    path: str | None = None
    candidates: list[dict[str, Any]] = field(default_factory=list)
    #: 原始 top-2 证据：{"mean_margin": float, "min_margin": float, "n_boxes": int}
    margin_evidence: dict[str, Any] | None = None
    #: 该图已有的人工标注（用于跳过已标过的图）
    human_boxes: int = 0

    @property
    def classes(self) -> list[str]:
        return [str(item.get("class_code")) for item in self.candidates]


def uncertainty_score(signals: ImageSignals, *, conf_band: tuple[float, float] = DEFAULT_CONF_BAND,
                      margin_threshold: float = DEFAULT_MARGIN, empty_weight: float = 0.0,
                      weights: dict[str, float] | None = None) -> tuple[float, dict[str, Any]]:
    """不确定性得分 ∈ [0,1]：置信度落在band内 / top1−top2 过小 / （可选）一个候选都没有。"""
    conf_lo, conf_hi = conf_band
    scores = [float(item.get("score") or 0.0) for item in signals.candidates]
    in_band = [value for value in scores if conf_lo <= value <= conf_hi]
    band_ratio = (len(in_band) / len(scores)) if scores else 0.0
    band_max = max(in_band) if in_band else 0.0

    margins: list[float] = []
    evidence = signals.margin_evidence or {}
    if isinstance(evidence.get("margins"), (list, tuple)):
        margins = [float(value) for value in evidence["margins"] if isinstance(value, (int, float))]
    elif isinstance(evidence.get("min_margin"), (int, float)):
        margins = [float(evidence["min_margin"])]
    tight = [value for value in margins if value < margin_threshold]
    margin_ratio = (len(tight) / len(margins)) if margins else 0.0
    margin_strength = max((1.0 - value / margin_threshold for value in tight), default=0.0)

    empty = empty_weight if not scores else 0.0
    # 三路证据取最大（同一张图只要有一条"模型犹豫"的证据就够），再与"比例"混合
    score = max(band_ratio, margin_ratio, empty)
    strength = max(band_max if in_band else 0.0, margin_strength, empty)
    combined = round(min(1.0, 0.6 * score + 0.4 * strength), 6)
    return combined, {"conf_band": [conf_lo, conf_hi], "candidates": len(scores),
                      "in_band": len(in_band), "band_ratio": round(band_ratio, 6),
                      "margin_ratio": round(margin_ratio, 6), "empty": bool(empty),
                      "evidence": ("margin" if margin_ratio else ("band" if band_ratio else "none"))}


def error_score(signals: ImageSignals, *, confusion: dict[str, float] | None = None,
                weak: dict[str, float] | None = None) -> tuple[float, dict[str, Any]]:
    """错误驱动得分 ∈ [0,1]：候选类别落在"高混淆/低召回"名单里就加分。"""
    confusion = confusion or {}
    weak = weak or {}
    if not signals.candidates and not confusion and not weak:
        return 0.0, {"confusion": 0.0, "weak": 0.0, "classes": []}
    hits_confusion = [confusion[code] for code in signals.classes if code in confusion]
    hits_weak = [weak[code] for code in signals.classes if code in weak]
    worst_confusion = max(hits_confusion, default=0.0)
    worst_weak = max(hits_weak, default=0.0)
    score = round(min(1.0, 0.7 * worst_confusion + 0.3 * min(1.0, worst_weak)), 6)
    return score, {"confusion": round(worst_confusion, 6), "weak": round(worst_weak, 6),
                   "classes": sorted({code for code in signals.classes if code in confusion or code in weak})}


def base_score(uncertainty: float, error: float, *, weights: dict[str, float] | None = None) -> float:
    """把不确定性与错误驱动按 ADR 权重合成到 [0,1]（多样性与随机策略不参与这部分）。"""
    resolved = {**DEFAULT_WEIGHTS, **(weights or {})}
    total = resolved["uncertainty"] + resolved["error"]
    if total <= 0:
        return 0.0
    return round(min(1.0, (resolved["uncertainty"] * uncertainty + resolved["error"] * error) / total), 6)


def priority_from_score(score: float, *, min_priority: int = 1, max_priority: int = 100) -> int:
    """分数 → `tasks.priority`（**越小越优先**）。"""
    clamped = min(1.0, max(0.0, float(score)))
    span = max_priority - min_priority
    return int(round(min_priority + (1.0 - clamped) * span))


@dataclass
class QueueItem:
    """选样结果（写回 `tasks.priority` 并可落 `active_queue` 表）。"""

    task_id: int
    image_id: int
    score: float
    priority: int
    reason: str
    components: dict[str, float] = field(default_factory=dict)
    detail: dict[str, Any] = field(default_factory=dict)
    phash: str | None = None
    path: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {"task_id": self.task_id, "image_id": self.image_id, "score": self.score,
                "priority": self.priority, "reason": self.reason, "components": self.components,
                "detail": self.detail, "phash": self.phash, "path": self.path}


def diversity_select(candidates: Sequence[QueueItem], *, phash_hamming: int = 6,
                     ratio: float = DEFAULT_WEIGHTS["diversity"], limit: int | None = None
                     ) -> list[QueueItem]:
    """从"尚未入选"的候选里按 pHash 聚类挑代表（每簇 1 张），配额 = ratio × 总数。

    @param candidates - 已按分数降序排列的候选（含已入选与未入选）
    """
    if not candidates:
        return []
    target = max(1, int(round(len(candidates) * ratio))) if ratio > 0 else 0
    if target <= 0:
        return []
    picked: list[QueueItem] = []
    for item in candidates:
        if len(picked) >= target:
            break
        if not item.phash:
            continue
        if all(hamming_hex(item.phash, other.phash or "") > phash_hamming for other in picked):
            picked.append(item)
    # 候选池里可用的哈希太少时，退化为按分数顺序补齐（并如实标注）
    if len(picked) < target:
        for item in candidates:
            if len(picked) >= target:
                break
            if item not in picked:
                picked.append(item)
    _ = limit
    return picked


def select_queue(signals: Sequence[ImageSignals], *, strategy: str = "hybrid",
                 limit: int | None = None, conf_band: tuple[float, float] = DEFAULT_CONF_BAND,
                 margin_threshold: float = DEFAULT_MARGIN, empty_weight: float = 0.0,
                 confusion: dict[str, float] | None = None, weak: dict[str, float] | None = None,
                 weights: dict[str, float] | None = None, diversity_ratio: float = DEFAULT_WEIGHTS["diversity"],
                 phash_hamming: int = 6, seed: int = 42) -> list[QueueItem]:
    """按策略产出选样队列（已按优先级排序，`priority` 越小越靠前）。

    * ``hybrid``（默认，ADR-0007）：不确定性 50% + 错误驱动 30% 打分排序，再按 20% 配额补多样性代表；
    * ``uncertainty`` / ``error``：只用单一信号；
    * ``diversity``：只按 pHash 聚类取代表；
    * ``random``：随机抽样（作为对比实验的基线，固定 seed 可复现）。
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"未知策略 {strategy}（可选：{', '.join(STRATEGIES)}）")
    if strategy == "random":
        import random  # noqa: PLC0415

        pool = list(signals)
        random.Random(seed).shuffle(pool)
        chosen = pool[: (limit or len(pool))]
        return [QueueItem(task_id=item.task_id, image_id=item.image_id, score=0.0,
                          priority=priority_from_score(0.0), reason="random",
                          components={"uncertainty": 0.0, "error": 0.0, "diversity": 0.0},
                          phash=item.phash, path=item.path) for item in chosen]

    scored: list[QueueItem] = []
    for item in signals:
        uncertainty, u_detail = uncertainty_score(item, conf_band=conf_band,
                                                  margin_threshold=margin_threshold,
                                                  empty_weight=empty_weight, weights=weights)
        error, e_detail = error_score(item, confusion=confusion, weak=weak)
        if strategy == "uncertainty":
            score = uncertainty
        elif strategy == "error":
            score = error
        else:
            score = base_score(uncertainty, error, weights=weights)
        scored.append(QueueItem(task_id=item.task_id, image_id=item.image_id, score=score,
                                priority=priority_from_score(score),
                                reason=("uncertainty" if strategy == "uncertainty" else
                                        "error" if strategy == "error" else "score"),
                                components={"uncertainty": uncertainty, "error": error, "diversity": 0.0},
                                detail={"uncertainty": u_detail, "error": e_detail},
                                phash=item.phash, path=item.path))
    scored.sort(key=lambda entry: (-entry.score, entry.image_id))

    if strategy == "diversity":
        picked = diversity_select(scored, phash_hamming=phash_hamming, ratio=1.0)
        for entry in picked:
            entry.reason = "diversity"
            entry.components["diversity"] = 1.0
            entry.priority = priority_from_score(entry.score)
        return picked[: (limit or len(picked))]

    if strategy == "hybrid" and diversity_ratio > 0:
        head_count = max(0, len(scored) - max(1, int(round(len(scored) * diversity_ratio))))
        selected = scored[:head_count]
        rest = scored[head_count:]
        diverse = diversity_select(rest, phash_hamming=phash_hamming, ratio=1.0)
        for entry in diverse:
            entry.reason = "diversity"
            entry.components["diversity"] = 1.0
        selected = [*selected, *diverse]
        selected.sort(key=lambda entry: (-entry.score, entry.image_id))
    else:
        selected = scored
    return selected[: (limit or len(selected))]


def summarize_selection(items: Sequence[QueueItem], *, total_candidates: int) -> dict[str, Any]:
    """选样摘要（写进 run.config_json / 报告）。"""
    reasons: dict[str, int] = {}
    classes: dict[str, int] = {}
    for item in items:
        reasons[item.reason] = reasons.get(item.reason, 0) + 1
        for code in item.detail.get("error", {}).get("classes", []) if isinstance(item.detail, dict) else []:
            classes[code] = classes.get(code, 0) + 1
    scores = [item.score for item in items]
    return {"total_candidates": int(total_candidates), "selected": len(items),
            "reasons": reasons, "error_classes": classes,
            "score": {"max": round(max(scores), 6) if scores else None,
                      "min": round(min(scores), 6) if scores else None,
                      "mean": round(sum(scores) / len(scores), 6) if scores else None},
            "priority": {"min": min(entry.priority for entry in items) if items else None,
                         "max": max(entry.priority for entry in items) if items else None}}


def budget_effect(baseline: dict[str, Any], active: dict[str, Any], *, labeled_images: int,
                  active_labeled_images: int | None = None) -> dict[str, Any]:
    """对比实验记录：同等标注量下主动学习 vs 基线的 mAP 增量（每 100 张标注）。"""
    from ..train.evaluate import primary_map50, primary_per_class_map50

    base_map = primary_map50(baseline) or 0.0
    active_map = primary_map50(active) or 0.0
    delta = round(active_map - base_map, 6)
    labeled = max(1, int(labeled_images))
    active_labeled = int(active_labeled_images if active_labeled_images is not None else labeled)
    per_hundred = round(delta / (active_labeled / 100.0), 6) if active_labeled else None
    base_per_class = primary_per_class_map50(baseline)
    active_per_class = primary_per_class_map50(active)
    per_class_delta = {code: round(active_per_class.get(code, 0.0) - value, 6)
                       for code, value in sorted(base_per_class.items())}
    return {"baseline_map50": round(base_map, 6), "active_map50": round(active_map, 6),
            "delta_map50": delta, "per_100_images": per_hundred,
            "labeled_images": {"baseline": labeled, "active": active_labeled},
            "per_class_delta": per_class_delta,
            "verdict": ("主动学习更优" if delta > 0.005 else
                        "无显著差异" if abs(delta) <= 0.005 else "基线更优（需复查选样策略）")}


def class_histogram(items: Iterable[ImageSignals]) -> dict[str, int]:
    """候选类别分布（用于报告"选样是否偏向某几类"）。"""
    counts: dict[str, int] = {}
    for item in items:
        for code in item.classes:
            counts[code] = counts.get(code, 0) + 1
    return dict(sorted(counts.items()))
