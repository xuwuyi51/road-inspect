"""主动学习编排（M5，ADR-0007）：收集信号 → 打分选样 → 写回优先级 → 记录运行与明细。

数据来源与降级：

1. **首选**：库里的模型候选（`source='model'`，由 M2 预标注产生）→ 免费、无需推理；
2. **可选增强**：传入导出包（`--package`）时，用边缘 ONNX 运行时对候选图**再跑一遍**，
   拿到 top1/top2 分数 → 得到"模型在两个类别之间摇摆"的证据（ADR 的 `top1−top2 < 0.1` 规则）；
3. **错误驱动**：读最近一次评估（`model_versions.metrics_json.evaluation`）的混淆矩阵与分类别召回；
4. **多样性**：用 `images.phash` 做汉明距离聚类。

副作用只有两处（都可 `--dry-run` 关掉）：写 `tasks.priority` 与写 `active_queue` 明细。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

from ..config import Config
from ..storage.repo import Repo
from . import scoring
from .scoring import ImageSignals, QueueItem

#: 最近一次评估里"弱类"的默认召回阈值
DEFAULT_RECALL_THRESHOLD = scoring.DEFAULT_WEAK_RECALL


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class QueueReport:
    """一次选样的结果（CLI/API 返回，可 JSON 化）。"""

    run_id: int | None
    strategy: str
    candidates: int
    selected: int
    updated_priorities: int
    applied: bool
    dry_run: bool
    summary: dict[str, Any] = field(default_factory=dict)
    items: list[dict[str, Any]] = field(default_factory=list)
    evaluation_source: dict[str, Any] | None = None
    margin_source: dict[str, Any] | None = None
    artifact: str | None = None

    def as_dict(self, *, include_items: bool = True) -> dict[str, Any]:
        payload = {"run_id": self.run_id, "strategy": self.strategy, "candidates": self.candidates,
                   "selected": self.selected, "updated_priorities": self.updated_priorities,
                   "applied": self.applied, "dry_run": self.dry_run, "summary": self.summary,
                   "evaluation": self.evaluation_source, "margin_pass": self.margin_source,
                   "artifact": self.artifact}
        if include_items:
            payload["items"] = self.items
        return payload


def latest_evaluation(repo: Repo, *, model_name: str | None = None,
                      model_id: int | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    """取最近一次评估结果：优先生产模型，其次指定模型，最后任意最近评估过的模型。"""
    row = None
    if model_id is not None:
        row = repo.get_model_version(int(model_id))
    elif model_name:
        rows = [item for item in repo.list_model_versions() if item["name"] == model_name]
        row = rows[0] if rows else None
    if row is None:
        row = repo.production_model()
    if row is None:
        for candidate in repo.list_model_versions():
            from ..train.gate import evaluation_of  # 局部导入避免循环

            if evaluation_of(candidate):
                row = candidate
                break
    if row is None:
        return {}, {}
    from ..train.gate import evaluation_of  # 局部导入避免循环

    return evaluation_of(row), {"model_id": int(row["id"]), "name": row["name"],
                                "version": row["version"], "status": row["status"]}


def collect_signals(config: Config, repo: Repo, *, status: str = "pending",
                    scan_limit: int | None = None,
                    include_labeled: bool = False) -> list[ImageSignals]:
    """从库里收集待标注图像的选样信号。

    @param scan_limit - 只扫描前 N 个待标注任务（None = 全部）。注意这不是"选样数量"：
                        选样数量由 :func:`build_queue` 的 ``limit`` 控制，混用会让候选池被截断。
    """
    rows = repo.pending_signal_rows(status=status, limit=scan_limit)
    candidates = repo.model_candidates_for_tasks([row["task_id"] for row in rows])
    signals: list[ImageSignals] = []
    for row in rows:
        human_boxes = int(row.get("human_boxes") or 0)
        if human_boxes and not include_labeled:
            continue                     # 已经有人工标注的图不必再排队
        signals.append(ImageSignals(
            image_id=int(row["image_id"]), task_id=int(row["task_id"]), phash=row.get("phash"),
            path=str(row.get("path") or ""), candidates=candidates.get(int(row["task_id"]), []),
            human_boxes=human_boxes))
    return signals


def margin_pass(config: Config, signals: Sequence[ImageSignals], package_dir: str | Path, *,
                conf: float = 0.25, limit: int | None = None,
                progress: Callable[[int, int], None] | None = None) -> dict[str, Any]:
    """用导出包再跑一遍候选图，取 top1−top2 差值作为"模型犹豫"的直接证据。

    只在传了 `--package` 时启用；推理失败**不影响**其余策略（返回的统计里如实记录）。
    """
    from ..edge.onnx_runtime import OnnxDetector, decode_predictions, letterbox
    from ..edge.package import validate_package
    from PIL import Image  # noqa: PLC0415

    package = validate_package(package_dir)
    detector = OnnxDetector(package.model_path, class_names=package.labels, imgsz=package.imgsz,
                            conf=conf, iou=package.iou, max_detections=package.max_detections)
    target = list(signals)[: (limit or len(signals))]
    scored = 0
    margins_all: list[float] = []
    errors: list[dict[str, Any]] = []
    for index, item in enumerate(target, start=1):
        if not item.path:
            continue
        path = config.abs_data_path(item.path)
        if not path.exists():
            continue
        try:
            with Image.open(path) as handle:
                image = handle.convert("RGB")
            tensor, meta = letterbox(image, package.imgsz)
            output = detector._session.run(None, {detector._input_name: tensor})[0]  # noqa: SLF001
            decoded = decode_predictions(output, meta, conf=conf, class_names=package.labels, topk=2)
            margins: list[float] = []
            candidates: list[dict[str, Any]] = []
            for entry in decoded:
                if (entry.get("class_index") or 0) < 0:
                    continue
                top = entry.get("topk") or []
                margin = float(entry.get("margin") or 0.0)
                margins.append(margin)
                candidates.append({"class_code": entry["class_name"], "score": float(entry["score"]),
                                   "runner_up": (top[1][0] if len(top) > 1 else None),
                                   "margin": round(margin, 6)})
            item.margin_evidence = {"margins": margins, "min_margin": min(margins) if margins else None,
                                    "mean_margin": (sum(margins) / len(margins)) if margins else None,
                                    "n_boxes": len(margins)}
            if candidates:
                # 原始分数的候选覆盖库里的旧候选（更可信）
                item.candidates = [{"class_code": entry["class_code"], "score": entry["score"]}
                                   for entry in candidates]
                item.margin_evidence["candidates"] = candidates
            margins_all.extend(margins)
            scored += 1
        except Exception as exc:  # noqa: BLE001 - 单图失败不中断选样
            errors.append({"image_id": item.image_id, "error": f"{type(exc).__name__}: {exc}"})
        if progress is not None:
            progress(index, len(target))
    return {"package": str(package.root), "model": package.model_label, "scored_images": scored,
            "requested": len(target), "boxes": len(margins_all),
            "mean_margin": round(sum(margins_all) / len(margins_all), 6) if margins_all else None,
            "errors": errors[:20], "error_count": len(errors)}


def build_queue(config: Config, repo: Repo, *, limit: int | None = None, strategy: str = "hybrid",
                status: str = "pending", conf_band: tuple[float, float] = scoring.DEFAULT_CONF_BAND,
                margin_threshold: float = scoring.DEFAULT_MARGIN, empty_weight: float = 0.0,
                diversity_ratio: float = scoring.DEFAULT_WEIGHTS["diversity"],
                phash_hamming: int = 6, recall_threshold: float = DEFAULT_RECALL_THRESHOLD,
                package_dir: str | Path | None = None, margin_limit: int | None = None,
                apply: bool = True, seed: int = 42, actor: str = "cli",
                include_labeled: bool = False, scan_limit: int | None = None,
                progress: Callable[[int, int], None] | None = None) -> QueueReport:
    """完整流程：收集信号 →（可选）ONNX 打分 → 排序选样 → 写优先级/明细 → 出报告。

    @param limit      - **选样数量**上限（写多少张进队列）
    @param scan_limit - 候选池扫描上限（None = 扫描全部待标注任务）
    """
    signals = collect_signals(config, repo, status=status, scan_limit=scan_limit,
                              include_labeled=include_labeled)
    evaluation, evaluation_source = latest_evaluation(repo)
    confusion = scoring.confusion_weights(evaluation)
    weak = scoring.weak_classes(evaluation, recall_threshold=recall_threshold)
    margin_info: dict[str, Any] | None = None
    if package_dir and signals:
        try:
            margin_info = margin_pass(config, signals, package_dir, conf=conf_band[0],
                                      limit=margin_limit, progress=progress)
        except Exception as exc:  # noqa: BLE001 - 拿不到 margin 证据就退回置信度区间
            margin_info = {"error": f"{type(exc).__name__}: {exc}", "scored_images": 0}

    items = scoring.select_queue(signals, strategy=strategy, limit=limit, conf_band=conf_band,
                                margin_threshold=margin_threshold, empty_weight=empty_weight,
                                confusion=confusion, weak=weak, diversity_ratio=diversity_ratio,
                                phash_hamming=phash_hamming, seed=seed)
    summary = scoring.summarize_selection(items, total_candidates=len(signals))
    summary["confusion_weights"] = confusion
    summary["weak_classes"] = weak
    summary["class_histogram"] = scoring.class_histogram(signals)
    notes: list[str] = []
    if not any(item.score > 0 for item in items):
        notes.append("没有任何不确定性/错误驱动信号（无模型候选且无评估记录）：队列退化为按图像 id 顺序，"
                     "请先跑 rdinspect prelabel，或用 --package 传入导出包以获取 top1−top2 证据")
    if not any(item.phash for item in items):
        notes.append("候选图缺少 pHash：多样性策略无法生效（重新导入图像即可生成）")
    if not evaluation:
        notes.append("还没有模型评估记录：错误驱动策略无输入（先跑 rdinspect model evaluate）")
    summary["notes"] = notes

    run_id: int | None = None
    updated = 0
    artifact: str | None = None
    if apply and items:
        run = repo.create_active_run(strategy=strategy, config_json={
            "status": status, "limit": limit, "conf_band": list(conf_band),
            "margin_threshold": margin_threshold, "empty_weight": empty_weight,
            "diversity_ratio": diversity_ratio, "phash_hamming": phash_hamming,
            "recall_threshold": recall_threshold, "package": str(package_dir) if package_dir else None,
            "evaluation": evaluation_source, "summary": summary})
        run_id = int(run["id"])
        repo.add_active_queue_items(run_id, [item.as_dict() for item in items], strategy=strategy)
        updated = repo.set_task_priorities({item.task_id: item.priority for item in items}, actor=actor)
        repo.finish_run(run_id, status="succeeded",
                        metrics={"selected": len(items), "updated": updated, "summary": summary})
        artifact = str(_write_artifact(config, run_id, items, summary))

    report = QueueReport(run_id=run_id, strategy=strategy, candidates=len(signals),
                         selected=len(items), updated_priorities=updated, applied=bool(apply and items),
                         dry_run=not apply, summary=summary,
                         items=[item.as_dict() for item in items],
                         evaluation_source=evaluation_source or None, margin_source=margin_info,
                         artifact=artifact)
    return report


def _write_artifact(config: Config, run_id: int, items: Sequence[QueueItem],
                    summary: dict[str, Any]) -> Path:
    """把选样结果写成 JSON 产物（`data/reports/active-<run_id>.json`），便于离线复盘。"""
    import json  # noqa: PLC0415

    directory = config.data_dir / "reports"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"active-{run_id}.json"
    path.write_text(json.dumps({"run_id": run_id, "generated_at": _now(), "summary": summary,
                                "items": [item.as_dict() for item in items]},
                               ensure_ascii=False, indent=2), encoding="utf-8")
    return path


# ─────────────────────────── 门禁连续失败告警（ADR-0007 §2） ───────────────────────────
def gate_failure_streak(repo: Repo, name: str, *, limit: int = 10) -> dict[str, Any]:
    """统计同名模型"最近连续未通过门禁"的次数（新→旧，遇到通过即停）。"""
    import json  # noqa: PLC0415

    streak = 0
    history: list[dict[str, Any]] = []
    for row in repo.gate_history(name, limit=limit):
        try:
            gate = json.loads(row.get("gate_json") or "{}")
        except json.JSONDecodeError:
            gate = {}
        passed = gate.get("passed")
        history.append({"id": int(row["id"]), "version": row["version"], "status": row["status"],
                        "gate_passed": passed, "reasons": (gate.get("reasons") or [])[:3]})
        if passed is False and streak == len(history) - 1:
            streak += 1
        elif passed is False:
            continue
        else:
            break
    return {"model": name, "streak": streak, "history": history}


def gate_alerts(config: Config, repo: Repo, *, threshold: int = 2) -> dict[str, Any]:
    """扫描所有模型名，给出"连续 N 次门禁不通过 → 回看标注规范"的告警。"""
    names = sorted({str(row["name"]) for row in repo.list_model_versions()})
    alerts: list[dict[str, Any]] = []
    for name in names:
        info = gate_failure_streak(repo, name)
        if info["streak"] >= threshold:
            alerts.append({
                **info, "action": ("连续 {0} 次门禁未通过：先回看标注规范与类别定义"
                                   "（复核 Kappa、打回率、易混类对），再考虑继续调参".format(info["streak"])),
                "checks": ["annotation-workflow#复核与一致性", "classes 注册表（易混类定义）",
                           "最新一次评估报告的混淆矩阵与失败样例"]})
    return {"threshold": threshold, "alerts": alerts, "checked_models": names}
