"""模型门禁与状态机：candidate → validated → production（M3）。

门禁要回答的问题只有一个：**新权重是否比现役权重更差**。因此判据全部是「相对基线」的差值，
并且对「类别塌陷」单独设阈值（整体 mAP 持平时，某一类掉到 0 是最危险的失败模式）。

* 通过 → 状态置 ``validated``（仍不是生产模型，需再显式 promote）；
* 未通过 → 状态保持 ``candidate``，抛 :class:`ConflictError`（HTTP 409），
  详情里带上 ``delta`` 与 ``reasons``，让人知道差在哪一类、差多少；
* 无基线（首个模型）→ 用绝对下限 ``min_map50`` 判定，并在报告里显式标注「无基线」。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from ..config import Config
from ..errors import ConflictError, NotFoundError
from ..prelabel.detector import Detector
from ..storage.repo import Repo
from .evaluate import evaluate_weights, primary_map50, primary_per_class_map50


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _num(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and value == value:  # 排除 NaN
        return float(value)
    return None


def model_metrics(row: dict[str, Any]) -> dict[str, Any]:
    """解析 model_versions.metrics_json。"""
    try:
        payload = json.loads(row.get("metrics_json") or "{}")
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def model_gate(row: dict[str, Any]) -> dict[str, Any]:
    try:
        payload = json.loads(row.get("gate_json") or "{}")
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def evaluation_of(row: dict[str, Any]) -> dict[str, Any]:
    """取模型上次评估结果（metrics_json.evaluation）。"""
    payload = model_metrics(row)
    evaluation = payload.get("evaluation")
    return evaluation if isinstance(evaluation, dict) else {}


# ─────────────────────────── 门禁判定 ───────────────────────────
@dataclass
class GateDecision:
    """门禁结论（可 JSON 化，写入 model_versions.gate_json）。"""

    passed: bool
    reasons: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    delta: dict[str, Any] = field(default_factory=dict)
    baseline: dict[str, Any] | None = None
    candidate_summary: dict[str, Any] = field(default_factory=dict)
    thresholds: dict[str, Any] = field(default_factory=dict)
    checked_at: str = field(default_factory=_now)

    def as_dict(self) -> dict[str, Any]:
        return {"passed": self.passed, "reasons": self.reasons, "notes": self.notes,
                "delta": self.delta, "baseline": self.baseline,
                "candidate": self.candidate_summary, "thresholds": self.thresholds,
                "checked_at": self.checked_at}

    def detail(self) -> str:
        parts = list(self.reasons) or ["门禁通过"]
        if self.delta:
            parts.append(f"delta={json.dumps(self.delta, ensure_ascii=False)}")
        return "；".join(parts)


def compare_metrics(candidate: dict[str, Any], baseline: dict[str, Any] | None,
                    *, map50_tolerance: float, per_class_tolerance: float, min_map50: float = 0.0,
                    min_class_recall: float | None = None,
                    baseline_label: str | None = None) -> GateDecision:
    """纯函数门禁判定（候选/基线均为 evaluate_weights 的返回结构）。"""
    reasons: list[str] = []
    notes: list[str] = []
    delta: dict[str, Any] = {}
    cand_map50 = primary_map50(candidate)
    cand_per_class = primary_per_class_map50(candidate)
    thresholds = {"map50_tolerance": map50_tolerance, "per_class_tolerance": per_class_tolerance,
                  "min_map50": min_map50, "min_class_recall": min_class_recall}

    if cand_map50 is None:
        reasons.append("候选模型 mAP50 缺失或非法（评估失败），无法判定")
    elif cand_map50 < min_map50:
        reasons.append(f"候选 mAP50 {cand_map50:.4f} 低于绝对下限 {min_map50:.4f}")

    if min_class_recall is not None:
        per_class_pr = ((candidate.get("internal") or {}).get("precision_recall") or {}).get("per_class") or {}
        for code, entry in sorted(per_class_pr.items()):
            recall = _num((entry or {}).get("recall"))
            if recall is None:
                continue
            if recall < min_class_recall:
                reasons.append(f"类别 {code} 召回 {recall:.4f} 低于下限 {min_class_recall:.4f}")

    base_map50 = primary_map50(baseline) if baseline else None
    if baseline and base_map50 is not None and cand_map50 is not None:
        delta["map50"] = round(cand_map50 - base_map50, 6)
        if delta["map50"] < -map50_tolerance:
            reasons.append(f"整体 mAP50 下降 {abs(delta['map50']):.4f} 超过容差 {map50_tolerance:.4f}"
                           f"（{base_map50:.4f} → {cand_map50:.4f}）")
        base_per_class = primary_per_class_map50(baseline)
        per_class_delta: dict[str, float] = {}
        for code, value in sorted(base_per_class.items()):
            candidate_value = cand_per_class.get(code)
            if candidate_value is None:
                reasons.append(f"类别 {code} 在候选模型中缺失（基线 {value:.4f}）")
                per_class_delta[code] = -value
                continue
            per_class_delta[code] = round(candidate_value - value, 6)
            if per_class_delta[code] < -per_class_tolerance:
                if value > 0.0 and candidate_value <= 0.0:
                    reasons.append(f"类别塌陷：{code} mAP50 由 {value:.4f} 掉到 {candidate_value:.4f}")
                else:
                    reasons.append(f"类别 {code} mAP50 下降 {abs(per_class_delta[code]):.4f} "
                                   f"超过容差 {per_class_tolerance:.4f}（{value:.4f} → {candidate_value:.4f}）")
        delta["per_class_map50"] = per_class_delta
        new_classes = sorted(set(cand_per_class) - set(base_per_class))
        if new_classes:
            notes.append(f"候选新增类别指标：{', '.join(new_classes)}")
    elif baseline is None:
        notes.append("无基线模型（首个模型）：仅按绝对下限判定")
    if not reasons and (cand_map50 is None or cand_map50 <= 0.0):
        # 兜底放行但不装作"很好"：mAP50 为 0 的候选进 production 等于上线一个不工作的模型
        notes.append(f"⚠️ 候选 mAP50={cand_map50 if cand_map50 is not None else 'N/A'} 仍被放行"
                     f"（gate.min_map50={min_map50}）：请在 configs/train.yaml 设置一个真实的绝对下限"
                     f"（例如 0.30），或补足训练轮次/数据后再提升生产")

    return GateDecision(
        passed=not reasons, reasons=reasons, notes=notes, delta=delta,
        baseline=({"label": baseline_label, "map50": base_map50,
                   "model": (baseline.get("model") if isinstance(baseline, dict) else None)}
                  if baseline else None),
        candidate_summary={"map50": cand_map50, "per_class_map50": cand_per_class,
                           "weights": candidate.get("weights"),
                           "weights_sha256": candidate.get("weights_sha256"),
                           "split": candidate.get("split"), "dataset": candidate.get("dataset")},
        thresholds=thresholds,
    )


def find_baseline(config: Config, repo: Repo, candidate: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    """找基线模型行：production（默认）或上一个 validated/production（不含候选自己）。"""
    mode = (config.train.gate.baseline or "production").lower()
    if mode == "none":
        return None, None
    candidate_id = int(candidate["id"])
    if mode == "production":
        row = repo.production_model(task=candidate.get("task") or "detection")
        if row is not None and int(row["id"]) != candidate_id:
            return row, f"{row['name']}:{row['version']}"
        return None, None
    rows = repo.list_model_versions(task=candidate.get("task") or "detection")
    for row in rows:
        if int(row["id"]) == candidate_id:
            continue
        if row["status"] in ("production", "validated") and evaluation_of(row):
            return row, f"{row['name']}:{row['version']}"
    return None, None


# ─────────────────────────── 高层动作 ───────────────────────────
def evaluate_model(config: Config, repo: Repo, model_id: int, *, split: str | None = None,
                   detector_factory: Callable[[Config, str], Detector] | None = None,
                   actor: str = "system", persist: bool = True) -> dict[str, Any]:
    """评估模型并把结果写回 runs + model_versions.metrics_json.evaluation。"""
    row = repo.get_model_version(model_id)
    if row is None:
        raise NotFoundError(f"模型 {model_id} 不存在")
    weights = row.get("weights_path")
    if not weights or not Path(weights).exists():
        raise NotFoundError(f"模型 {model_id} 的权重文件不存在: {weights}")
    dataset_id = row.get("dataset_id")
    dataset = repo.get_dataset(dataset_id=dataset_id) if dataset_id else None
    if dataset is None:
        raise ConflictError(f"模型 {model_id} 未关联数据集（dataset_id={dataset_id}），无法评估")
    resolved_split = split or (config.train.evaluate.splits[0] if config.train.evaluate.splits else "val")
    run = repo.create_run("evaluate", dataset_id=int(dataset["id"]), model_id=int(model_id),
                          config_json={"model": f"{row['name']}:{row['version']}", "weights": weights,
                                       "split": resolved_split, "dataset": dataset["name"]})
    run_id = int(run["id"])
    try:
        metrics = evaluate_weights(config, repo, weights=str(weights), dataset=dataset,
                                   split=resolved_split, detector_factory=detector_factory,
                                   run_id=run_id)
    except Exception as exc:  # noqa: BLE001 - 失败要落库，便于排查
        repo.finish_run(run_id, status="failed", metrics={"model_id": model_id}, error=f"{type(exc).__name__}: {exc}")
        raise
    repo.finish_run(run_id, status="succeeded",
                    metrics={"model_id": model_id, "split": resolved_split,
                             "map50": primary_map50(metrics),
                             "map50_95": (metrics.get("ultralytics") or {}).get("map50_95"),
                             "per_class_map50": primary_per_class_map50(metrics),
                             "artifacts": metrics.get("artifacts"), "evaluation": metrics})
    if persist:
        payload = model_metrics(row)
        payload["evaluation"] = metrics
        payload["evaluated_at"] = _now()
        repo.update_model_version(int(model_id), metrics_json=payload, actor=actor)
    return {"model_id": int(model_id), "run_id": run_id, "split": resolved_split, "metrics": metrics}


def gate_model(config: Config, repo: Repo, model_id: int, *, evaluation: dict[str, Any] | None = None,
               actor: str = "system") -> dict[str, Any]:
    """对已评估的模型跑门禁；通过则置 validated，否则抛 409 并写入 gate_json。"""
    row = repo.get_model_version(model_id)
    if row is None:
        raise NotFoundError(f"模型 {model_id} 不存在")
    if row["status"] == "production":
        raise ConflictError(f"模型 {row['name']}:{row['version']} 已是 production，无需再校验")
    metrics = evaluation or evaluation_of(row)
    if not metrics:
        raise ConflictError(f"模型 {model_id} 还没有评估结果，请先 rdinspect model evaluate")
    baseline_row, baseline_label = find_baseline(config, repo, row)
    baseline_metrics = evaluation_of(baseline_row) if baseline_row is not None else None
    if baseline_row is not None and not baseline_metrics:
        baseline_metrics = None
    if baseline_row is not None and baseline_metrics is not None:
        baseline_metrics = {**baseline_metrics,
                            "model": {"id": int(baseline_row["id"]), "name": baseline_row["name"],
                                      "version": baseline_row["version"], "weights": baseline_row.get("weights_path")}}
    gate_cfg = config.train.gate
    decision = compare_metrics(
        metrics, baseline_metrics,
        map50_tolerance=gate_cfg.map50_tolerance, per_class_tolerance=gate_cfg.per_class_tolerance,
        min_map50=gate_cfg.min_map50, min_class_recall=gate_cfg.min_class_recall,
        baseline_label=baseline_label,
    )
    payload = decision.as_dict()
    payload["evaluation_run"] = {"split": metrics.get("split"), "dataset": metrics.get("dataset"),
                                 "map50": primary_map50(metrics)}
    if baseline_label is None:
        payload["baseline"] = None
    repo.update_model_version(int(model_id), gate_json=payload, actor=actor)
    if decision.passed:
        repo.set_model_status(int(model_id), "validated", actor=actor)
    else:
        if row["status"] != "candidate":
            repo.set_model_status(int(model_id), "candidate", actor=actor)
        raise ConflictError(f"门禁未通过：{decision.detail()}")
    return {"model_id": int(model_id), "status": "validated", "gate": payload}


def validate_model(config: Config, repo: Repo, model_id: int, *, split: str | None = None,
                   detector_factory: Callable[[Config, str], Detector] | None = None,
                   actor: str = "system") -> dict[str, Any]:
    """评估 + 门禁（`rdinspect model validate`）。"""
    evaluation = evaluate_model(config, repo, model_id, split=split, detector_factory=detector_factory,
                               actor=actor)
    result = gate_model(config, repo, model_id, evaluation=evaluation["metrics"], actor=actor)
    return {**result, "run_id": evaluation["run_id"], "metrics": evaluation["metrics"]}


def promote_model(config: Config, repo: Repo, model_id: int, *, actor: str = "system") -> dict[str, Any]:
    """把 validated 模型提升为 production（同任务旧生产模型自动归档）。"""
    row = repo.get_model_version(model_id)
    if row is None:
        raise NotFoundError(f"模型 {model_id} 不存在")
    if row["status"] == "production":
        return {"model_id": int(model_id), "status": "production", "changed": False, "archived": [],
                "gate": model_gate(row) or None, "weights_path": row.get("weights_path"),
                "message": "已是生产模型"}
    if row["status"] != "validated":
        gate = model_gate(row)
        hint = "；最近一次门禁未通过：" + "；".join(gate.get("reasons") or []) if gate and not gate.get("passed") else ""
        raise ConflictError(f"模型 {row['name']}:{row['version']} 状态为 {row['status']}，"
                            f"只有 validated 模型可提升为 production{hint}")
    gate = model_gate(row)
    if not gate.get("passed"):
        raise ConflictError(f"模型 {row['name']}:{row['version']} 缺少通过的门禁记录，拒绝提升为生产模型")
    repo.set_model_status(int(model_id), "production", actor=actor)
    archived = repo.archive_other_production(int(model_id), task=row.get("task") or "detection", actor=actor)
    return {"model_id": int(model_id), "status": "production", "changed": True, "archived": archived,
            "gate": gate, "weights_path": row.get("weights_path")}


def require_promotable(config: Config, repo: Repo, model_id: int) -> dict[str, Any]:
    """导出等动作的前置检查：必须是 validated/production。"""
    row = repo.get_model_version(model_id)
    if row is None:
        raise NotFoundError(f"模型 {model_id} 不存在")
    if row["status"] not in ("validated", "production"):
        raise ConflictError(f"模型 {row['name']}:{row['version']} 状态为 {row['status']}，"
                            f"只有 validated/production 模型可以导出（先跑门禁）")
    if not row.get("weights_path") or not Path(str(row["weights_path"])).exists():
        raise NotFoundError(f"模型 {model_id} 的权重文件不存在: {row.get('weights_path')}")
    return row


def model_summary(row: dict[str, Any]) -> dict[str, Any]:
    """给 API/CLI 的紧凑模型摘要（含门禁结论与主指标）。"""
    metrics = model_metrics(row)
    evaluation = evaluation_of(row)
    gate = model_gate(row)
    return {
        "id": int(row["id"]), "name": row["name"], "version": row["version"], "task": row["task"],
        "status": row["status"], "weights_path": row.get("weights_path"), "onnx_path": row.get("onnx_path"),
        "sha256": row.get("sha256"), "run_id": row.get("run_id"), "dataset_id": row.get("dataset_id"),
        "map50": primary_map50(evaluation) if evaluation else None,
        "map50_95": (evaluation.get("ultralytics") or {}).get("map50_95") if evaluation else None,
        "per_class_map50": primary_per_class_map50(evaluation) if evaluation else {},
        "gate": gate or None, "gate_passed": bool(gate.get("passed")) if gate else None,
        "train_metrics": metrics.get("train"), "evaluated_at": metrics.get("evaluated_at"),
        "artifacts": (evaluation.get("artifacts") if evaluation else None),
    }


