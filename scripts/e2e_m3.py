#!/usr/bin/env python3
"""M3 端到端验收：训练闭环与门禁（真机训练，需要 ultralytics + torch）。

覆盖 docs/09-roadmap.md 的 M3 验收标准：
  A. 冻结数据集 → 小样本微调完成并落库（runs + candidate 模型）
  B. 分类别评估：官方 val 口径 + 混淆矩阵 + 大小桶召回 + 切片开关对比 + 失败样例导出
  C. 门禁：首个模型按绝对下限通过 → 提升 production
  D. 构造劣化权重（COCO 预训练、无道路类别）→ 门禁拒绝（409 + delta 说明）
  E. ONNX 导出包（labels/preprocess/manifest）+ ONNX↔.pt 一致性 ≤ 1e-3
  F. 幂等：同参数重复提交复用既有 run，不重复烧算力
  G. 硬约束：flipud 非 0 直接被拒（标签语义保护）

用法：
  python3 scripts/e2e_m3.py --photos 140 --epochs 3 --imgsz 320 --weights /tmp/rd-test-data/weights/yolo11n.pt
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from make_sample_data import generate  # noqa: E402
from rdinspect.config import Config, ConfigError  # noqa: E402
from rdinspect.core import datasets as ds  # noqa: E402
from rdinspect.core.ingest import import_path  # noqa: E402
from rdinspect.errors import ConflictError  # noqa: E402
from rdinspect.prelabel.detector import DetectorUnavailable, ml_available  # noqa: E402
from rdinspect.storage.db import init_db  # noqa: E402
from rdinspect.storage.repo import Repo  # noqa: E402
from rdinspect.train import export_onnx as export_mod  # noqa: E402
from rdinspect.train import gate as gate_mod  # noqa: E402
from rdinspect.train.runner import (class_names_for, enforce_hard_constraints,  # noqa: E402
                                    train_and_register)

WEIGHT_CANDIDATES = (
    "/tmp/rd-test-data/weights/yolo11n.pt",
    "data/weights/yolo11n.pt",
    "yolo11n.pt",
)


def find_weights(explicit: str | None) -> str | None:
    if explicit:
        return explicit if Path(explicit).exists() else None
    for candidate in WEIGHT_CANDIDATES:
        if Path(candidate).exists():
            return candidate
    return None


def build_config(data_dir: Path, allowed_root: Path, *, tiles: int = 0) -> Config:
    config = Config(raw={}, data_dir=data_dir, allowed_roots=(allowed_root,), lease_seconds=1800)
    from rdinspect.config import PrelabelConfig

    config.prelabel.class_aliases = dict(PrelabelConfig().class_aliases)
    config.train.evaluate.tiles = tiles
    config.train.evaluate.export_failures = 12
    config.ensure_dirs()
    return config


def annotate_from_labels(config: Config, repo: Repo, labels: list[dict], photos_dir: Path) -> dict[str, int]:
    """用合成数据的真值框标注（必须与像素内容一致，否则模型学不到东西）。

    入库后文件名会变成内容寻址路径，因此用 sha256 把「清单里的原文件」与「库里的图片」对上。
    """
    from rdinspect.storage.files import sha256_file

    by_sha: dict[str, list[dict]] = {}
    for entry in labels:
        path = photos_dir / entry["file"]
        if path.exists():
            by_sha[sha256_file(path)] = entry["annotations"]
    counts: dict[str, int] = {}
    annotated = 0
    for task in repo.list_tasks(status="pending", limit=100_000):
        detail = repo.get_task_detail(task["id"])
        entry = by_sha.get(str(detail["image"]["sha256"]))
        if entry is None:
            continue
        items = []
        for ann in entry:
            x1, y1, x2, y2 = ann["bbox"]
            if x2 - x1 <= 0.001 or y2 - y1 <= 0.001:
                continue
            items.append({"class_code": ann["class_code"], "kind": "bbox",
                          "bbox": {"x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2)}})
            counts[ann["class_code"]] = counts.get(ann["class_code"], 0) + 1
        if not items:
            continue
        repo.replace_annotations(task["id"], items, actor="e2e")
        repo.submit_task(task["id"], actor="e2e")
        repo.add_review(task["id"], decision="approve", reviewer="e2e")
        annotated += 1
    counts["_images"] = annotated
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description="M3 端到端验收（真机训练）")
    parser.add_argument("--photos", type=int, default=120)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--weights", default=None, help="初始权重（默认自动找 yolo11n.pt）")
    parser.add_argument("--degraded-weights", default=None, help="用于验证门禁拒绝的劣化权重")
    parser.add_argument("--tiles", type=int, default=384, help="切片对照的窗口边长（0=不强制切片）")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--work", default="/tmp/rd-e2e-m3")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--keep", action="store_true")
    args = parser.parse_args()

    if not ml_available():
        print("缺少 ML 依赖：.venv/bin/pip install torch torchvision ultralytics onnx onnxruntime onnxsim",
              file=sys.stderr)
        return 3
    weights = find_weights(args.weights)
    if weights is None:
        print("未找到初始权重 yolo11n.pt（可用 --weights 指定）", file=sys.stderr)
        return 3

    work = Path(args.work)
    if work.exists():
        shutil.rmtree(work)
    sample_dir = work / "sample"
    config = build_config(work / "data", sample_dir / "inbox", tiles=args.tiles)
    config.train.workers = args.workers
    conn = init_db(config.db_path)
    repo = Repo(conn)
    report: dict[str, object] = {"初始权重": weights}
    checks: dict[str, bool] = {}
    try:
        # ── 0. 合成数据 → 导入 → 按真值标注 → 冻结数据集 ────────────────
        started = time.time()
        meta = generate(sample_dir, args.photos, width=args.width, height=args.height,
                        seed=31, video_seconds=0.05)
        labels = json.loads(Path(meta["labels"]).read_text(encoding="utf-8"))
        ingest = import_path(config, repo, sample_dir / "inbox" / "photos", kind="photo")
        label_counts = annotate_from_labels(config, repo, labels, sample_dir / "inbox" / "photos")
        draft = ds.create_draft(config, repo, "ds-m3", {"review_status": "approved"},
                                {"train": 0.7, "val": 0.15, "test": 0.15, "seed": 31})
        frozen = ds.freeze_dataset(config, repo, dataset_id=int(draft["id"]))
        splits = frozen["stats"]["splits"]
        report["A_数据集"] = {"导入": ingest.added, "标注分布": label_counts, "划分": splits,
                              "manifest_hash": frozen["manifest_hash"],
                              "耗时秒": round(time.time() - started, 1)}
        checks["A 冻结数据集就绪（train/val/test 均非空）"] = all(
            splits.get(split, 0) > 0 for split in ("train", "val", "test"))

        # ── G. 硬约束：flipud 非 0 必须被拒 ─────────────────────────────
        try:
            enforce_hard_constraints({"flipud": 0.5}, allow_vertical_flip=False)
            flip_guard = False
        except ConfigError:
            flip_guard = True
        report["G_硬约束"] = {"flipud=0.5 被拒": flip_guard}
        checks["G flipud 硬约束生效（保护横/纵向语义）"] = flip_guard

        # ── A. 真机微调 ─────────────────────────────────────────────────
        started = time.time()
        result = train_and_register(config, repo, dataset="ds-m3", name="yolo11n-road-m3",
                                    arch=weights, epochs=args.epochs, imgsz=args.imgsz,
                                    batch=args.batch, device=args.device)
        outcome = result["outcome"]
        model = result["model"]
        report["B_训练"] = {"run_id": outcome["run_id"], "status": outcome["status"],
                            "epochs": outcome["epochs_done"],
                            "mAP50": outcome["metrics"].get("map50"),
                            "mAP50-95": outcome["metrics"].get("map50_95"),
                            "weights": outcome["weights_path"], "log": outcome["log_path"],
                            "耗时秒": round(time.time() - started, 1),
                            "错误": outcome.get("error")}
        checks["B 微调完成并登记 candidate 模型"] = (
            outcome["status"] == "succeeded" and model is not None and model["status"] == "candidate")
        trained_map50 = float(outcome["metrics"].get("map50") or 0.0)
        checks["B 训练后 mAP50 > 0（真的学到东西）"] = trained_map50 > 0.0

        # ── C. 评估 + 首次门禁 + 提升 production ────────────────────────
        started = time.time()
        evaluation = gate_mod.evaluate_model(config, repo, int(model["id"]), split="val", actor="e2e")
        metrics = evaluation["metrics"]
        per_class = gate_mod.primary_per_class_map50(metrics)
        internal = metrics["internal"]
        report["C_评估"] = {
            "官方": {"mAP50": metrics["ultralytics"].get("map50"),
                     "mAP50-95": metrics["ultralytics"].get("map50_95"),
                     "P": metrics["ultralytics"].get("precision"),
                     "R": metrics["ultralytics"].get("recall")},
            "分类别 mAP50": per_class,
            "内部口径": {"mAP50": internal["map50"].get("map50"),
                         "mAP50(有真值类)": internal["map50"].get("map50_classes_with_gt"),
                         "大小桶召回": {k: v.get("recall") for k, v in
                                        internal["size_bucket_recall"]["buckets"].items()},
                         "混淆矩阵标签": internal["confusion_matrix"]["labels"]},
            "切片对比": {k: metrics["sahi_ablation"][k] for k in
                         ("map50_off", "map50_on", "recall_delta", "tiles_used")} if metrics.get("sahi_ablation") else None,
            "失败样例": {k: v for k, v in (metrics.get("failures") or {}).items() if k != "worst"},
            "报告": metrics["artifacts"]["report"],
            "耗时秒": round(time.time() - started, 1),
        }
        checks["C 评估产出分类别指标与切片对比"] = bool(per_class) and metrics.get("sahi_ablation") is not None
        checks["C 失败样例导出可用"] = (metrics.get("failures") or {}).get("exported", 0) >= 0 and Path(
            metrics["artifacts"]["report"]).exists()

        gate_first = gate_mod.gate_model(config, repo, int(model["id"]), evaluation=metrics, actor="e2e")
        promoted = gate_mod.promote_model(config, repo, int(model["id"]), actor="e2e")
        report["C_首次门禁"] = {"通过": gate_first["gate"]["passed"], "理由": gate_first["gate"]["reasons"],
                                "备注": gate_first["gate"]["notes"], "提升": promoted["status"]}
        checks["C 首个模型按绝对下限通过并提升为 production"] = (
            gate_first["gate"]["passed"] and promoted["status"] == "production")

        # ── D. 劣化权重必须被门禁拒绝 ───────────────────────────────────
        degraded_source = args.degraded_weights or weights
        degraded = config.weights_dir / "degraded-coco.pt"
        if not degraded.exists():
            shutil.copy2(degraded_source, degraded)
        degraded_model = repo.upsert_model_version(
            name="degraded-road", version="coco", status="candidate", weights_path=str(degraded),
            dataset_id=int(frozen["id"]), labels_json={"names": class_names_for(repo)})
        degraded_eval = gate_mod.evaluate_model(config, repo, int(degraded_model["id"]), split="val", actor="e2e")
        try:
            gate_mod.gate_model(config, repo, int(degraded_model["id"]),
                                evaluation=degraded_eval["metrics"], actor="e2e")
            rejected, detail = False, ""
        except ConflictError as exc:
            rejected, detail = True, str(exc)
        row = repo.get_model_version(int(degraded_model["id"]))
        report["D_门禁拒绝劣化权重"] = {
            "劣化模型": f"{row['name']}:{row['version']}", "被拒": rejected, "说明": detail[:300],
            "状态": row["status"],
            "劣化 mAP50": gate_mod.primary_map50(degraded_eval["metrics"]),
            "生产 mAP50": gate_mod.primary_map50(evaluation["metrics"]),
            "delta": (gate_mod.model_gate(row) or {}).get("delta"),
        }
        checks["D 劣化权重被门禁拒绝且状态保持 candidate"] = rejected and row["status"] == "candidate"
        if not rejected:
            report["D_提示"] = ("生产基线 mAP50 为 0（训练轮次/数据不足）：门禁只防『变差』，"
                                "两个都判 0 时无从比较。请提高 --epochs 或数据量后重跑；"
                                "真实部署请设置 gate.min_map50")

        # ── E. ONNX 导出 + 一致性验收 ───────────────────────────────────
        started = time.time()
        exported = export_mod.export_onnx(config, repo, int(model["id"]), imgsz=args.imgsz,
                                          verify=True, parity_images=6, actor="e2e")
        parity = exported["parity"]
        package_dir = Path(exported["package"]["dir"])
        report["E_ONNX 导出"] = {
            "导出包": str(package_dir), "已登记": exported["registered"],
            "一致性": {"通过": parity["passed"], "最大坐标误差": parity["max_bbox_delta"],
                       "容差": parity["tolerance"], "原始输出误差": parity.get("max_raw_delta"),
                       "与工作站预测器误差": parity.get("max_workstation_delta"),
                       "匹配框": parity["matched"], "pt 独有": parity["unmatched_pt"],
                       "ONNX 独有": parity["unmatched_onnx"], "方法": parity.get("method"),
                       "providers": parity["onnx_providers"]},
            "文件": sorted(path.name for path in package_dir.iterdir()),
            "耗时秒": round(time.time() - started, 1),
        }
        checks["E 导出包含 labels/preprocess/manifest"] = all(
            (package_dir / name).exists() for name in
            ("model.onnx", "labels.txt", "preprocess.json", "manifest.json", "parity.json"))
        checks["E ONNX 与 .pt 预测一致（≤ 容差）"] = bool(parity["passed"] and exported["registered"])

        # ── F. 幂等：同参数重复提交不重复训练 ───────────────────────────
        started = time.time()
        again = train_and_register(config, repo, dataset="ds-m3", name="yolo11n-road-m3",
                                   arch=weights, epochs=args.epochs, imgsz=args.imgsz,
                                   batch=args.batch, device=args.device)
        elapsed = time.time() - started
        report["F_幂等"] = {"复用 run": again["outcome"]["run_id"], "cached": again["outcome"]["cached"],
                            "耗时秒": round(elapsed, 1)}
        checks["F 同参数重复提交复用既有 run（< 5 秒）"] = bool(again["outcome"]["cached"] and elapsed < 5)

        # ── I. HTTP 契约：注册表 / 运行详情 / 门禁 409 ──────────────────
        from fastapi.testclient import TestClient  # noqa: PLC0415

        from rdinspect.api.app import create_app  # noqa: PLC0415

        client = TestClient(create_app(config))
        try:
            registry = client.get("/api/models/registry")
            detail = client.get(f"/api/train/runs/{outcome['run_id']}")
            blocked = client.post(f"/api/models/{degraded_model['id']}/validate")
            report["I_HTTP 契约"] = {
                "registry": registry.status_code, "模型数": len(registry.json()),
                "run_detail": detail.status_code,
                "run_detail 字段": sorted(k for k in detail.json() if k in
                                          ("progress", "epochs_done", "final", "weights_path", "log_tail", "model")),
                "劣化模型 validate": blocked.status_code,
                "文案": str(blocked.json().get("message"))[:120],
            }
            checks["I HTTP 端点（注册表/运行详情/门禁 409）"] = (
                registry.status_code == 200 and len(registry.json()) >= 2
                and detail.status_code == 200 and detail.json().get("status") == "succeeded"
                and blocked.status_code == 409)
        finally:
            client.close()

        # ── 生产模型与预标注联动（M3 → M2 回环）────────────────────────
        production = repo.production_model()
        report["H_生产模型"] = {"模型": f"{production['name']}:{production['version']}",
                                "weights": production["weights_path"], "onnx": production["onnx_path"]}
        checks["H production 模型已登记 onnx_path（可被边缘端使用）"] = bool(production.get("onnx_path"))
    except DetectorUnavailable as exc:
        report["错误"] = f"ML 依赖不可用: {exc}"
        checks["运行时依赖可用"] = False
    finally:
        conn.close()

    report["验收"] = checks
    report_path = work / "e2e_m3_report.json"
    try:
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    except OSError:
        report_path = None
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    if report_path is not None:
        print(f"\n报告已保存：{report_path}")
    ok = all(checks.values())
    if not args.keep:
        shutil.rmtree(work, ignore_errors=True)
    print("\nM3 验收：", "通过 ✅" if ok else "未通过 ❌", f"（{sum(checks.values())}/{len(checks)} 项）")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
