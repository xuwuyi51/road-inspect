#!/usr/bin/env python3
"""M5 端到端验收：主动学习与类别扩展（真机训练，需要 ultralytics + torch）。

覆盖 docs/09-roadmap.md 的 M5 验收标准：

  A. **新增类别不改代码跑通全链路**：`rdinspect classes add` → 标注新类 → 冻结 → 训练 → 导出
     （导出包的 labels 里出现新类，且类别顺序校验通过）
  B. **主动学习队列可用**：种子模型 → 预标注产生候选 → `active queue`（不确定性 + 错误驱动 +
     多样性）写回 `tasks.priority` 与 `active_queue` 明细，`tasks?strategy=active` 可只看队列
  C. **单位标注量的 mAP 增量可测**：同一批图、同样 40 张新标注预算，主动学习选样 vs 随机选样
     各训一个模型，在同一冻结测试集上比较（如实记录结论，包括"无显著差异"）
  D. 门禁连续两次不通过 → 告警（含"回看标注规范"处置建议）
  E. 统计报表：时间趋势 / 批次对比 / 类别覆盖 / GIS 导出（GeoJSON + CSV）+ API 契约

用法：
  .venv/bin/python scripts/e2e_m5.py --photos 160 --epochs 15 --imgsz 320
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from make_sample_data import generate  # noqa: E402

BOX_KEYS = ("x1", "y1", "x2", "y2")


def run_cli(args: list[str], *, data_dir: Path | None = None) -> subprocess.CompletedProcess:
    command = [str(PROJECT_ROOT / ".venv" / "bin" / "rdinspect")]
    if data_dir is not None:
        command += ["--data-dir", str(data_dir)]
    return subprocess.run([*command, *args], capture_output=True, text=True, cwd=str(PROJECT_ROOT),
                          timeout=1800)


def synth(count: int, out: Path) -> tuple[Path, dict[str, list[dict]]]:
    """生成合成数据并按文件名返回真值框（用 sha256 与库内影像对应）。"""
    from rdinspect.storage.files import sha256_file

    meta = generate(out, count, width=640, height=360, seed=515, video_seconds=0.05)
    labels = json.loads(Path(meta["labels"]).read_text(encoding="utf-8"))
    photos = out / "inbox" / "photos"
    by_sha: dict[str, list[dict]] = {}
    for entry in labels:
        path = photos / entry["file"]
        if path.exists():
            by_sha[sha256_file(path)] = entry["annotations"]
    return photos, by_sha


def truth_box(annotation: dict) -> dict[str, float]:
    x1, y1, x2, y2 = annotation["bbox"]
    return {"x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2)}


def annotate_images(repo, config, images: list[dict], by_sha: dict[str, list[dict]], *,
                    remap: dict[str, str] | None = None, limit_classes: set[str] | None = None,
                    actor: str = "e2e") -> int:
    """把生成器的真值写进库（可重映射类别），并提交+复核通过。返回标注框数。"""
    boxes = 0
    for image in images:
        annotations = by_sha.get(str(image["sha256"]), [])
        items = []
        for annotation in annotations:
            code = (remap or {}).get(annotation["class_code"], annotation["class_code"])
            if limit_classes is not None and code not in limit_classes:
                continue
            bbox = truth_box(annotation)
            if bbox["x2"] - bbox["x1"] <= 0.001 or bbox["y2"] - bbox["y1"] <= 0.001:
                continue
            items.append({"class_code": code, "kind": "bbox", "bbox": bbox})
        if not items:
            continue
        task_id = int(repo.conn.execute("SELECT id FROM tasks WHERE image_id = ?",
                                        (int(image["id"]),)).fetchone()["id"])
        repo.replace_annotations(task_id, items, actor=actor)
        repo.submit_task(task_id, actor=actor)
        repo.add_review(task_id, decision="approve", reviewer=actor)
        boxes += len(items)
    return boxes


def build_and_train(config, repo, name: str, image_ids: list[int], *, epochs: int, imgsz: int,
                    weights: str, model_name: str, val_ratio: float = 0.2) -> dict:
    from rdinspect.core import datasets as ds
    from rdinspect.train.runner import train_and_register

    draft = ds.create_draft(config, repo, name,
                            {"review_status": "approved", "image_ids": sorted(image_ids)},
                            {"train": 1.0 - val_ratio, "val": val_ratio, "test": 0.0, "seed": 11})
    frozen = ds.freeze_dataset(config, repo, dataset_id=int(draft["id"]))
    trained = train_and_register(config, repo, dataset=name, name=model_name, arch=weights,
                                 epochs=epochs, imgsz=imgsz, batch=8, device="cpu")
    return {"dataset": name, "manifest_hash": frozen["manifest_hash"],
            "splits": frozen["stats"]["splits"], "model": trained["model"],
            "train_map50": (trained["outcome"]["metrics"] or {}).get("map50")}


def main() -> int:
    parser = argparse.ArgumentParser(description="M5 端到端验收")
    parser.add_argument("--photos", type=int, default=160)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--budget", type=int, default=40, help="主动学习/随机各自的新标注预算")
    parser.add_argument("--work", default="/tmp/rd-e2e-m5")
    parser.add_argument("--keep", action="store_true")
    args = parser.parse_args()

    weights = None
    for candidate in ("/tmp/rd-test-data/weights/yolo11n.pt", "data/weights/yolo11n.pt"):
        if Path(candidate).exists():
            weights = candidate
            break
    if weights is None:
        print("未找到 yolo11n.pt（可用 --weights 指定）", file=sys.stderr)
        return 3

    work = Path(args.work)
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)
    report: dict[str, object] = {}
    checks: dict[str, bool] = {}
    started_all = time.time()

    from rdinspect.active import scoring as active_scoring
    from rdinspect.active import service as active_service
    from rdinspect.config import Config, PrelabelConfig
    from rdinspect.core.ingest import import_path
    from rdinspect.prelabel.service import prelabel_tasks
    from rdinspect.storage.db import init_db
    from rdinspect.storage.repo import Repo
    from rdinspect.train import export_onnx as export_mod
    from rdinspect.train import gate as gate_mod
    from rdinspect.train.evaluate import primary_map50

    data_dir = work / "data"
    photos, by_sha = synth(args.photos, work / "sample")
    config = Config(raw={}, data_dir=data_dir, allowed_roots=(work / "sample" / "inbox",),
                    lease_seconds=1800)
    config.prelabel.class_aliases = dict(PrelabelConfig().class_aliases)
    config.ensure_dirs()
    conn = init_db(config.db_path)
    repo = Repo(conn)
    try:
        ingest = import_path(config, repo, photos, kind="photo")
        images = [dict(row) for row in repo.conn.execute(
            "SELECT id, sha256 FROM images WHERE duplicate_of IS NULL ORDER BY id")]
        report["准备"] = {"生成": args.photos, "入库": ingest.added, "去重后": len(images)}

        # 三分区：种子 40 / 候选池 / 评估集（评估集全程不参与训练）
        seed_count = max(20, len(images) // 4)
        eval_count = max(20, len(images) // 4)
        seed_images = images[:seed_count]
        eval_images = images[seed_count:seed_count + eval_count]
        candidate_images = images[seed_count + eval_count:]
        seed_boxes = annotate_images(repo, config, seed_images, by_sha)
        eval_boxes = annotate_images(repo, config, eval_images, by_sha)
        report["A_数据分区"] = {"种子": len(seed_images), "候选池": len(candidate_images),
                                "评估集": len(eval_images), "种子框": seed_boxes, "评估框": eval_boxes}

        # 固定评估集（A/B 共用，保证可比）
        from rdinspect.core import datasets as ds

        eval_draft = ds.create_draft(config, repo, "ds-m5-eval",
                                     {"review_status": "approved",
                                      "image_ids": [int(row["id"]) for row in eval_images]},
                                     {"train": 0.5, "val": 0.25, "test": 0.25, "seed": 5})
        eval_dataset = ds.freeze_dataset(config, repo, dataset_id=int(eval_draft["id"]))
        report["A_评估集"] = {"name": eval_dataset["name"], "hash": eval_dataset["manifest_hash"],
                              "splits": eval_dataset["stats"]["splits"]}

        # ── B. 种子模型 → 预标注 → 主动学习队列 ──────────────────────────
        started = time.time()
        seed = build_and_train(config, repo, "ds-m5-seed",
                               [int(row["id"]) for row in seed_images],
                               epochs=args.epochs, imgsz=args.imgsz, weights=weights,
                               model_name="yolo11n-m5-seed")
        seed_eval = gate_mod.evaluate_model(config, repo, int(seed["model"]["id"]), split="test",
                                           dataset=eval_dataset, actor="e2e")
        seed_gate = gate_mod.gate_model(config, repo, int(seed["model"]["id"]),
                                        evaluation=seed_eval["metrics"], actor="e2e")
        gate_mod.promote_model(config, repo, int(seed["model"]["id"]), actor="e2e")
        package = export_mod.export_onnx(config, repo, int(seed["model"]["id"]), imgsz=args.imgsz,
                                         verify=True, parity_images=3, actor="e2e")
        prelabel = prelabel_tasks(config, repo, limit=len(candidate_images), status="pending",
                                  model=str(seed["model"]["weights_path"]), actor="e2e")
        report["B_种子模型"] = {"模型": f"{seed['model']['name']}:{seed['model']['version']}",
                                "训练 mAP50": seed["train_map50"],
                                "评估 mAP50(官方)": primary_map50(seed_eval["metrics"]),
                                "门禁": seed_gate["gate"]["passed"],
                                "导出包": package["package"]["dir"],
                                "ONNX 一致性": package["parity"]["passed"],
                                "预标注候选": prelabel.candidates,
                                "预标注任务": prelabel.processed,
                                "耗时秒": round(time.time() - started, 1)}

        # ── C. 主动学习队列（不确定性 + 错误驱动 + 多样性）──────────────
        started = time.time()
        queued = active_service.build_queue(
            config, repo, limit=args.budget, strategy="hybrid", status="pending",
            package_dir=package["package"]["dir"], apply=True, actor="e2e")
        # 种子模型很弱时默认口径会"无信号"（如实记录）；再用 opt-in 的 empty_weight 验证写回路径。
        # empty_weight>0 表示"模型一个框都没出的图也算不确定"（很可能漏检），是主动学习的常用变体。
        queued_signal = active_service.build_queue(
            config, repo, limit=args.budget, strategy="hybrid", status="pending",
            package_dir=package["package"]["dir"], apply=True, actor="e2e", empty_weight=0.5)
        active_view = repo.list_tasks(status="pending", strategy="active", limit=500)
        report["C_主动学习队列"] = {
            "候选池": queued.candidates, "选中": queued.selected,
            "更新优先级": queued.updated_priorities, "理由分布": queued.summary.get("reasons"),
            "分数": queued.summary.get("score"), "弱类": list((queued.summary.get("weak_classes") or {})),
            "高混淆类": list((queued.summary.get("confusion_weights") or {})),
            "margin 证据": (queued.margin_source or {}).get("scored_images"),
            "active 视图任务数": len(active_view), "产物": queued.artifact,
            "提示": queued.summary.get("notes"), "耗时秒": round(time.time() - started, 1),
            "启用 empty_weight 后": {"更新优先级": queued_signal.updated_priorities,
                                     "理由分布": queued_signal.summary.get("reasons"),
                                     "分数": queued_signal.summary.get("score")}}
        checks["C 主动学习队列落库（run + 明细 + active 视图）"] = (
            queued.selected == args.budget and len(active_view) == args.budget
            and queued.run_id is not None and Path(str(queued.artifact)).exists())
        checks["C 有信号时优先级写回生效"] = (
            queued_signal.updated_priorities > 0
            and (queued_signal.summary.get("score") or {}).get("max", 0) > 0)

        # ── D. A/B 对比：同样 40 张新标注预算 ─────────────────────────────
        import random

        seed_ids = [int(row["id"]) for row in seed_images]
        active_ids = [int(item["image_id"]) for item in queued_signal.items][:args.budget]
        remaining = [int(row["id"]) for row in candidate_images if int(row["id"]) not in set(active_ids)]
        rng = random.Random(11)
        baseline_ids = rng.sample(remaining, min(args.budget, len(remaining)))
        pool_baseline = sorted(set(seed_ids) | set(baseline_ids))
        pool_active = sorted(set(seed_ids) | set(active_ids))
        annotate_images(repo, config, [row for row in candidate_images if int(row["id"]) in set(pool_active) | set(pool_baseline)], by_sha)
        started = time.time()
        baseline_trained = build_and_train(config, repo, "ds-m5-ab-baseline", pool_baseline,
                                           epochs=args.epochs, imgsz=args.imgsz, weights=weights,
                                           model_name="yolo11n-m5-ab-baseline")
        active_trained = build_and_train(config, repo, "ds-m5-ab-active", pool_active,
                                         epochs=args.epochs, imgsz=args.imgsz, weights=weights,
                                         model_name="yolo11n-m5-ab-active")
        baseline_eval = gate_mod.evaluate_model(config, repo, int(baseline_trained["model"]["id"]),
                                                split="test", dataset=eval_dataset,
                                                actor="e2e")["metrics"]
        active_eval = gate_mod.evaluate_model(config, repo, int(active_trained["model"]["id"]),
                                              split="test", dataset=eval_dataset, actor="e2e")["metrics"]
        record = active_scoring.budget_effect(baseline_eval, active_eval,
                                              labeled_images=len(pool_baseline),
                                              active_labeled_images=len(pool_active))
        record["models"] = {"baseline": f"{baseline_trained['model']['name']}:{baseline_trained['model']['version']}",
                            "active": f"{active_trained['model']['name']}:{active_trained['model']['version']}"}
        record["selection"] = {"active_reasons": queued_signal.summary.get("reasons"),
                               "active_scores": queued_signal.summary.get("score"),
                               "active_ids": active_ids[:10], "baseline_ids": baseline_ids[:10]}
        record["evaluation"] = {"dataset": eval_dataset["name"], "split": "test",
                                "manifest_hash": eval_dataset["manifest_hash"],
                                "baseline": {"map50": primary_map50(baseline_eval),
                                             "per_class": active_scoring.__dict__ and None},
                                "active": {"map50": primary_map50(active_eval)}}
        record["evaluation"]["baseline"].pop("per_class", None)
        reports_dir = data_dir / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        (reports_dir / "active-comparison.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        report["D_单位标注量对比"] = {**record, "耗时秒": round(time.time() - started, 1)}
        checks["D 对比实验记录完整（同一测试集、同预算）"] = (
            "delta_map50" in record and record["labeled_images"]["baseline"] == len(pool_baseline)
            and (reports_dir / "active-comparison.json").exists())

        # ── E. 新增类别全链路（不改代码）─────────────────────────────────
        started = time.time()
        added = run_cli(["classes", "add", "--code", "water_puddle", "--zh", "积水",
                         "--en", "Water Puddle", "--order", "10"], data_dir=data_dir)
        listed = run_cli(["classes", "list"], data_dir=data_dir)
        new_class_pool = [row for row in candidate_images[:60]]
        remapped = annotate_images(repo, config, new_class_pool, by_sha,
                                   remap={"garbage": "water_puddle"}, actor="e2e")
        new_trained = build_and_train(config, repo, "ds-m5-6cls",
                                      [int(row["id"]) for row in new_class_pool],
                                      epochs=max(8, args.epochs // 2), imgsz=args.imgsz,
                                      weights=weights, model_name="yolo11n-m5-6cls")
        new_eval = gate_mod.evaluate_model(config, repo, int(new_trained["model"]["id"]), split="test",
                                           dataset=eval_dataset, actor="e2e")
        gate_mod.gate_model(config, repo, int(new_trained["model"]["id"]),
                            evaluation=new_eval["metrics"], actor="e2e")
        new_package = export_mod.export_onnx(config, repo, int(new_trained["model"]["id"]),
                                             imgsz=args.imgsz, verify=True, parity_images=3, actor="e2e")
        manifest = json.loads((Path(new_package["package"]["dir"]) / "manifest.json").read_text(encoding="utf-8"))
        report["E_新增类别"] = {
            "classes add 退出码": added.returncode,
            "classes list 含新类": "water_puddle" in listed.stdout,
            "新类标注框": remapped, "数据集划分": new_trained["splits"],
            "清单哈希": new_trained["manifest_hash"],
            "包内 labels": manifest["labels"], "nc": manifest["nc"],
            "ONNX 一致性": new_package["parity"]["passed"],
            "耗时秒": round(time.time() - started, 1)}
        checks["E 新增类别不改代码跑通全链路"] = (
            added.returncode == 0 and "water_puddle" in listed.stdout
            and "water_puddle" in manifest["labels"] and manifest["nc"] == len(manifest["labels"])
            and new_package["parity"]["passed"])

        # ── F. 门禁连续两次不通过 → 告警 ─────────────────────────────────
        # 门禁是"相对现役模型"的比较，且只防"变差"：种子模型在评估集上 mAP50=0.0，
        # 拿它当基线和劣化权重（也是 0.0）比不出差异。因此先把 D 的主动学习模型（明显更好）
        # 过门禁并提升为 production，再用劣化候选验证"连续两次被拒 → 告警"。
        ab_gate = gate_mod.gate_model(config, repo, int(active_trained["model"]["id"]),
                                      evaluation=active_eval, actor="e2e")
        promoted = gate_mod.promote_model(config, repo, int(active_trained["model"]["id"]), actor="e2e")
        report["F_基线"] = {"production": f"{promoted['weights_path']}".split("/")[-3],
                            "seed_map50": primary_map50(seed_eval["metrics"]),
                            "active_map50": primary_map50(active_eval),
                            "A/B 门禁": ab_gate["gate"]["passed"],
                            "提升": promoted["status"]}
        degraded = config.weights_dir / "m5-degraded.pt"
        if not degraded.exists():
            shutil.copy2(weights, degraded)
        for version in ("v1", "v2"):
            candidate = repo.upsert_model_version(name="yolo11n-m5-degraded", version=version,
                                                 status="candidate", weights_path=str(degraded),
                                                 dataset_id=int(eval_dataset["id"]))
            try:
                gate_mod.validate_model(config, repo, int(candidate["id"]), split="test", actor="e2e")
                failed = False
            except Exception:  # noqa: BLE001 - 期望门禁拒绝
                failed = True
            report.setdefault("F_门禁", {})[version] = {"被拒": failed}
        alerts = run_cli(["active", "alerts", "--threshold", "2"], data_dir=data_dir)
        alerts_payload = active_service.gate_alerts(config, repo, threshold=2)
        report["F_门禁"] = {**report.get("F_门禁", {}), "告警": alerts_payload["alerts"],
                            "CLI 退出码": alerts.returncode}
        checks["F 连续两次门禁失败触发规范复查告警"] = (
            alerts.returncode == 5 and any(alert["model"] == "yolo11n-m5-degraded"
                                           for alert in alerts_payload["alerts"]))

        # ── G. 统计报表 ──────────────────────────────────────────────────
        markdown = run_cli(["report", "--format", "markdown", "--days", "30"], data_dir=data_dir)
        geojson = run_cli(["report", "--format", "gis-geojson"], data_dir=data_dir)
        gis_csv = run_cli(["report", "--format", "gis-csv"], data_dir=data_dir)
        try:
            geo = json.loads(geojson.stdout)
        except json.JSONDecodeError:
            geo = {}
        report["G_报表"] = {
            "markdown 退出码": markdown.returncode,
            "markdown 含趋势/批次/覆盖": all(key in markdown.stdout for key in ("趋势", "批次", "类别")),
            "GeoJSON 类型": geo.get("type"), "要素数": len(geo.get("features") or []),
            "CSV 表头": gis_csv.stdout.splitlines()[0] if gis_csv.stdout else "",
            "markdown 前 3 行": markdown.stdout.splitlines()[:3]}
        checks["G 报表（趋势/批次/GIS）可用"] = (
            markdown.returncode == 0 and geo.get("type") == "FeatureCollection"
            and gis_csv.stdout.startswith("image_id,path,captured_at"))
    finally:
        conn.close()

    report["验收"] = checks
    report["总耗时秒"] = round(time.time() - started_all, 1)
    report_path = work / "e2e_m5_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    print(f"\n报告已保存：{report_path}")
    ok = all(checks.values())
    if not args.keep:
        shutil.rmtree(work, ignore_errors=True)
    print("\nM5 验收：", "通过 ✅" if ok else "未通过 ❌", f"（{sum(checks.values())}/{len(checks)} 项）")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
