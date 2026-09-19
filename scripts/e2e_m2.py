#!/usr/bin/env python3
"""M2 端到端验收：模型辅助标注闭环。

覆盖 docs/09-roadmap.md 的 M2 验收标准：
  A. 批量预标注 200 张完成（确定性假检测器，无需 ML 依赖）
  B. 候选可逐条采纳 / 忽略，质量看板指标正确
  C. 采纳后的标签可冻结进数据集（预标注 → 人工确认 → 训练数据）
  D. 关闭预标注时人工标注流程不受影响（降级为纯人工）
  E. 真机（可选）：用道路病害权重预标注真实路面图，类别正确映射
  F. 真机（可选）：SAM 对裂缝候选生成掩膜并给出宽度/长度派生指标

用法：
  python3 scripts/e2e_m2.py --photos 200
  python3 scripts/e2e_m2.py --detector-weights /tmp/weights/yolo12s_rdd.pt \
                            --sam-weights /tmp/weights/sam2.1_t.pt \
                            --real-dir /tmp/real-road-images
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from make_sample_data import generate  # noqa: E402
from rdinspect.config import Config  # noqa: E402
from rdinspect.core import datasets as ds  # noqa: E402
from rdinspect.core.ingest import import_path  # noqa: E402
from rdinspect.prelabel.detector import Detection, DetectorUnavailable  # noqa: E402
from rdinspect.prelabel.metrics import prelabel_metrics  # noqa: E402
from rdinspect.prelabel.sam import SamMasker  # noqa: E402
from rdinspect.prelabel.service import attach_mask_for_annotation, build_detector, prelabel_tasks  # noqa: E402
from rdinspect.storage.db import init_db  # noqa: E402
from rdinspect.storage.repo import Repo  # noqa: E402


class ScriptedDetector:
    """确定性假检测器：按图内固定比例给一个横向裂缝框（用于 A–D 的可复现验收）。"""

    names = {0: "D10"}
    device = "cpu"
    imgsz = 640
    conf = 0.25
    iou = 0.5

    def __init__(self, weights: str = "scripted.pt", box=(0.15, 0.25, 0.55, 0.45), score: float = 0.8):
        self.weights = weights
        self.box = box
        self.score = score

    def predict(self, image: Image.Image) -> list[Detection]:
        width, height = image.size
        x1, y1, x2, y2 = self.box
        return [Detection(0, "D10", self.score,
                          (x1 * width, y1 * height, x2 * width, y2 * height))]


def build_config(data_dir: Path, allowed_root: Path) -> Config:
    config = Config(raw={}, data_dir=data_dir, allowed_roots=(allowed_root,), lease_seconds=1800)
    # 使用出厂默认别名表（含 d00/d10/d20/d40/d44 与英文名），保证验收与真实部署一致
    from rdinspect.config import PrelabelConfig

    config.prelabel.class_aliases = dict(PrelabelConfig().class_aliases)
    config.ensure_dirs()
    return config


def main() -> int:
    parser = argparse.ArgumentParser(description="M2 端到端验收")
    parser.add_argument("--photos", type=int, default=200)
    parser.add_argument("--work", default="/tmp/rd-e2e-m2")
    parser.add_argument("--detector-weights", default=None, help="道路病害检测权重（.pt）")
    parser.add_argument("--sam-weights", default=None, help="SAM 权重（.pt）")
    parser.add_argument("--real-dir", default=None, help="真实路面图目录（可选）")
    parser.add_argument("--keep", action="store_true")
    args = parser.parse_args()

    work = Path(args.work)
    if work.exists():
        shutil.rmtree(work)
    sample_dir = work / "sample"
    config = build_config(work / "data", sample_dir / "inbox")
    conn = init_db(config.db_path)
    repo = Repo(conn)
    report: dict[str, object] = {}
    checks: dict[str, bool] = {}
    try:
        # ── A. 批量预标注 200 张 ─────────────────────────────────────────
        started = time.time()
        generate(sample_dir, args.photos, width=640, height=360, seed=21, video_seconds=0.1)
        ingest = import_path(config, repo, sample_dir / "inbox" / "photos", kind="photo")
        prelabel = prelabel_tasks(config, repo, limit=args.photos, detector=ScriptedDetector(), actor="e2e")
        report["A_批量预标注"] = {
            "导入": ingest.added, "请求": prelabel.requested, "处理": prelabel.processed,
            "候选": prelabel.candidates, "耗时秒": round(time.time() - started, 1),
        }
        checks["A 批量预标注完成"] = (
            ingest.added >= args.photos and prelabel.processed == args.photos
            and prelabel.candidates == args.photos
        )

        # ── B. 逐条采纳 / 忽略 + 质量看板 ───────────────────────────────
        tasks = repo.list_tasks(status="prelabeled", limit=10_000)
        adopt_count = int(len(tasks) * 0.6)
        ignore_count = int(len(tasks) * 0.2)
        for task in tasks[:adopt_count]:                       # 采纳：转 model_edited 并微调位置
            repo.adopt_model_candidates(task["id"], actor="e2e")
            repo.replace_annotations(task["id"], [{
                "class_code": "transverse_crack", "kind": "bbox",
                "bbox": {"x1": 0.16, "y1": 0.26, "x2": 0.56, "y2": 0.46}, "source": "human"}])
            repo.submit_task(task["id"], actor="e2e")
            repo.add_review(task["id"], decision="approve", reviewer="e2e")
        ignored = 0
        for task in tasks[adopt_count:adopt_count + ignore_count]:
            for candidate in repo.list_annotations(task_id=task["id"]):
                if candidate["source"] == "model" and repo.delete_annotation(candidate["id"], actor="e2e"):
                    ignored += 1
        metrics = prelabel_metrics(repo)
        report["B_采纳与忽略"] = {"采纳任务": adopt_count, "忽略候选": ignored, "看板": metrics}
        checks["B 候选可采纳/忽略且看板正确"] = (
            metrics["adopted"] == adopt_count and metrics["ignored"] == ignored
            and metrics["adoption_rate"] is not None and metrics["model_human_iou_mean"] is not None
            and 0.0 < metrics["model_human_iou_mean"] <= 1.0
        )

        # ── C. 采纳结果可冻结进数据集 ──────────────────────────────────
        draft = ds.create_draft(config, repo, "ds-e2e-m2", {"review_status": "approved"},
                                {"train": 0.8, "val": 0.1, "test": 0.1, "seed": 5, "group_by": "none"})
        frozen = ds.freeze_dataset(config, repo, name="ds-e2e-m2", export_formats=("yolo",))
        label_files = list((config.datasets_dir / "ds-e2e-m2" / "labels").rglob("*.txt"))
        non_empty = [path for path in label_files if path.read_text(encoding="utf-8").strip()]
        expected = repo.conn.execute(
            """SELECT COUNT(*) AS n FROM tasks t JOIN images i ON i.id = t.image_id
               WHERE t.status = 'approved' AND i.duplicate_of IS NULL""").fetchone()["n"]
        report["C_冻结"] = {"采纳任务": adopt_count, "可入集样本(非近似重复)": expected,
                        "数据集样本": draft["stats"]["images"], "标注文件": len(label_files),
                        "非空标注": len(non_empty), "manifest": frozen["manifest_hash"][:16]}
        checks["C 采纳标签进入数据集"] = (
            draft["stats"]["images"] == expected and len(non_empty) == expected and expected > 0
        )

        # ── D. 关闭预标注不影响人工标注 ────────────────────────────────
        config.prelabel.enabled = False
        disabled_ok = False
        try:
            prelabel_tasks(config, repo, limit=1, actor="e2e")
        except DetectorUnavailable:
            disabled_ok = True
        config.prelabel.enabled = True
        # 未采纳也未忽略的任务仍可用纯人工方式标注（预标注只是候选）
        leftovers = repo.list_tasks(status="prelabeled", limit=1)
        manual_ok = False
        if leftovers:
            task_id = leftovers[0]["id"]
            result = repo.replace_annotations(task_id, [{
                "class_code": "pothole", "kind": "bbox",
                "bbox": {"x1": 0.2, "y1": 0.2, "x2": 0.5, "y2": 0.5}}])
            manual_ok = result["added"] == 1 and repo.submit_task(task_id) is not None
        report["D_降级"] = {"关闭预标注抛不可用": disabled_ok, "人工标注仍可用": manual_ok}
        checks["D 关闭预标注不影响人工流程"] = disabled_ok and manual_ok

        # ── E. 真机：道路病害权重预标注真实图 ──────────────────────────
        if args.detector_weights and args.real_dir:
            real_root = Path(args.real_dir).resolve()
            config.allowed_roots = tuple(config.allowed_roots) + (real_root,)
            real_batch = import_path(config, repo, real_root, kind="photo", note="e2e-real")
            real_task_ids = [task["id"] for task in repo.list_tasks(status="pending", limit=10_000)]
            detector = build_detector(config, args.detector_weights, conf=0.25, device="auto")
            real = prelabel_tasks(config, repo, limit=len(real_task_ids), task_ids=real_task_ids,
                                  detector=detector, actor="e2e")
            placeholders = ",".join("?" * len(real_task_ids)) or "NULL"
            per_class = dict(repo.conn.execute(
                f"""SELECT class_code, COUNT(*) FROM annotations WHERE source='model' AND deleted_at IS NULL
                    AND task_id IN ({placeholders}) GROUP BY class_code""", real_task_ids).fetchall())
            total_detected = real.candidates + real.unmapped
            mapped_ratio = round(real.candidates / total_detected, 3) if total_detected else 0.0
            report["E_真机预标注"] = {"真实图": real_batch.added, "处理": real.processed,
                                  "候选": real.candidates, "未映射": real.unmapped,
                                  "未映射类别": real.unmapped_classes,
                                  "映射率": mapped_ratio, "按类别": per_class,
                                  "检测器": real.model.get("weights")}
            # 模型可能输出本体系外的类别（如 RDD 的 Repair/白线模糊）——丢弃是设计行为，
            # 只要求「有候选 且 映射率过半」，并在报告里列出被丢弃的类别名。
            checks["E 真机模型产出已映射候选"] = real.candidates > 0 and mapped_ratio >= 0.5
        else:
            report["E_真机预标注"] = "跳过（未提供 --detector-weights/--real-dir）"

        # ── F. 真机：SAM 裂缝掩膜 ─────────────────────────────────────
        if args.sam_weights:
            candidate = repo.conn.execute(
                """SELECT id, class_code FROM annotations WHERE source='model' AND deleted_at IS NULL
                   AND class_code IN ('transverse_crack','longitudinal_crack') ORDER BY id LIMIT 1""").fetchone()
            if candidate is None:
                report["F_SAM 掩膜"] = "跳过（没有裂缝候选）"
            else:
                masker = SamMasker(args.sam_weights, device="cpu")
                made = 0
                started = time.time()
                for row in repo.conn.execute(
                        """SELECT id FROM annotations WHERE source='model' AND deleted_at IS NULL
                           AND class_code IN ('transverse_crack','longitudinal_crack')
                           ORDER BY id LIMIT 3""").fetchall():
                    try:
                        attach_mask_for_annotation(config, repo, row["id"], masker=masker)
                        made += 1
                    except Exception as exc:  # noqa: BLE001
                        report.setdefault("F_SAM 错误", []).append(str(exc)[:120])
                masks = repo.mask_annotations(limit=5)
                report["F_SAM 掩膜"] = {
                    "生成数": made, "耗时秒": round(time.time() - started, 1),
                    "掩膜指标": [{"class": m["class_code"], "area": m["mask_metrics"].get("area_ratio"),
                                  "length_px": m["mask_metrics"].get("length_px"),
                                  "width_px": m["mask_metrics"].get("width_px")} for m in masks],
                }
                checks["F SAM 掩膜可用且给出宽度指标"] = made > 0 and all(
                    m["mask_metrics"].get("width_px", 0) > 0 for m in masks)
        else:
            report["F_SAM 掩膜"] = "跳过（未提供 --sam-weights）"
    finally:
        conn.close()

    report["验收"] = checks
    print(json.dumps(report, ensure_ascii=False, indent=2))
    ok = all(checks.values())
    if not args.keep:
        shutil.rmtree(work, ignore_errors=True)
    print("\nM2 验收：", "通过 ✅" if ok else "未通过 ❌", f"（{sum(checks.values())}/{len(checks)} 项）")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
