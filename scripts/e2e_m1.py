#!/usr/bin/env python3
"""M1 端到端验收：合成素材 → 导入 → 标注 → 复核 → 数据集冻结 → 导出 → 三格式往返校验。

对应 docs/09-roadmap.md 的 M1 验收标准：
  * 500 张照片 + 1 段视频导入正确（含去重/抽帧/缩略图/任务生成）
  * 四类各 ≥50 张标注并复核通过
  * 三格式互转零误差
用法：
  python3 scripts/e2e_m1.py --photos 500 --per-class 50
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rdinspect.config import Config  # noqa: E402
from rdinspect.core import datasets as ds  # noqa: E402
from rdinspect.core.formats import roundtrip_check  # noqa: E402
from rdinspect.core.ingest import import_path  # noqa: E402
from rdinspect.storage.db import init_db  # noqa: E402
from rdinspect.storage.repo import Repo  # noqa: E402

sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from make_sample_data import generate  # noqa: E402

CLASSES = ("transverse_crack", "longitudinal_crack", "pothole", "garbage")


def build_config(data_dir: Path, sample_dir: Path) -> Config:
    config = Config(raw={}, data_dir=data_dir, allowed_roots=(sample_dir / "inbox",),
                    thumb_long_edge=256, lease_seconds=1800)
    config.ensure_dirs()
    return config


def main() -> int:
    parser = argparse.ArgumentParser(description="M1 端到端验收")
    parser.add_argument("--photos", type=int, default=500)
    parser.add_argument("--per-class", type=int, default=50)
    parser.add_argument("--work", default="/tmp/rd-e2e")
    parser.add_argument("--keep", action="store_true", help="结束后保留工作目录")
    args = parser.parse_args()

    work = Path(args.work)
    if work.exists():
        shutil.rmtree(work)
    sample_dir = work / "sample"
    data_dir = work / "data"
    report: dict[str, object] = {}
    timings: dict[str, float] = {}

    t0 = time.time()
    generated = generate(sample_dir, args.photos, width=640, height=360, seed=11, video_seconds=4)
    timings["生成素材"] = time.time() - t0

    config = build_config(data_dir, sample_dir)
    conn = init_db(config.db_path)
    repo = Repo(conn)
    try:
        # ── 1. 导入 ──────────────────────────────────────────────────────
        t0 = time.time()
        ingest = import_path(config, repo, sample_dir / "inbox", kind="photo", note="e2e")
        timings["导入"] = time.time() - t0
        dup = import_path(config, repo, sample_dir / "inbox", kind="photo", note="e2e-again")
        report["导入"] = {
            "首次": ingest.as_dict() | {"image_ids": len(ingest.image_ids)},
            "重复导入": {"added": dup.added, "dup_sha": dup.dup_sha},
            "影像总数": repo.count_images(),
            "任务总数": len(repo.list_tasks(status="pending", limit=100000)),
            "缩略图数": len(list(config.thumbs_dir.glob("*.webp"))),
        }

        # ── 2. 标注 + 复核（四类各 ≥ per-class 张）─────────────────────────
        t0 = time.time()
        annotated: dict[str, int] = {code: 0 for code in CLASSES}
        tasks = repo.list_tasks(status="pending", limit=100000)
        boxes = {
            "transverse_crack": {"x1": 0.10, "y1": 0.20, "x2": 0.45, "y2": 0.26},
            "longitudinal_crack": {"x1": 0.60, "y1": 0.10, "x2": 0.66, "y2": 0.55},
            "pothole": {"x1": 0.70, "y1": 0.60, "x2": 0.88, "y2": 0.78},
            "garbage": {"x1": 0.20, "y1": 0.65, "x2": 0.38, "y2": 0.85},
        }
        cursor = 0
        for class_code in CLASSES:
            for _ in range(args.per_class):
                if cursor >= len(tasks):
                    break
                task_id = tasks[cursor]["id"]
                cursor += 1
                repo.replace_annotations(task_id, [
                    {"class_code": class_code, "kind": "bbox", "bbox": boxes[class_code]}])
                repo.submit_task(task_id, actor="e2e")
                repo.add_review(task_id, decision="approve", reviewer="e2e")
                annotated[class_code] += 1
        # 剩余任务标记为跳过，避免进入数据集
        for task in tasks[cursor:]:
            repo.set_task_status(task["id"], "skipped", actor="e2e")
        timings["标注+复核"] = time.time() - t0
        report["标注"] = {"每类张数": annotated, "合计": sum(annotated.values()),
                        "复核统计": repo.review_stats()}

        # ── 3. 数据集草稿 → 冻结 ──────────────────────────────────────────
        t0 = time.time()
        draft = ds.create_draft(config, repo, "ds-e2e-m1", {"review_status": "approved"},
                                {"train": 0.7, "val": 0.15, "test": 0.15, "seed": 42, "group_by": "gps_grid"})
        frozen = ds.freeze_dataset(config, repo, name="ds-e2e-m1",
                                   export_formats=("yolo", "coco", "labelme"))
        timings["冻结+导出"] = time.time() - t0
        report["数据集"] = {
            "草稿统计": draft["stats"],
            "状态": frozen["status"],
            "manifest_hash": frozen["manifest_hash"],
            "导出": {key: value.get("images", value.get("file")) for key, value in frozen["exports"].items()},
            "root": frozen["root_path"],
        }

        # ── 4. 三格式往返零误差 ───────────────────────────────────────────
        rows = ds.fetch_dataset_rows(repo, repo.dataset_items(frozen["id"]))
        abs_rows = [{**row, "abs_path": str(config.abs_data_path(row["image"]["path"]))} for row in rows]
        check = roundtrip_check(abs_rows, repo.list_classes(), work / "roundtrip")
        report["往返校验"] = {"ok": check["ok"], "mismatches": check["count"]}

        # ── 5. 重启后一致性（哈希稳定）───────────────────────────────────
        conn.close()
        conn2 = init_db(config.db_path)
        repo2 = Repo(conn2)
        again = ds.fetch_dataset_rows(repo2, repo2.dataset_items(frozen["id"]))
        order = [cls["code"] for cls in sorted(repo2.list_classes(), key=lambda c: (c["order_index"], c["code"]))]
        report["哈希稳定性"] = ds.compute_manifest_hash(again, order) == frozen["manifest_hash"]
        conn2.close()
        conn = None
    finally:
        if conn is not None:
            conn.close()

    report["耗时(秒)"] = {key: round(value, 2) for key, value in timings.items()}
    report["工作目录"] = str(work)
    print(json.dumps(report, ensure_ascii=False, indent=2))

    ok = (
        report["导入"]["重复导入"]["added"] == 0
        and report["导入"]["重复导入"]["dup_sha"] >= args.photos
        and min(report["标注"]["每类张数"].values()) >= args.per_class
        and report["往返校验"]["ok"] is True
        and report["哈希稳定性"] is True
    )
    if not args.keep:
        shutil.rmtree(work, ignore_errors=True)
    print("\nM1 验收：", "通过 ✅" if ok else "未通过 ❌")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
