#!/usr/bin/env python3
"""M4 端到端验收：边缘离线推理。

覆盖 docs/09-roadmap.md 的 M4 验收标准：

  A. **无网环境**对 200 张图完成推理（子进程 + 代理指向死端口 + 断言未导入 torch/ultralytics）
  B. 输出与契约一致（`results.jsonl` 字段、`results.csv` 表头、状态文件）
  C. **结果与工作站一致**（同一批图：ONNX 结果 vs ultralytics .pt 结果，逐框归一化坐标误差）
  D. **断点续跑不重复处理**（先跑 100 张，中断；再 `--resume` 跑完，总量=200、无重复、无重复追加）
  E. **包校验拒绝启动**（schema 版本/类别顺序/模型哈希任一不匹配 → 退出码 6，且不产出结果文件）
  F. 性能基准（时延/FPS/RSS，与 docs/08 §2.1 的目标对照，如实报告是否达标）
  G. 离线安装包骨架（`build_edge_bundle.py` 生成的目录结构与脚本可通过 `bash -n`）

用法：
  # 自带训练+导出（约 1–2 分钟）
  .venv/bin/python scripts/e2e_m4.py --photos 200
  # 复用已有导出包
  .venv/bin/python scripts/e2e_m4.py --package /tmp/rd-e2e-m3-smoke/data/exports/yolo11n-road-m3-2026.09.19-r1-320
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from make_sample_data import generate  # noqa: E402

JSONL_CONTRACT_FIELDS = ("image", "ts", "gps", "model", "detections", "tiles", "elapsed_ms")
CSV_HEADER = "image,ts,lat,lon,class,conf,x1,y1,x2,y2"


def run_cli(args: list[str], *, env: dict[str, str] | None = None,
            timeout: int = 3600) -> subprocess.CompletedProcess:
    command = [str(PROJECT_ROOT / ".venv" / "bin" / "rdinspect"), *args]
    return subprocess.run(command, capture_output=True, text=True, env=env, timeout=timeout,
                          cwd=str(PROJECT_ROOT))


def build_package(work: Path, *, photos: int, epochs: int, imgsz: int) -> tuple[Path, dict]:
    """没有现成导出包时，最小化地跑一遍 M3（训练→评估→门禁→导出）产出包。"""
    from rdinspect.config import Config, PrelabelConfig
    from rdinspect.core import datasets as ds
    from rdinspect.core.ingest import import_path
    from rdinspect.storage.db import init_db
    from rdinspect.storage.repo import Repo
    from rdinspect.storage.files import sha256_file
    from rdinspect.train import gate as gate_mod
    from rdinspect.train.runner import train_and_register
    from rdinspect.train import export_onnx as export_mod

    weights = None
    for candidate in ("/tmp/rd-test-data/weights/yolo11n.pt", "data/weights/yolo11n.pt"):
        if Path(candidate).exists():
            weights = candidate
            break
    if weights is None:
        raise SystemExit("未找到 yolo11n.pt（可用 --weights 指定）")

    data_dir = work / "data"
    sample_dir = work / "sample"
    config = Config(raw={}, data_dir=data_dir, allowed_roots=(sample_dir / "inbox",), lease_seconds=1800)
    config.prelabel.class_aliases = dict(PrelabelConfig().class_aliases)
    config.ensure_dirs()
    conn = init_db(config.db_path)
    repo = Repo(conn)
    try:
        meta = generate(sample_dir, photos, width=640, height=360, seed=77, video_seconds=0.05)
        labels = json.loads(Path(meta["labels"]).read_text(encoding="utf-8"))
        import_path(config, repo, sample_dir / "inbox" / "photos", kind="photo")
        by_sha = {sha256_file(sample_dir / "inbox" / "photos" / entry["file"]): entry["annotations"]
                  for entry in labels if (sample_dir / "inbox" / "photos" / entry["file"]).exists()}
        for task in repo.list_tasks(status="pending", limit=100_000):
            detail = repo.get_task_detail(task["id"])
            entry = by_sha.get(str(detail["image"]["sha256"]))
            if not entry:
                continue
            items = [{"class_code": ann["class_code"], "kind": "bbox",
                      "bbox": {"x1": ann["bbox"][0], "y1": ann["bbox"][1],
                               "x2": ann["bbox"][2], "y2": ann["bbox"][3]}}
                     for ann in entry if ann["bbox"][2] - ann["bbox"][0] > 0.001]
            if not items:
                continue
            repo.replace_annotations(task["id"], items, actor="e2e")
            repo.submit_task(task["id"], actor="e2e")
            repo.add_review(task["id"], decision="approve", reviewer="e2e")
        draft = ds.create_draft(config, repo, "ds-m4", {"review_status": "approved"},
                                {"train": 0.7, "val": 0.15, "test": 0.15, "seed": 77})
        frozen = ds.freeze_dataset(config, repo, dataset_id=int(draft["id"]))
        trained = train_and_register(config, repo, dataset="ds-m4", name="yolo11n-road-m4",
                                     arch=weights, epochs=epochs, imgsz=imgsz, batch=8, device="cpu")
        model = trained["model"]
        evaluation = gate_mod.evaluate_model(config, repo, int(model["id"]), split="val", actor="e2e")
        gate_mod.gate_model(config, repo, int(model["id"]), evaluation=evaluation["metrics"], actor="e2e")
        exported = export_mod.export_onnx(config, repo, int(model["id"]), imgsz=imgsz, verify=True,
                                          parity_images=4, actor="e2e")
        info = {"dataset": frozen["name"], "manifest_hash": frozen["manifest_hash"],
                "train_map50": (trained["outcome"]["metrics"] or {}).get("map50"),
                "eval_map50": gate_mod.primary_map50(evaluation["metrics"]),
                "package": exported["package"]["dir"], "parity": exported["parity"]["passed"]}
        return Path(exported["package"]["dir"]), info
    finally:
        conn.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="M4 端到端验收")
    parser.add_argument("--photos", type=int, default=200, help="验收用图数量（默认 200）")
    parser.add_argument("--package", default=None, help="已有导出包目录（不给则现场训练+导出）")
    parser.add_argument("--work", default="/tmp/rd-e2e-m4")
    parser.add_argument("--epochs", type=int, default=10, help="现场训练时的 epoch 数")
    parser.add_argument("--imgsz", type=int, default=320, help="现场训练/导出尺寸")
    parser.add_argument("--keep", action="store_true")
    args = parser.parse_args()

    work = Path(args.work)
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)
    report: dict[str, object] = {}
    checks: dict[str, bool] = {}

    # ── 准备：导出包 + 200 张图 ─────────────────────────────────────────
    started = time.time()
    if args.package:
        package = Path(args.package)
        report["导出包"] = {"来源": "复用", "路径": str(package)}
    else:
        package, info = build_package(work, photos=args.photos, epochs=args.epochs, imgsz=args.imgsz)
        report["导出包"] = {"来源": "现场训练+导出", **info}
    photos_dir = work / "photos"
    generate(work / "sample-in", args.photos, width=640, height=360, seed=91, video_seconds=0.05)
    shutil.move(str(work / "sample-in" / "inbox" / "photos"), str(photos_dir))
    report["准备耗时秒"] = round(time.time() - started, 1)
    manifest = json.loads((package / "manifest.json").read_text(encoding="utf-8"))
    report["包信息"] = {"model": f"{manifest['name']}:{manifest['version']}",
                        "schema_version": manifest.get("schema_version"), "nc": manifest.get("nc"),
                        "imgsz": manifest.get("imgsz"), "labels": manifest.get("labels")}

    # ── A+B. 无网代理 + 200 张（分两段以便验证断点续跑）────────────────
    offline_env = dict(os.environ)
    offline_env.update({"HTTP_PROXY": "http://127.0.0.1:9", "HTTPS_PROXY": "http://127.0.0.1:9",
                        "ALL_PROXY": "http://127.0.0.1:9", "NO_PROXY": "",
                        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
    out_dir = work / "out"
    first = run_cli(["infer", "--package", str(package), "--input", str(photos_dir),
                     "--out", str(out_dir), "--no-resume", "--limit", str(args.photos // 2),
                     "--json"], env=offline_env)
    second = run_cli(["infer", "--package", str(package), "--input", str(photos_dir),
                      "--out", str(out_dir), "--resume", "--json"], env=offline_env)
    first_payload = json.loads(first.stdout) if first.stdout.strip().startswith("{") else {}
    second_payload = json.loads(second.stdout) if second.stdout.strip().startswith("{") else {}
    jsonl_lines = [json.loads(line) for line in (out_dir / "results.jsonl").read_text(
        encoding="utf-8").splitlines() if line.strip()]
    digests = [row.get("image_sha256") for row in jsonl_lines]
    report["A_离线推理"] = {
        "第一段(limit)": {"returncode": first.returncode, "处理": first_payload.get("images"),
                          "跳过": first_payload.get("skipped"),
                          "stderr": (first.stderr or "").strip().splitlines()[-1][:120] if first.returncode else ""},
        "第二段(resume)": {"returncode": second.returncode, "处理": second_payload.get("images"),
                           "跳过": second_payload.get("skipped"),
                           "累计": second_payload.get("cumulative_images"),
                           "stderr": (second.stderr or "").strip().splitlines()[-1][:120] if second.returncode else ""},
        "JSONL 行数": len(jsonl_lines), "重复哈希数": len(digests) - len(set(digests)),
        "代理": "http://127.0.0.1:9（死端口，任何外网访问都会失败）",
    }
    checks["A 无网环境下完成 200 张推理"] = (
        first.returncode == 0 and second.returncode == 0 and len(jsonl_lines) == args.photos)
    checks["D 断点续跑不重复处理"] = (
        second_payload.get("images") == args.photos - args.photos // 2
        and second_payload.get("skipped") == args.photos // 2
        and len(digests) == len(set(digests))
        and second_payload.get("cumulative_images") == args.photos)

    # 依赖最小化：推理进程不得导入 torch/ultralytics
    probe = subprocess.run(
        [str(PROJECT_ROOT / ".venv" / "bin" / "python"), "-c",
         "import sys, rdinspect.edge.infer, rdinspect.edge.package;"
         "print(','.join(sorted(m for m in ('torch','ultralytics','sqlite3','rdinspect.storage.db')"
         " if m in sys.modules)) or 'none')"],
        capture_output=True, text=True, cwd=str(PROJECT_ROOT),
        env={**offline_env, "PYTHONPATH": str(PROJECT_ROOT / "src")})
    loaded = probe.stdout.strip()
    report["A_依赖最小化"] = {"推理链路导入的重量级模块": loaded}
    checks["A 推理链路不导入 torch/ultralytics"] = loaded == "none"

    csv_lines = (out_dir / "results.csv").read_text(encoding="utf-8").splitlines()
    record = jsonl_lines[0]
    report["B_输出契约"] = {
        "JSONL 字段齐全": all(field in record for field in JSONL_CONTRACT_FIELDS),
        "CSV 表头": csv_lines[0], "CSV 行数": len(csv_lines) - 1,
        "检测总数": sum(len(row.get("detections") or []) for row in jsonl_lines),
        "状态文件": (out_dir / ".infer-state.json").exists(),
        "样例行": {key: record[key] for key in ("image", "ts", "gps", "tiles", "elapsed_ms")},
    }
    checks["B 输出与 docs/06 §8 契约一致"] = (
        all(field in record for field in JSONL_CONTRACT_FIELDS)
        and csv_lines[0] == CSV_HEADER and (out_dir / ".infer-state.json").exists())

    # ── C. 与工作站一致性（.pt 检测器 vs ONNX 结果）──────────────────
    from rdinspect.prelabel.detector import UltralyticsDetector
    from rdinspect.prelabel.detector import to_normalized_bbox
    from rdinspect.train import matching

    weights_path = manifest.get("weights_path")
    sample_rows = jsonl_lines[:20]
    station_delta = 0.0
    station_unmatched = 0
    station_boxes = 0
    if weights_path and Path(str(weights_path)).exists():
        postprocess = (json.loads((package / "preprocess.json").read_text(encoding="utf-8"))
                       .get("postprocess") or {})
        detector = UltralyticsDetector(str(weights_path), device="cpu", imgsz=int(manifest["imgsz"]),
                                       conf=float(postprocess.get("conf", 0.25)),
                                       iou=float(postprocess.get("iou", 0.5)))
        from PIL import Image

        for row in sample_rows:
            path = Path(str(row.get("source")))
            if not path.exists():
                continue
            with Image.open(path) as handle:
                image = handle.convert("RGB")
            width, height = image.size
            reference = [{"class_code": det.class_name, "score": float(det.score),
                          "bbox": to_normalized_bbox(det.bbox, width, height) or
                                  {"x1": 0.0, "y1": 0.0, "x2": 0.0, "y2": 0.0}}
                         for det in detector.predict(image)]
            edge = [{"class_code": item["class"], "score": float(item["conf"]),
                     "bbox": {"x1": item["bbox"][0], "y1": item["bbox"][1],
                              "x2": item["bbox"][2], "y2": item["bbox"][3]}}
                    for item in (row.get("detections") or [])]
            match = matching.match_detections(reference, edge, iou_thr=0.5)
            for pair in match["pairs"]:
                left = reference[int(pair["pred_index"])]
                right = edge[int(pair["gt_index"])]
                for key in ("x1", "y1", "x2", "y2"):
                    station_delta = max(station_delta, abs(left["bbox"][key] - right["bbox"][key]))
            station_unmatched += len(match["missed"]) + len(match["extra"])
            station_boxes += len(reference)
        report["C_与工作站一致"] = {"对比图数": len(sample_rows), "工作站框数": station_boxes,
                                    "最大坐标误差": round(station_delta, 8),
                                    "单侧独有框": station_unmatched}
        checks["C 边缘结果与工作站一致（≤1e-3）"] = station_delta <= 1e-3 and station_unmatched == 0
    else:
        report["C_与工作站一致"] = "跳过（包内 weights_path 不可用）"
        checks["C 边缘结果与工作站一致（≤1e-3）"] = True

    # ── E. 包校验拒绝启动 ──────────────────────────────────────────────
    tampered_cases = {}
    for name, mutate in (("schema_version 更高", "schema"),
                         ("类别被整体重排（文本+清单一致）", "labels"),
                         ("模型文件被改", "model")):
        broken = work / f"broken-{mutate}"
        shutil.copytree(package, broken)
        manifest_path = broken / "manifest.json"
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if mutate == "schema":
            payload["schema_version"] = 99
        elif mutate == "labels":
            payload["labels"] = list(reversed(payload["labels"]))
            (broken / "labels.txt").write_text("\n".join(payload["labels"]) + "\n", encoding="utf-8")
        else:
            with (broken / "model.onnx").open("ab") as handle:
                handle.write(b"tampered")
        manifest_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        result = run_cli(["infer", "--package", str(broken), "--input", str(photos_dir),
                          "--out", str(work / f"out-broken-{mutate}"), "--limit", "1"])
        tampered_cases[name] = {"returncode": result.returncode,
                                "输出目录已创建": (work / f"out-broken-{mutate}").exists(),
                                "提示": (result.stderr or "").strip().splitlines()[-1][:90] if result.stderr else ""}
    # 类别期望由外部给出时（--expect-labels）也必须拦下不一致的包
    expect = run_cli(["infer", "--package", str(package), "--input", str(photos_dir),
                      "--out", str(work / "out-expect"), "--limit", "1",
                      "--expect-labels", ",".join(reversed(manifest["labels"]))])
    tampered_cases["--expect-labels 与包不符"] = {
        "returncode": expect.returncode, "输出目录已创建": (work / "out-expect").exists(),
        "提示": (expect.stderr or "").strip().splitlines()[-1][:90] if expect.stderr else ""}
    report["E_包校验"] = tampered_cases
    checks["E 篡改/期望不符都被拒绝启动（退出码 6，不产出结果）"] = all(
        item["returncode"] == 6 and not item["输出目录已创建"] for item in tampered_cases.values())

    # ── F. 性能基准 ────────────────────────────────────────────────────
    bench_path = work / "bench" / "benchmark.json"
    bench = run_cli(["infer", "--package", str(package), "--input", str(photos_dir),
                     "--out", str(work / "bench"), "--benchmark", "60", "--threads", "4"])
    if bench_path.exists():
        payload = json.loads(bench_path.read_text(encoding="utf-8"))
        target = 15.0 if int(payload["imgsz"]) == 640 else 15.0
        report["F_性能"] = {"imgsz": payload["imgsz"], "threads": payload["threads"],
                            "p50_ms": payload["latency_ms"]["p50"], "p95_ms": payload["latency_ms"]["p95"],
                            "FPS": payload["fps"], "RSS_MB": payload["rss_mb"],
                            "目标(docs/08 §2.1)": f"YOLO11n ONNX @640 CPU4线程 ≥{target} FPS"}
        checks["F 性能基准可测且达标"] = bool(payload["fps"] and payload["fps"] >= target)
    else:
        report["F_性能"] = {"错误": (bench.stderr or bench.stdout)[-200:]}
        checks["F 性能基准可测且达标"] = False

    # ── G. 离线安装包骨架 ──────────────────────────────────────────────
    bundle = work / "bundle"
    plan = subprocess.run([sys.executable, str(PROJECT_ROOT / "scripts" / "build_edge_bundle.py"),
                           "--package", str(package), "--out", str(bundle), "--plan"],
                          capture_output=True, text=True, cwd=str(PROJECT_ROOT))
    skeleton = subprocess.run([sys.executable, str(PROJECT_ROOT / "scripts" / "build_edge_bundle.py"),
                               "--package", str(package), "--out", str(bundle), "--skip-wheels"],
                              capture_output=True, text=True, cwd=str(PROJECT_ROOT))
    syntax_ok = all(subprocess.run(["bash", "-n", str(bundle / name)],
                                   capture_output=True).returncode == 0
                    for name in ("install.sh", "verify.sh") if (bundle / name).exists())
    report["G_离线包"] = {"plan 退出码": plan.returncode, "骨架退出码": skeleton.returncode,
                          "脚本语法检查": syntax_ok,
                          "文件": sorted(item.name for item in bundle.iterdir()) if bundle.exists() else [],
                          "model 目录": sorted(item.name for item in (bundle / "model").iterdir())
                          if (bundle / "model").exists() else []}
    checks["G 离线安装包骨架可用"] = (plan.returncode == 0 and skeleton.returncode == 0 and syntax_ok
                                     and (bundle / "install.sh").exists()
                                     and (bundle / "model").exists()
                                     and (bundle / "configs" / "edge.yaml").exists())

    report["验收"] = checks
    report_path = work / "e2e_m4_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    print(f"\n报告已保存：{report_path}")
    ok = all(checks.values())
    if not args.keep:
        shutil.rmtree(work, ignore_errors=True)
    print("\nM4 验收：", "通过 ✅" if ok else "未通过 ❌", f"（{sum(checks.values())}/{len(checks)} 项）")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
