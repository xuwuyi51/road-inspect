#!/usr/bin/env python3
"""road-inspect 交付自检（M0 验收）

校验四件事：
  1. db/schema.sql 可建库（表/索引/视图/触发器计数 + 关键约束）
  2. docs/openapi.yaml 结构与 $ref 完整（OpenAPI 3.1 最小校验）
  3. docs/format-samples 三格式一致（YOLO ↔ COCO ↔ LabelMe 归一化坐标零误差）
  4. 配置 YAML 可解析 + 文档内相对链接可达

用法：python3 scripts/check_docs.py
退出码：0 全通过；1 有失败项
"""
from __future__ import annotations

import json
import pathlib
import re
import sqlite3
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
FAILS: list[str] = []
OKS: list[str] = []


def check(name: str, fn) -> None:
    try:
        detail = fn()
        OKS.append(f"{name}: {detail}")
    except Exception as exc:  # noqa: BLE001
        FAILS.append(f"{name}: {type(exc).__name__}: {exc}")


# ── 1. 数据库 DDL ────────────────────────────────────────────────────────────
def check_schema() -> str:
    sql = (ROOT / "db/schema.sql").read_text(encoding="utf-8")
    con = sqlite3.connect(":memory:")
    con.executescript(sql)
    tables = [r[0] for r in con.execute(
        "select name from sqlite_master where type='table' and name not like 'sqlite_%'")]
    views = [r[0] for r in con.execute("select name from sqlite_master where type='view'")]
    triggers = [r[0] for r in con.execute("select name from sqlite_master where type='trigger'")]
    indexes = [r[0] for r in con.execute(
        "select name from sqlite_master where type='index' and name not like 'sqlite_%'")]
    expected = {"classes", "images", "tasks", "annotations", "reviews", "dataset_versions",
                "dataset_items", "runs", "model_versions", "audit_log", "batches", "ingest_items",
                "schema_meta"}
    missing = expected - set(tables)
    if missing:
        raise AssertionError(f"缺少表 {sorted(missing)}")
    if len(views) < 2:
        raise AssertionError("视图不足（期望 v_task_queue / v_class_counts）")
    if len(triggers) < 2:
        raise AssertionError("触发器不足（期望 updated_at 触发器）")
    n_classes = con.execute("select count(*) from classes").fetchone()[0]
    if n_classes < 4:
        raise AssertionError(f"内置类别不足：{n_classes}")
    # CHECK 约束必须生效：bbox 缺坐标应被拒
    con.execute("insert into batches(kind) values('photo')")
    con.execute("insert into images(batch_id,path,sha256,width,height,bytes,source_kind) "
                "values(1,'a.jpg','h1',100,100,10,'photo')")
    con.execute("insert into tasks(image_id) values(1)")
    try:
        con.execute("insert into annotations(task_id,image_id,class_code,kind) "
                    "values(1,1,'pothole','bbox')")
        raise AssertionError("bbox CHECK 约束未生效")
    except sqlite3.IntegrityError:
        pass
    return (f"{len(tables)} 表 / {len(indexes)} 索引 / {len(views)} 视图 / "
            f"{len(triggers)} 触发器 / {n_classes} 类别，CHECK 生效")


# ── 2. OpenAPI ───────────────────────────────────────────────────────────────
def check_openapi() -> str:
    import yaml  # noqa: PLC0415

    raw = (ROOT / "docs/openapi.yaml").read_text(encoding="utf-8")
    doc = yaml.safe_load(raw)
    assert str(doc.get("openapi", "")).startswith("3.1"), "openapi 版本非 3.1"
    for key in ("info", "paths", "components"):
        assert key in doc, f"缺少顶层字段 {key}"
    for key in ("title", "version"):
        assert key in doc["info"], f"info.{key} 缺失"
    ops = 0
    for path, item in doc["paths"].items():
        assert path.startswith("/"), f"路径格式异常 {path}"
        for method, op in item.items():
            if method not in ("get", "post", "put", "patch", "delete", "head", "options"):
                continue
            ops += 1
            assert "responses" in op, f"{method.upper()} {path} 缺 responses"
            for code, resp in op["responses"].items():
                assert re.fullmatch(r"[1-5](\d\d|XX)", str(code)), f"状态码异常 {code}"
                if isinstance(resp, dict) and "$ref" not in resp:
                    assert "description" in resp, f"{method.upper()} {path} {code} 缺 description"
    refs = set(re.findall(r'\$ref:\s*"#/components/([^"]+)"', raw))
    for ref in refs:
        node = doc["components"]
        for part in ref.split("/"):
            node = node.get(part) if isinstance(node, dict) else None
        assert node is not None, f"悬空 $ref: {ref}"
    return (f"{len(doc['paths'])} 路径 / {ops} 操作 / "
            f"{len(doc['components']['schemas'])} schema / {len(refs)} 引用")


# ── 3. 格式样例一致性 ─────────────────────────────────────────────────────────
def _iou_free_equal(a: float, b: float, tol: float = 1e-6) -> bool:
    return abs(a - b) <= tol


def check_samples() -> str:
    base = ROOT / "docs/format-samples"
    classes = base.joinpath("classes.txt").read_text(encoding="utf-8").split()
    yolo_rows = [line.split() for line in
                 base.joinpath("sample.yolo.txt").read_text(encoding="utf-8").splitlines() if line.strip()]
    coco = json.loads(base.joinpath("sample.coco.json").read_text(encoding="utf-8"))
    labelme = json.loads(base.joinpath("sample.labelme.json").read_text(encoding="utf-8"))

    assert len(yolo_rows) == len(coco["annotations"]) == len(labelme["shapes"]) == 4, "对象数不一致"
    assert [c["name"] for c in coco["categories"]] == classes, "COCO 类别与 classes.txt 不一致"

    width = coco["images"][0]["width"]
    height = coco["images"][0]["height"]
    assert width == labelme["imageWidth"] and height == labelme["imageHeight"], "图像尺寸不一致"

    def norm_from_yolo(row):
        idx, cx, cy, w, h = int(row[0]), *map(float, row[1:])
        return classes[idx], (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)

    def norm_from_coco(ann):
        name = next(c["name"] for c in coco["categories"] if c["id"] == ann["category_id"])
        x, y, w, h = ann["bbox"]
        return name, (x / width, y / height, (x + w) / width, (y + h) / height)

    def norm_from_labelme(shape):
        (x1, y1), (x2, y2) = shape["points"]
        return shape["label"], (min(x1, x2) / width, min(y1, y2) / height,
                                max(x1, x2) / width, max(y1, y2) / height)

    yolo_map = {name: box for name, box in map(norm_from_yolo, yolo_rows)}
    coco_map = {name: box for name, box in map(norm_from_coco, coco["annotations"])}
    lme_map = {name: box for name, box in map(norm_from_labelme, labelme["shapes"])}
    assert set(yolo_map) == set(coco_map) == set(lme_map), "类别集合不一致"

    for name in yolo_map:
        for other, label in ((coco_map, "COCO"), (lme_map, "LabelMe")):
            for i in range(4):
                if not _iou_free_equal(yolo_map[name][i], other[name][i]):
                    raise AssertionError(
                        f"{name} 第 {i} 个坐标不一致: YOLO={yolo_map[name][i]:.6f} {label}={other[name][i]:.6f}")
    return f"4 类 × 3 格式往返一致（{len(classes)} 类别表）"


# ── 4. 配置与文档链接 ────────────────────────────────────────────────────────
def check_configs() -> str:
    import yaml  # noqa: PLC0415

    files = sorted((ROOT / "configs").glob("*.yaml"))
    assert len(files) >= 4, f"配置不足：{[f.name for f in files]}"
    keys = {}
    for path in files:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert isinstance(data, dict) and data, f"{path.name} 为空或非映射"
        keys[path.name] = len(data)
    return ", ".join(f"{k}({v} 个顶层键)" for k, v in keys.items())


def check_links() -> str:
    pattern = re.compile(r"\[[^\]]+\]\((?!https?://|#|mailto:)([^)]+)\)")
    checked = 0
    for md in list((ROOT / "docs").rglob("*.md")) + [ROOT / "README.md"]:
        text = md.read_text(encoding="utf-8")
        for target in pattern.findall(text):
            target = target.split("#", 1)[0].strip()
            if not target:
                continue
            resolved = (md.parent / target).resolve()
            assert resolved.exists(), f"{md.relative_to(ROOT)} → 断链 {target}"
            checked += 1
    return f"{checked} 条相对链接全部可达"


def main() -> int:
    check("DDL 建库", check_schema)
    check("OpenAPI 结构", check_openapi)
    check("格式样例", check_samples)
    check("配置 YAML", check_configs)
    check("文档链接", check_links)

    print("== 通过 ==")
    for line in OKS:
        print("  ✓", line)
    if FAILS:
        print("== 失败 ==")
        for line in FAILS:
            print("  ✗", line)
        return 1
    docs = sorted((ROOT / "docs").glob("*.md"))
    adrs = sorted((ROOT / "docs/adr").glob("*.md"))
    samples = sorted((ROOT / "docs/format-samples").glob("*"))
    configs = sorted((ROOT / "configs").glob("*.yaml"))
    print(f"\n交付物统计：docs {len(docs)} 份 + ADR {len(adrs)} 份 + 样例 {len(samples)} 个 "
          f"+ DDL 1 + 配置 {len(configs)} 份")
    return 0


if __name__ == "__main__":
    sys.exit(main())
