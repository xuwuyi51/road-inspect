"""三种标注格式的导出/导入与往返校验（YOLO / COCO / LabelMe）。

约定（docs/format-samples/README.md）：
  * 类别以「名称」为唯一键；数字索引只在导出时由 class_order 决定；
  * 内部坐标恒为归一化 bbox；像素换算依赖图像宽高；
  * 往返校验要求类别与坐标零误差（YOLO 文本保留 6 位小数，像素保留 2 位）。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Sequence

from .geometry import (
    bbox_from_coco,
    bbox_from_labelme,
    bbox_to_coco,
    bbox_to_labelme,
    class_index_map,
    parse_yolo_line,
    yolo_line,
)

#: 导出时可携带的来源标记（COCO→attributes，LabelMe→flags）
SOURCE_KEYS = ("source", "score", "difficult")


def _annotation_payload(ann: dict[str, Any]) -> dict[str, Any]:
    payload = {"class_code": ann["class_code"], "bbox": ann["bbox"]}
    for key in SOURCE_KEYS:
        if key in ann and ann[key] is not None:
            payload[key] = ann[key]
    return payload


# ─────────────────────────────── YOLO ───────────────────────────────
def export_yolo(items: Sequence[dict[str, Any]], classes: Sequence[dict[str, Any]], out_dir: Path,
                *, copy_images: bool = True, splits: Sequence[str] = ("train", "val", "test")) -> dict[str, Any]:
    """导出 Ultralytics YOLO 目录结构。

    @param items - [{split, image:{path,width,height,sha256}, annotations:[{class_code,bbox}]}]
    """
    index_map = class_index_map(classes)
    names = [cls["code"] for cls in sorted(classes, key=lambda c: (c["order_index"], c["code"]))]
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = {split: 0 for split in splits}
    label_count = 0
    for item in items:
        split = item["split"]
        if split not in counts:
            counts[split] = 0
        image = item["image"]
        source = Path(item["abs_path"]) if item.get("abs_path") else Path(image["path"])
        stem = Path(image["path"]).stem
        target_image = out_dir / "images" / split / source.name
        target_label = out_dir / "labels" / split / f"{stem}.txt"
        target_image.parent.mkdir(parents=True, exist_ok=True)
        target_label.parent.mkdir(parents=True, exist_ok=True)
        if copy_images:
            if not target_image.exists():
                try:
                    target_image.hardlink_to(source)  # 同分区硬链接：零拷贝导出
                except Exception:
                    import shutil

                    shutil.copy2(source, target_image)
        lines = [yolo_line(index_map[ann["class_code"]], ann["bbox"]) for ann in item["annotations"]]
        target_label.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        label_count += len(lines)
        counts[split] += 1
    data_yaml = out_dir / "data.yaml"
    data_yaml.write_text(
        "path: .\ntrain: images/train\nval: images/val\ntest: images/test\n"
        f"nc: {len(names)}\nnames: {json.dumps(names, ensure_ascii=False)}\n",
        encoding="utf-8",
    )
    return {"format": "yolo", "root": str(out_dir), "images": counts, "labels": label_count,
            "classes": names, "data_yaml": str(data_yaml)}


def parse_yolo_file(text: str) -> list[tuple[str, dict[str, float]]]:
    """解析 YOLO txt（返回 [(class_index_str, bbox)]，类别需调用方按 class_order 映射）。"""
    rows: list[tuple[str, dict[str, float]]] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        class_index, bbox = parse_yolo_line(line)
        rows.append((str(class_index), bbox))
    return rows


# ─────────────────────────────── COCO ───────────────────────────────
def export_coco(items: Sequence[dict[str, Any]], classes: Sequence[dict[str, Any]], out_file: Path,
                *, copy_images: bool = True, image_root: Path | None = None) -> dict[str, Any]:
    ordered = sorted(classes, key=lambda c: (c["order_index"], c["code"]))
    category_id = {cls["code"]: index + 1 for index, cls in enumerate(ordered)}
    coco: dict[str, Any] = {
        "info": {"description": "road-inspect export (COCO)", "version": "1.0"},
        "licenses": [],
        "images": [],
        "categories": [{"id": index + 1, "name": cls["code"], "supercategory": "crack" if cls.get("is_crack") else "defect"}
                       for index, cls in enumerate(ordered)],
        "annotations": [],
    }
    ann_id = 1
    for index, item in enumerate(items, start=1):
        image = item["image"]
        coco["images"].append({
            "id": index, "file_name": Path(image["path"]).name,
            "width": image["width"], "height": image["height"],
            "date_captured": image.get("captured_at"),
            "gps_lat": image.get("gps_lat"), "gps_lon": image.get("gps_lon"),
        })
        for ann in item["annotations"]:
            bbox_px = bbox_to_coco(ann["bbox"], image["width"], image["height"])
            coco["annotations"].append({
                "id": ann_id, "image_id": index, "category_id": category_id[ann["class_code"]],
                "bbox": bbox_px, "area": round(bbox_px[2] * bbox_px[3], 2), "iscrowd": 0,
                "attributes": {k: ann[k] for k in SOURCE_KEYS if ann.get(k) is not None},
            })
            ann_id += 1
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(coco, ensure_ascii=False, indent=2), encoding="utf-8")
    if copy_images and image_root is not None:
        target_dir = out_file.parent / "images"
        target_dir.mkdir(parents=True, exist_ok=True)
        for item in items:
            source = Path(item["abs_path"])
            if source.exists() and not (target_dir / source.name).exists():
                try:
                    (target_dir / source.name).hardlink_to(source)
                except Exception:
                    import shutil

                    shutil.copy2(source, target_dir / source.name)
    return {"format": "coco", "file": str(out_file), "images": len(coco["images"]),
            "labels": len(coco["annotations"]), "categories": [c["name"] for c in coco["categories"]]}


def import_coco(data: dict[str, Any], classes: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """把 COCO json 解析为内部结构（按名称匹配类别）。"""
    code_by_name = {cls["code"]: cls["code"] for cls in classes}
    categories = {cat["id"]: cat["name"] for cat in data.get("categories", [])}
    images = {img["id"]: img for img in data.get("images", [])}
    rows: dict[str, dict[str, Any]] = {}
    for ann in data.get("annotations", []):
        image = images.get(ann["image_id"])
        if image is None:
            continue
        name = categories.get(ann["category_id"], "")
        if name not in code_by_name:
            continue
        key = image["file_name"]
        entry = rows.setdefault(key, {"file_name": key, "width": image["width"],
                                      "height": image["height"], "annotations": []})
        entry["annotations"].append({
            "class_code": name,
            "bbox": bbox_from_coco(ann["bbox"], image["width"], image["height"]),
            "source": (ann.get("attributes") or {}).get("source"),
        })
    return list(rows.values())


# ─────────────────────────────── LabelMe ───────────────────────────────
def export_labelme(items: Sequence[dict[str, Any]], classes: Sequence[dict[str, Any]], out_dir: Path,
                   *, copy_images: bool = False) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    label_total = 0
    for item in items:
        image = item["image"]
        stem = Path(image["path"]).stem
        shapes = []
        for ann in item["annotations"]:
            shapes.append({
                "label": ann["class_code"],
                "points": bbox_to_labelme(ann["bbox"], image["width"], image["height"]),
                "group_id": None, "description": None, "shape_type": "rectangle",
                "flags": {k: ann[k] for k in SOURCE_KEYS if ann.get(k) is not None},
            })
        payload = {
            "version": "5.4.1", "flags": {}, "shapes": shapes,
            "imagePath": Path(image["path"]).name, "imageData": None,
            "imageWidth": image["width"], "imageHeight": image["height"],
            "capturedAt": image.get("captured_at"),
            "gps": {"lat": image.get("gps_lat"), "lon": image.get("gps_lon")},
        }
        (out_dir / f"{stem}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                                              encoding="utf-8")
        written += 1
        label_total += len(shapes)
    return {"format": "labelme", "root": str(out_dir), "images": written, "labels": label_total}


def import_labelme(data: dict[str, Any], classes: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """把单个 LabelMe json 解析为内部结构（按名称匹配类别）。"""
    names = {cls["code"] for cls in classes}
    width, height = int(data["imageWidth"]), int(data["imageHeight"])
    annotations = []
    for shape in data.get("shapes", []):
        label = shape.get("label")
        if label not in names or shape.get("shape_type") != "rectangle":
            continue
        annotations.append({
            "class_code": label,
            "bbox": bbox_from_labelme(shape["points"], width, height),
            "source": (shape.get("flags") or {}).get("source"),
        })
    return {"file_name": Path(data.get("imagePath", "")).name, "width": width, "height": height,
            "annotations": annotations}


# ─────────────────────────────── 往返校验 ───────────────────────────────
def roundtrip_check(items: Sequence[dict[str, Any]], classes: Sequence[dict[str, Any]],
                    work_dir: Path, *, tolerance: float = 1e-6) -> dict[str, Any]:
    """导出三格式并回读，校验类别与坐标零误差（M1 验收项）。

    @returns {"ok": bool, "mismatches": [...], "formats": {...}}
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    index_map = class_index_map(classes)
    reverse_index = {str(index): code for code, index in index_map.items()}
    expected: dict[str, dict[str, dict[str, float]]] = {}
    for item in items:
        stem = Path(item["image"]["path"]).stem
        expected[stem] = {}
        for ann in item["annotations"]:
            expected[stem][ann["class_code"]] = ann["bbox"]

    mismatches: list[dict[str, Any]] = []

    def compare(stem: str, code: str, bbox: dict[str, float], where: str) -> None:
        want = expected.get(stem, {}).get(code)
        if want is None:
            mismatches.append({"where": where, "stem": stem, "code": code, "reason": "unexpected-class"})
            return
        for key in ("x1", "y1", "x2", "y2"):
            if abs(want[key] - bbox[key]) > tolerance:
                mismatches.append({"where": where, "stem": stem, "code": code, "key": key,
                                   "expected": want[key], "actual": bbox[key]})
                return

    # YOLO
    yolo_dir = work_dir / "yolo"
    yolo_stats = export_yolo(items, classes, yolo_dir, copy_images=False)
    for label_file in sorted((yolo_dir / "labels").rglob("*.txt")):
        for class_index, bbox in parse_yolo_file(label_file.read_text(encoding="utf-8")):
            compare(label_file.stem, reverse_index.get(class_index, ""), bbox, "yolo")

    # COCO
    coco_file = work_dir / "coco.json"
    export_coco(items, classes, coco_file, copy_images=False)
    for row in import_coco(json.loads(coco_file.read_text(encoding="utf-8")), classes):
        stem = Path(row["file_name"]).stem
        for ann in row["annotations"]:
            compare(stem, ann["class_code"], ann["bbox"], "coco")

    # LabelMe
    labelme_dir = work_dir / "labelme"
    export_labelme(items, classes, labelme_dir)
    for json_file in sorted(labelme_dir.glob("*.json")):
        row = import_labelme(json.loads(json_file.read_text(encoding="utf-8")), classes)
        for ann in row["annotations"]:
            compare(json_file.stem, ann["class_code"], ann["bbox"], "labelme")

    return {"ok": not mismatches, "mismatches": mismatches[:20], "count": len(mismatches),
            "formats": {"yolo": yolo_stats}}
