# 格式样例（Format Samples）

同一张影像（1920×1080，4 个目标、各占一类）在三种格式下的最小样例，用于：

1. 实现阶段的**往返测试**（YOLO ↔ COCO ↔ LabelMe 转换后，类别与坐标误差必须为 0）；
2. 与外部工具（X-AnyLabeling / LabelMe / Roboflow / CVAT）对接时的字段对照。

## 文件

| 文件 | 格式 | 说明 |
|---|---|---|
| `classes.txt` | YOLO 类别表 | 顺序 = `classes.order_index`（0 纵向裂缝、1 横向裂缝、2 坑洞、3 垃圾、4 网裂预留） |
| `sample.yolo.txt` | Ultralytics YOLO | `class cx cy w h`，归一化 |
| `sample.coco.json` | COCO detection | `bbox=[x,y,w,h]` 像素；类别 id 从 1 开始；扩展字段 `attributes.source/score` |
| `sample.labelme.json` | LabelMe / X-AnyLabeling | `shapes[].points` 像素；扩展字段 `flags.source` |

## 目标对照表（三者必须一一对应）

| # | 类别 | 归一化框（x1,y1,x2,y2） | YOLO 行 | COCO bbox（像素） | LabelMe points |
|---|---|---|---|---|---|
| 1 | transverse_crack | 0.10,0.20 → 0.45,0.26 | `1 0.275 0.23 0.35 0.06` | `[192, 216, 672, 64.8]` | `[[192,216],[864,280.8]]` |
| 2 | longitudinal_crack | 0.60,0.10 → 0.66,0.55 | `0 0.63 0.325 0.06 0.45` | `[1152, 108, 115.2, 486]` | `[[1152,108],[1267.2,594]]` |
| 3 | pothole | 0.70,0.60 → 0.88,0.78 | `2 0.79 0.69 0.18 0.18` | `[1344, 648, 345.6, 194.4]` | `[[1344,648],[1689.6,842.4]]` |
| 4 | garbage | 0.20,0.65 → 0.38,0.85 | `3 0.29 0.75 0.18 0.20` | `[384, 702, 345.6, 216]` | `[[384,702],[729.6,918]]` |

## 转换规则（实现约定）

- **类别以名称为唯一键**；数字索引只在导出瞬间由 `class_order` 决定，导入时按名称映射回 code。
- YOLO → 内部：`cx,cy,w,h` → `x1=cx-w/2, y1=cy-h/2, x2=cx+w/2, y2=cy+h/2`（保持归一化）。
- COCO → 内部：`x1=x/W, y1=y/H, x2=(x+w)/W, y2=(y+h)/H`；`W/H` 取 `images[].width/height`。
- LabelMe → 内部：`points` 顺序不保证 `[左上,右下]`，转换时按 `min/max` 规范化。
- 保留 6 位小数（YOLO 文本）与 2 位小数（像素）以通过零误差往返测试。
- 扩展字段（`source`/`score`/`difficult`）在 COCO 放 `attributes`、在 LabelMe 放 `flags`；缺失时按 `source='external'` 处理。

## 校验方式（M1 验收用）

```bash
# 设计阶段：结构可解析
python3 - <<'PY'
import json, pathlib
base = pathlib.Path('docs/format-samples')
yolo = [l.split() for l in base.joinpath('sample.yolo.txt').read_text().splitlines() if l.strip()]
classes = base.joinpath('classes.txt').read_text().split()
coco = json.loads(base.joinpath('sample.coco.json').read_text())
labelme = json.loads(base.joinpath('sample.labelme.json').read_text())
assert len(yolo) == len(coco['annotations']) == len(labelme['shapes']) == 4
assert [c['name'] for c in coco['categories']] == classes
assert 1920 == coco['images'][0]['width'] == labelme['imageWidth']
print('samples ok:', len(yolo), 'objects /', len(classes), 'classes')
PY
```
