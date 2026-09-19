# 13 · 主动学习与类别扩展操作手册（M5）

> 面向"标注预算有限，标哪些图最值"和"要加一个新类别"的人。设计依据 [ADR-0007](./adr/0007-active-learning-and-model-gate.md)；
> 上游是 [11-training-guide](./11-training-guide.md)（训练与门禁），报表口径见 [03-data-model](./03-data-model.md)。

## 1. 三类选样策略（ADR-0007 权重）

| 策略 | 权重 | 证据从哪来 | 本实现的落点 |
|---|---|---|---|
| 不确定性 | 50% | 候选框置信度落在 `[0.25, 0.45]`；或 `top1−top2 < 0.1` | 库里 `source='model'` 的预标注候选；传 `--package` 时用 ONNX 重跑拿 top-2 分数 |
| 错误驱动 | 30% | 最近一次评估的混淆矩阵（易混类对）+ 分类别召回（弱类） | `model_versions.metrics_json.evaluation`，无需额外推理 |
| 多样性 | 20% | `images.phash` 汉明距离聚类，每簇最多 1 张 | 按 20% 配额从"未被分数选中"的池子里补代表 |

`priority = 1 + (1−score) × 99`，写回 `tasks.priority`（**数值越小越优先**）；同时写 `active_queue`
明细（得分、理由、证据），标注台可以用 `GET /api/tasks?strategy=active` 只看队列里的任务。

**可选开关 `--empty-weight`**：模型一个框都没出的图也算不确定（很可能漏检）。默认 0（空路面不浪费预算）。
种子模型很弱时默认口径会"无信号"，此时队列退化为按图像 id 顺序，报告里会**明确提示**而不是假装排过序。

## 2. 用法

```bash
# 1) 先有模型候选：跑一次预标注（或直接用导出包做 margin 打分）
.venv/bin/rdinspect prelabel --limit 200 --model data/weights/<现役或种子权重>.pt

# 2) 生成队列（默认 hybrid；--package 时额外用 ONNX 取 top1−top2 证据）
.venv/bin/rdinspect active queue --limit 100 --strategy hybrid \
    --package data/exports/<name>-<version>-320
.venv/bin/rdinspect active show                      # 看最近一次选样明细与理由
.venv/bin/rdinspect active queue --limit 100 --dry-run   # 只算不写库

# 3) 标注台只看队列：/api/tasks?status=pending&strategy=active
```

| 参数 | 说明 |
|---|---|
| `--strategy` | `hybrid`(默认) / `uncertainty` / `error` / `diversity` / `random`（对比实验基线，固定 seed 可复现） |
| `--conf-lo/--conf-hi` | 不确定性区间，默认 0.25–0.45 |
| `--margin-threshold` | top1−top2 阈值，默认 0.1 |
| `--diversity-ratio` | 多样性配额，默认 0.2 |
| `--phash-hamming` | 聚类阈值，默认 6（越大越宽松） |
| `--scan-limit` | 候选池扫描上限（默认全部待标注任务）；**注意它与 `--limit` 不是一回事**：`--limit` 是选多少张 |

## 3. 效果怎么测（不靠感觉）

`scripts/e2e_m5.py` 做的是**同预算 A/B**：同一批图、同样 40 张新标注，主动学习选样 vs 随机选样各训一个模型，
在**同一份冻结测试集**上比较，并写 `data/reports/active-comparison.json`：

```bash
.venv/bin/python scripts/e2e_m5.py --photos 160 --epochs 15 --imgsz 320   # 约 2.5 分钟（含 4 次训练）
.venv/bin/rdinspect active compare --baseline <模型id> --candidate <模型id> --labeled 80
```

本机实测（合成数据、yolo11n@320、CPU）：基线 mAP50 **0.166** → 主动学习 **0.209**，
Δ **+0.043**（每 100 张标注 +0.055），结论"主动学习更优"。
⚠️ 这是**合成小样本**上的单次结果，样本量小、方差大；真实数据请按 [05-model-plan](./05-model-plan.md) §8 的轮次持续记录，
`budget_effect()` 会如实给出"更优 / 无显著差异 / 基线更优（需复查选样策略）"三种结论之一。

## 4. 增量重训与"连续两次门禁不通过"

```bash
# 增量重训 = 在现役权重上继续微调（M3 就有能力，这里是用它做迭代）
.venv/bin/rdinspect train --dataset ds-2026w39 --epochs 30 --imgsz 320 \
    --resume-from data/runs/run-<id>-*/weights/best.pt --name yolo11n-road
.venv/bin/rdinspect model validate --id <新模型id>     # 未通过 → 退出码 5，状态保持 candidate
.venv/bin/rdinspect active alerts --threshold 2        # 连续 N 次不通过 → 告警
```

告警内容会指出**先回看标注规范与类别定义**（复核 Kappa、打回率、易混类对），而不是继续调参——
这是 ADR-0007 的处置约定。告警同时可用 `GET /api/active/alerts?threshold=2` 取。

门禁在本轮加固了两点（都影响"能不能相信这个结论"）：

1. **基线为 0 的类"在候选里缺失"不再算回退**（0.0 → 无信号，多半是该类在评估集里没有实例）；
2. **基线与候选的评估数据集不一致时会显式标注"不可比"**（`delta.comparable=false` + 说明），
   因为拿不同冻结集比 mAP 是无效比较。评估时可用 `model evaluate --id N --dataset <冻结集>` 统一到同一份测试集。

## 5. 新增类别（不改代码）

```bash
.venv/bin/rdinspect classes add --code water_puddle --zh 积水 --en "Water Puddle" --order 10
.venv/bin/rdinspect classes list                      # 确认注册与顺序
# 标注新类 → 新建数据集版本（旧版本冻结不动）→ 冻结 → 训练 → 门禁 → 导出
.venv/bin/rdinspect dataset create --name ds-2026w40 --review-status approved
.venv/bin/rdinspect train --dataset ds-2026w40 --epochs 30 --imgsz 320
.venv/bin/rdinspect model export --id <id> --imgsz 320
```

实测（`scripts/e2e_m5.py` E 段）：注册 `water_puddle` → 104 个新类标注 → 冻结 → 训练 → 导出，
导出包 `labels` 为 6 类且含新类，ONNX 一致性验收通过 —— **全程没有改任何代码**。

⚠️ 三个必须知道的约束：
1. **类别顺序会变**：新增类排到末尾会让"旧模型的类别下标"整体错位，因此**必须新建数据集版本重新训练**；
   用旧模型配新类别表导出会被拒（`类别顺序与当前类别表不一致`）。
2. 冻结数据集**不可变**：新类别的样本要进新版本，不要试图改旧版本。
3. 新类别样本少时，先按 [04-annotation-workflow](./04-annotation-workflow.md) 定义清边界（如"积水"vs"坑洞"），
   否则门禁会以"类别塌陷/易混类回退"的形式反复拒绝。

## 6. 统计报表

```bash
.venv/bin/rdinspect report --format markdown --days 30            # 时间趋势 + 批次对比 + 类别覆盖
.venv/bin/rdinspect report --format json --out /tmp/report.json
.venv/bin/rdinspect report --format gis-geojson --classes transverse_crack,pothole --since 2026-09-01
.venv/bin/rdinspect report --format gis-csv --bbox 121.3,31.1,121.6,31.4 --out hits.csv
```

- **时间趋势**：按天/周/月统计导入数、人工框数、提交与复核数、分类别框数（近似重复影像不计）；
- **批次对比**：每批次的图像数/重复数/框数/已标图数/分类别分布/导入统计（`stats_json`）；
- **类别覆盖**：每个类别的已复核图像数、框数、模型候选数、采纳数 + 待办与复核通过率；
- **GIS 导出**：只有带 GPS 的检测点才导出，`without_gps` 会如实报出"有多少点画不到地图上"；
  GeoJSON 坐标是 `[lon, lat]`，CSV 表头固定（含 `length_px/width_px/area_ratio`，取不到时为空）。

REST 对应 `GET /api/reports/summary?format=json|markdown` 与 `GET /api/reports/gis?format=geojson|csv`。

## 7. 排障速查

| 症状 | 原因 | 处理 |
|---|---|---|
| 队列分数全是 0、优先级没变 | 没有模型候选、也没有评估记录（提示里会写明） | 先 `prelabel`，或 `--package` 传入导出包；确要"空图也算不确定"用 `--empty-weight 0.5` |
| `--limit 3` 却只选出 3 个候选 | 把 `--limit` 当成了候选池上限 | 候选池用 `--scan-limit`（默认全部）；`--limit` 只限制入选数量 |
| 门禁报"类别 X 在候选模型中缺失" | 基线该类 mAP50 > 容差，候选真的没有 | 检查新数据集是否漏标该类；基线≈0 的情况现在只会给提示 |
| 门禁结论带 `comparable=false` | 基线与候选的评估集不是同一份冻结集 | `model evaluate --id N --split test --dataset <同一冻结集>` 后复评 |
| 导出被拒"类别顺序与当前类别表不一致" | 加过类别但用的是旧模型 | 用新类别顺序重新冻结数据集并训练（§5） |

## 8. 自动化验收

```bash
.venv/bin/python scripts/e2e_m5.py --photos 160 --epochs 15 --imgsz 320
```

覆盖 6 项断言：队列落库（run + 明细 + `strategy=active` 视图）/ 有信号时优先级写回 /
同预算 A/B 记录完整 / 新增类别全链路 / 连续两次门禁失败告警 / 报表（趋势·批次·GIS），
末行打印 `M5 验收：通过 ✅（6/6 项）`，报告写 `<work>/e2e_m5_report.json`。
