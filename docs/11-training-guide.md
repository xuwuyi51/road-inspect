# 11 · 训练闭环操作手册（M3）

> 面向"要跑一次训练并决定能不能上线"的人。命令都可直接复制；每一步都给出**产物路径**与**失败时看什么**。
> 设计与阈值来源见 [05-model-plan](./05-model-plan.md) §3/§6/§7，决策背景见 [ADR-0005](./adr/0005-dataset-freeze-and-hash.md) 与
> [ADR-0007](./adr/0007-active-learning-and-model-gate.md)。

## 1. 全链路一图

```
冻结数据集(draft→frozen)        训练(candidate)          评估+门禁            提升            导出
─────────────────────────►  ─────────────────────►  ────────────────►  ───────────►  ──────────────►
rdinspect dataset freeze      rdinspect train          rdinspect model     model promote   model export
（不可变 + manifest_hash）     （run_key 幂等）          validate(409=拒绝)  → production    → ONNX 导出包
```

模型状态机：`candidate → validated → production`；同任务下**只能有一个** production，提升新模型会自动把旧的
置为 `archived`。门禁不通过时模型**保持 candidate**，并且 409 的 detail 里给出差在哪一类、差多少。

## 2. 准备：数据集必须已冻结

```bash
# 从已复核（approved）的标注里选样 → 划定 train/val/test → 冻结
.venv/bin/rdinspect dataset create --name ds-2026w38 --review-status approved \
    --split-train 0.7 --split-val 0.15 --split-test 0.15 --seed 42
.venv/bin/rdinspect dataset freeze --name ds-2026w38
```

冻结会导出 YOLO 目录（图片用硬链接，不额外占盘）并写 `manifest.sha256`。**冻结后不可变**：任何新的标注都要
另建数据集版本（否则"这次训练用了哪些数据"就无法复现）。训练只接受 `frozen` 数据集，对 draft 直接 409。

## 3. 训练

```bash
# 最小可用（其余超参取 configs/train.yaml）
.venv/bin/rdinspect train --dataset ds-2026w38 --arch data/weights/yolo11n.pt \
    --epochs 30 --imgsz 320 --batch 8 --device cpu
```

| 关键行为 | 说明 |
|---|---|
| 幂等 | `run_key = sha256(数据集清单哈希 + 架构 + 初始权重指纹 + 全部超参 + 增强参数)`；同参数重复提交**复用**既有成功运行（秒回，不再烧算力），正在运行时返回 409 |
| 硬约束 | `augment.flipud` 非 0 直接报错（垂直翻转会把"横向裂缝"翻成"纵向裂缝"）；确需打开要显式设 `allow_vertical_flip: true` |
| 日志 | `data/logs/runs/train-<run_id>-*.log`，每个 epoch 的指标同时增量写进 `runs.metrics_json.progress` |
| 断点 | `--resume-from <run>/weights/last.pt` 续训（保留优化器/epoch 状态）；传 `best.pt` 或第三方权重时按"初始化权重"做域适配微调 |
| 产物 | `data/runs/run-<run_id>-<slug>/weights/best.pt`、`last.pt`、`results.csv`；成功后自动登记 `candidate` 模型 |
| 失败 | 失败原因（含堆栈）落在 `runs.error` 与日志尾部；**重跑会把旧目录改名 `.prev-<时刻>`**，不会拿陈旧 results.csv 充数 |

后台模式（Web/API 用）：

```bash
curl -X POST localhost:8787/api/train/runs -H 'content-type: application/json' \
     -d '{"dataset":"ds-2026w38","epochs":30,"imgsz":320}'
# → 202 {"run_id":7,...}（异步；命中幂等时 cached=true 且 message 说明"未启动新训练"）
# 加 "wait":true 则同步跑完再返回 200 + status；随后轮询
curl localhost:8787/api/train/runs/7          # progress/epochs_done/final/log_tail
curl -X POST localhost:8787/api/train/runs/7/cancel   # epoch 边界停止，已产出的权重保留
```

## 4. 评估

```bash
.venv/bin/rdinspect model evaluate --id 3 --split val
```

产物（`data/runs/eval-<run_id>-<dataset>-<split>/`）：

| 文件 | 内容 |
|---|---|
| `report-val.md` | 人类可读报告：整体指标 → 分类别 → 大小桶召回 → 切片开关对比 → 混淆矩阵 → 失败样例清单 |
| `metrics-val.json` | 全量结构化指标（含 `ultralytics` 官方口径与 `internal` 内部口径两套） |
| `failures/` | 失败样例叠加图（绿=真值、红=预测，标注 `miss`/`fp`）+ `failures.json` 索引（按严重度排序） |

**两套口径为什么要都留着**：门禁主判据用官方 `val`（只在有真值的类别上平均，行业可比）；内部口径负责官方
`val` 给不出的东西——混淆矩阵、大小桶召回、**固定阈值下的 P/R**，以及**切片开关对比**（官方 val 不支持逐图切片）。
实测两者相差 0.004（0.9167 vs 0.9125），互为交叉验证；口径细节见 [05-model-plan](./05-model-plan.md) §6.1。

> 注意：报告中某类别显示 `—` 表示**该划分里没有这个类的实例**，不是"指标为 0"；大小桶里 `null` 同理。

## 5. 门禁与上线

```bash
.venv/bin/rdinspect model validate --id 3      # 评估 + 门禁；通过 → validated，未通过 → 退出码 5 且状态保持 candidate
.venv/bin/rdinspect model promote  --id 3      # 只有 validated 能提升；旧 production 自动 archived
.venv/bin/rdinspect models                     # 一览：状态 / mAP50 / 门禁结论
```

门禁判据（`configs/train.yaml: gate`，相对**现役 production** 比较）：

| 规则 | 默认 | 含义 |
|---|---|---|
| `map50_tolerance` | 0.005 | 整体 mAP50 允许的最大下降 |
| `per_class_tolerance` | 0.02 | 任何一类 mAP50 允许的最大下降（防"总体没掉、某类塌陷"） |
| `min_map50` | 0.0 | 绝对下限；**没有基线时（首个模型）只按它判定**，并在报告里标注"无基线"。⚠️ 默认 0.0 等于不做绝对把关——实测把 epoch 数从 30 降到 8 后，模型 mAP50 掉到 0 仍会通过门禁（门禁只防"变差"，不防"一开始就差"）；真实数据请设 `0.30` 左右 |
| `min_class_recall` | null | 可选：固定阈值下每类召回的硬下限 |

实测：把 COCO 预训练权重（无道路类别）登记为 candidate 后，门禁给出
`整体 mAP50 下降 0.9125 超过容差 0.0050（0.9125 → 0.0000）` + 4 条"类别塌陷"，返回 409 且状态保持 candidate。

## 6. 导出与一致性验收

```bash
.venv/bin/rdinspect model export --id 3 --imgsz 320      # 只有 validated/production 可导出
```

导出包 `data/exports/<name>-<version>-<imgsz>/`（同一版本可按 320/640 各导一份，互不覆盖）：

```
model.onnx        opset 17，dynamic batch，simplify
labels.txt        类别 code，行号 = 类别下标（顺序来自数据集冻结时的类别顺序）
preprocess.json   输入尺寸/RGB/letterbox(居中,pad 114)/归一化 + NMS 阈值（边缘端唯一契约）
manifest.json     数据集清单哈希、训练/导出 run、门禁结论、权重与 ONNX 的 sha256、类别明细
parity.json       一致性验收结果
README.md         边缘端 10 行用法
```

**一致性验收怎么做的**：用**包里的**预处理（居中 letterbox）把同一张图变成同一个张量，分别喂 torch 与
onnxruntime，比原始输出，再用**同一套**解码 + NMS 比逐框归一化坐标。判据：最大坐标误差 ≤ `export.tolerance`
（默认 1e-3）且两侧不得有独有框；不通过时 `registered=false`，不会更新模型的 `onnx_path`。

实测（120 张合成数据、yolo11n@320）：逐框坐标最大误差 **1e-06**、与工作站检测器误差 **0.0**、原始输出误差 1.3e-3。

## 7. 排障速查

| 症状 | 原因 | 处理 |
|---|---|---|
| `PermissionError: [Errno 13]` 紧跟在 "Fast image access ✅" 之后 | 容器/沙箱 `/dev/shm` 不可写，ultralytics 线程池建不了 POSIX 信号量 | 已内置兼容（自动串行扫描）；或 `--shm-size=1g` / `RDINSPECT_SERIAL_SCAN=1` |
| `images not found, missing path '.../images/val'` | ultralytics 用自身 `datasets_dir` 解析 data.yaml 里的相对 `path:` | 程序已在 `data/runs/_data/` 生成绝对路径副本；手工调 data.yaml 时也要用绝对路径 |
| 训练指标都是 0 | epoch 太少 / 标注与像素不匹配 / 学习率过低 | 先看 `report-val.md` 的失败样例叠加图：框与目标错位说明标注问题，框很准但漏检说明训练不足 |
| 门禁 409 | 新权重确实更差，或**类别顺序变了** | 看 `gate_json.reasons`；若是新增类别，必须新建数据集版本重新训练（类别下标会整体位移） |
| 导出报"类别顺序与注册表不一致" | 训练后又新增/停用/调整了类别 | 用新类别顺序重新冻结数据集并训练，不要复用旧权重 |
| 切片后指标反而变差 | 目标偏大，被切碎 | 这是预期行为（实测 0.9167 → 0.047）；按数据类型决定是否开切片，细裂缝/航拍单独做对照 |
| CWD 里出现 `:memory:.ses` | HOME 不可写，onnxruntime 把遥测设备 ID 落到了当前目录 | 无害（已 gitignore）；容器里把 `HOME`/`XDG_CACHE_HOME` 指到可写目录即可 |

## 8. 自动化验收

```bash
.venv/bin/python scripts/e2e_m3.py --photos 120 --epochs 30 --imgsz 320 --batch 8 --workers 2
```

覆盖 13 项断言（数据集 / 训练 / 评估 / 首个模型过门禁 / **劣化权重被 409 拒绝** / 导出包内容 /
ONNX 一致性 / 幂等 / `flipud` 硬约束 / **HTTP 契约** / 生产模型登记 `onnx_path`），末行打印
`M3 验收：通过 ✅（13/13 项）`，并把完整报告写成 `<work>/e2e_m3_report.json`。
整机 CPU 实测约 100 秒（含 120 张合成数据生成 20s + 30 epoch 训练 71s）。
