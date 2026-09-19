# 09 · 里程碑与路线图（Roadmap）

> 本阶段（M0）只交付**架构与文档**；M1 起进入实现，每个里程碑都有独立验收标准，不通过不进入下一阶段。

## 1. 总览

| 里程碑 | 目标 | 主要产出 | 预估工作量 | 依赖 |
|---|---|---|---|---|
| **M0 文档与骨架 ✅** | 设计定稿、可评审 | 本 docs 全集、`db/schema.sql`、`openapi.yaml`、格式样例 | 1–2 人日 | — |
| **M1 采集与标注闭环 ✅** | 能导入、能标、能导出 | FastAPI 服务、SQLite 仓储、导入管线（EXIF/抽帧/切片/去重）、标注台、三格式导出 | 8–12 人日 | M0 |
| **M2 模型辅助标注 ✅** | 机器先打草稿 | 预标注服务（YOLO + SAHI）、SAM 掩膜辅助、候选采纳/忽略、来源标记、质量看板 | 5–8 人日 | M1 |
| **M3 训练闭环与门禁 ✅** | 小样本训练出可用权重 | 数据集冻结、训练/评估 runner、分类别指标与混淆矩阵、门禁、ONNX 导出 | 6–9 人日 | M1 |
| **M4 边缘离线推理 ✅** | 车载/巡检无网可用 | `rdinspect infer` CLI、导出包格式、JSONL/CSV 输出、性能达标 | 4–6 人日 | M3 |
| **M5 主动学习与扩展** | 越用越准、可加类 | 优先队列策略、增量重训、新增类别全流程、统计报表完善 | 5–8 人日 | M2/M3 |

合计约 **29–45 人日**（不含数据采集与标注人工工时）。

## 2. 各里程碑任务分解

### M1 采集与标注闭环 ✅（已完成，验收结果见 README「测试」）

**实现与验收摘要**
- 代码：`src/rdinspect/{config,cli,errors}.py`、`storage/{db,files,repo}.py`、`core/{hashing,geometry,images,ingest,formats,datasets}.py`、`api/app.py` + `api/static/index.html`（单文件标注台）。
- 测试：44 项单测 + 端到端验收（500 张合成照片 + 1 段视频 → 导入 → 四类各 50 张标注复核 → 冻结导出 → 三格式往返零误差 → 清单哈希可复现）。
- 与设计的两处一致偏差：① 标注台为单文件原生 Canvas（非 Vite+React+Konva），零构建零外部依赖；② 当时预标注/训练端点返回 501（M2/M3 已补齐；现在只有 M4/M5 返回 501）。

#### 原始任务分解（8–12 人日）
1. 工程骨架：`pyproject.toml`、`src/rdinspect/{api,core,storage,cli}`、配置加载、日志、`systemd --user` 单元。
2. 存储层：`schema.sql` 落地、仓储 API（事务/原子写/迁移 runner）、SQLite WAL 参数。
3. 导入管线：EXIF 解析、SHA256/pHash 去重、视频抽帧（ffmpeg）、航拍切片、批次统计与失败清单。
4. 标注 API：任务队列 + 租约、标注全量提交（diff + 审计）、复核决定。
5. 标注台前端：画布（Konva）、类别条、快捷键、缩略图队列、亮度/对比度、本地草稿恢复。
6. 导出：YOLO / COCO / LabelMe 三格式 + `data.yaml`，并用往返测试保证零误差。
7. **验收**：上传 500 张照片 + 1 段视频 → 去重与抽帧正确；四类各标 50 张 → 三格式互转零误差；`openapi.yaml` 与实际实现路径一致。

### M2 模型辅助标注 ✅（已完成，验收见 README「测试」）

**实现与验收摘要**
- 代码：`src/rdinspect/prelabel/{detector,sahi,sam,service,metrics}.py`；API 新增预标注/忽略候选/SAM 掩膜/质量看板/模型注册；CLI 新增 `prelabel`、`prelabel-metrics`、`models`、`classes`；标注台新增「预标注本图」「批量预标注」「采纳/忽略候选」「预标注质量」面板。
- 验收（`scripts/e2e_m2.py`，6/6 通过）：200 张批量预标注（15.7s）→ 120 采纳 / 40 忽略 / 40 待处理，采纳率 0.60、模型-人工 IoU 0.86、匹配率 100% → 采纳标签冻结进数据集（115 张，近似重复被正确排除）；关闭预标注时人工流程不受影响；真机用 RDD2022 权重（YOLOv12s）在 10 张真实路面图产出 18 个候选，类别映射率 100%（M3 修复"numpy 输入被 ultralytics 当 BGR"的通道顺序缺陷后复测；修复前为 17 个候选 / 89.5%，R/B 交换会静默降低精度）；SAM2.1-t 对裂缝候选生成掩膜并给出宽度指标（16.6–55.0 px）。

#### 原始任务分解（5–8 人日）
1. `Detector` 抽象 + ultralytics 适配器（GPU/CPU 自动选择）。
2. SAHI 切片推理与结果合并（tile 尺寸/重叠/NMS IoU 可配）。
3. 预标注任务化（`runs.prelabel`）、候选落库（`source=model`）、采纳/忽略交互、低置信度可视化。
4. SAM 掩膜辅助（框提示 → 掩膜 → 人工修正 → PNG 入库）。
5. 预标注质量看板：采纳率、模型-人工一致率。
6. **验收**：批量预标注 200 张完成；候选可逐条采纳；关闭预标注不影响人工流程；跑通 SAM 掩膜在 20 张裂缝样本上可用。

### M3 训练闭环与门禁 ✅（已完成，验收见 README「测试」）

**实现与验收摘要**
- 代码：`src/rdinspect/train/{matching,runner,evaluate,gate,export_onnx,service,compat}.py` + `src/rdinspect/edge/onnx_runtime.py`；API 新增 `POST /api/train/runs`、`GET /api/train/runs/{id}`、`.../cancel`、`/api/models/registry`、`/api/models/{id}`、`.../evaluate|validate|promote|export`；CLI 新增 `train`、`runs`、`model {list,show,evaluate,validate,promote,export}`。
- 验收（`scripts/e2e_m3.py`，**13/13 通过**，整机 CPU 约 100 秒）：
  - 120 张合成路面图（202 个真值框）→ 冻结数据集（train 76 / val 14 / test 23，manifest `04e96f4d…`）；
  - 30 epoch 微调（yolo11n，320px，batch 8，CPU 71s）→ 训练末 mAP50 **0.9345**；
  - 独立评估（val，imgsz=320）：官方 val mAP50 **0.9125** / mAP50-95 0.7249 / P 1.000 / R 0.912；分类别 mAP50：纵向 0.995、坑洞 0.995、垃圾 0.995、横向 0.665（网裂在 val 中无实例）；
  - 内部匹配口径（自研贪心匹配 + VOC2010 AP）给出 **0.9167**（仅有真值的类），与官方 0.9125 相差 0.004 → 两套实现互相印证；
  - 门禁：首个模型按绝对下限通过 → 提升 production；把 COCO 预训练权重登记为 candidate 后，门禁以「整体 mAP50 下降 0.9125」+「4 类类别塌陷」拒绝（409），状态保持 candidate；
  - ONNX 导出包（labels/preprocess/manifest/parity）：同 letterbox 张量下 ONNX 与 .pt 的**逐框归一化坐标最大误差 1e-06**（容差 1e-3），与工作站检测器误差 0.0；
  - 幂等：同参数重复提交复用既有 run（0.0s）；`flipud ≠ 0` 直接拒绝；
  - HTTP 契约：`/api/models/registry`、`/api/train/runs/{id}`（含 progress/log_tail）、劣化模型 `validate` 409 文案均符合 `docs/06-api-spec.md`。
- 三个实测发现（都已修/已记录）：
  1. **通道顺序**：ultralytics 把 numpy 彩色输入当 BGR，预标注原先传 RGB 导致 R/B 被交换（M2 遗留缺陷）→ 传 BGR 并写清约定；
  2. **letterbox 不一致**：ultralytics `predict` 默认 `rect=True`，按 stride 对齐做**非居中**填充，与导出包的居中 letterbox 不同（同图分数差 ~2%）→ 检测器显式 `rect=False`，工作站与边缘预处理对齐（验收里与工作站误差 0.0）；
  3. **切片并非总是更好**：本数据集以大目标为主，384px 切片把目标切碎，召回 0.76 → 0.05；切片只对细裂缝/航拍小目标有益，必须按数据开关（ADR-0006 的前提被实测确认）。
- 受限容器兼容：ultralytics 扫描标签要建 POSIX 信号量（`/dev/shm`），只读 `/dev/shm` 的沙箱/容器会 `PermissionError` 挂掉；`train/compat.py` 探测到信号量不可用时把线程池换成串行实现（正常环境零影响，可用 `RDINSPECT_SERIAL_SCAN=1` 强制）。
- 进程中断自愈：服务被杀/重启后遗留的 `running` 运行会在服务启动时对账为 `failed`（也可 `rdinspect runs --reconcile`），同一配置的训练因此不会被永久挡住；依赖缺失在**请求期**就返回 501，不会先 202 再失败。

#### 原始任务分解（6–9 人日）
1. 数据集选样/预览/冻结（含 `manifest_hash` 与划分分组策略）。
2. 训练 runner：ultralytics 微调、日志流、断点、`flipud=false` 硬约束、可选 `resume_from`。
3. 评估：分类别指标、混淆矩阵、大小桶召回、SAHI 开关对比、失败样例导出。
4. 模型注册表与门禁：`candidate→validated→production`，未通过返回 409 并说明 delta。
5. ONNX 导出（opset/dynamic batch）+ 导出包（labels/preprocess/manifest）。
6. **验收**：3–5k 张小样本训练完成并出报告；构造劣化权重被门禁拒绝；导出包可用工作站 ONNX 复现同一批预测（容差 1e-3）。
   > 实测口径说明：本轮用 120 张合成数据（3–5k 张真实数据要等采集到量），三项验收均已在 `scripts/e2e_m3.py` 中自动断言。

### M4 边缘离线推理 ✅（已完成，验收见 README「测试」）

**实现与验收摘要**
- 代码：`src/rdinspect/edge/{package,sources,writers,infer,onnx_runtime}.py` + CLI `rdinspect infer` + `scripts/build_edge_bundle.py`；不上数据库、不依赖 torch/ultralytics（实测推理链路 `sys.modules` 里没有 torch/ultralytics/sqlite3）。
- 验收（`scripts/e2e_m4.py`，**8/8 通过**）：
  - **无网**：代理指向死端口 + `HF_HUB_OFFLINE=1`，200 张图分两段跑完（100 + 100），JSONL 200 行、重复哈希 0、累计计数 200；
  - **与工作站一致**：同批 20 张图，ONNX 与 ultralytics `.pt` 逐框归一化坐标最大误差 **1e-06**、单侧独有框 0；
  - **断点续跑**：第二段 `--resume` 跳过 100 张、不重复追加；状态文件删除后也能用 JSONL 重建已完成集合；
  - **拒绝启动**：schema 版本更高 / 类别被整体重排（含 manifest）/ 模型文件被追加字节 / `--expect-labels` 不符，四种情况全部退出码 **6** 且不产出任何结果文件；
  - **性能**（CPU 4 线程，端到端含解码+letterbox+NMS+写盘）：imgsz 640 **43.7 FPS**（p50 22.9ms）达标（目标 ≥15 FPS）；imgsz 320 141 FPS（p50 6.96ms）；
  - **离线安装包**：`build_edge_bundle.py --verify` 在临时 venv 里用 `--no-index` 安装并冒烟推理通过，整包 59MB（wheels 48.7MB + 模型 10.5MB）。
- 三个实测发现（都已修）：
  1. **导出包硬链接互相覆盖**：同名 `.onnx` 被两次导出复用时，先前那个包的模型会被悄悄改掉 → 改为复制，并加回归测试；
  2. **edge.yaml 覆盖包内 imgsz**：会让 letterbox 比例与验收时不一致 → 包内 `preprocess.json` 为唯一契约（`imgsz: null`），静态尺寸模型被显式覆盖时直接报错；
  3. **pip 安装后没有 configs/**：非 editable 安装时 `rdinspect infer` 曾直接报"配置文件不存在" → 缺省回退内置默认值（显式 `--config` 仍报错）。

#### 原始任务分解（4–6 人日）
1. `rdinspect infer` CLI：目录/视频/RTSP、`--resume`、JSONL/CSV 输出。
2. 边缘依赖最小化（onnxruntime + pillow/opencv + numpy）；离线安装脚本。
3. 性能达标测试（CPU 4 线程 ≥15 FPS；GPU ≥100 FPS）与显存上限验证。
4. 版本校验：`manifest.json` 与模型/类别不匹配时拒绝启动。
5. **验收**：无网环境对 200 张图完成推理；结果与工作站一致；断点续跑不重复处理。
   > 实测口径：本轮用 200 张合成图 + M3 训练出的真实 ONNX 包，三条验收均已自动断言（含"篡改包必须拒绝启动"）。

### M5 主动学习与扩展（5–8 人日）
1. 选样策略：不确定性（置信度区间/top1-top2 差值）、多样性（pHash 聚类）、错误驱动（混淆矩阵）。
2. 优先队列（`tasks.priority`）与批量标注工作流。
3. 增量重训流程（在既有权重上微调）与"两次门禁不通过 → 回看标注规范"的告警。
4. 新增类别全流程验证（以网裂 `alligator_crack` 为例：注册 → 标注 → 冻结 → 训练 → 导出）。
5. 统计报表完善（时间趋势、批次对比、导出 GIS 字段）。
6. **验收**：新增第五类不改代码即可跑通全链路；主动学习队列使单位标注量的 mAP 增量可测（对比实验记录）。

## 3. 关键路径与并行建议

```
M0 ──▶ M1（导入∥标注台 可并行）──▶ M2（预标注）──▶ M3（训练）──▶ M4（边缘）
                          └────────────▶ M5（主动学习，依赖 M2+M3）
```
- 数据采集与标注可与 M1/M2 并行启动（越早越好，属于长周期外部依赖）。
- M3 的"数据集冻结"建议在 M1 完成后立即做最小版本，先跑通一次"训练-评估-门禁"闭环，再逐步加量。

## 4. 风险登记（里程碑视角）

| 风险 | 触发信号 | 应对 |
|---|---|---|
| 标注人力不足 | 每天 approved < 100 张 | 优先用公开数据预训练；提高预标注采纳率；缩小类别范围先做 3 类 |
| 数据域不匹配 | 自有验证集 mAP 远低于公开集 | 增加自有样本；域适应微调；引入夜间/逆光增强 |
| 磁盘不足 | 可用 < 10GB | 小样本轮换、硬链接导出、接入外接盘（见 08 §3） |
| GPU 竞争（与本地大模型/视频生成冲突） | 训练排队 | `runs` 串行化 + 空闲时段训练；预标注降为 CPU |
| 类别定义争议 | 复核打回率高、Kappa < 0.7 | 更新标注手册 + 复训标注员 + 增加示例图库 |
| 依赖许可变更 | ultralytics 升级改许可 | 按 ADR-0003 切换 RTMDet/YOLOX；导出与标注格式不变 |
