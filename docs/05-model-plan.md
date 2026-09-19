# 05 · 模型方案（Model Plan）

## 1. 类别体系与映射

| 本项目 code | 中文 | 公开数据集对应 | 备注 |
|---|---|---|---|
| `longitudinal_crack` | 纵向裂缝 | RDD2022 **D00** | 与行车方向近似平行 |
| `transverse_crack` | 横向裂缝 | RDD2022 **D10** | 与行车方向近似垂直 |
| `pothole` | 坑洞 | RDD2022 **D40** | 有深度、边缘破损 |
| `garbage` | 垃圾 | 无（TACO 等通用垃圾数据辅助） | 需自采 + 零样本冷启动 |
| `alligator_crack` | 网裂（预留） | RDD2022 **D20** | 数据自带，首版可不标不训 |

扩展规则：新增类别 = `classes` 表注册 + 补样本 + 重训；`order_index` 变化时数据集必须新建版本（YOLO 索引随顺序）。

## 2. 数据来源与规模（首版小样本 3–5k）

| 来源 | 用途 | 规模建议 | 获取方式 |
|---|---|---|---|
| **Unified Road Defect Dataset**（HF，四类 CRDDC） | 主训练集 | 2,000–3,000（含 UAV-PDD2023 航拍子集） | HF 镜像分片 `train_a/b.tar.gz`、`val.tar.gz` 按需下载（本机直连 huggingface.co 不通，用 hf-mirror.com 镜像；实测镜像 API 可用） |
| **RDD2022**（原始，6 国） | 补充与域多样性 | 1,000（中国/日本优先，与我国路况更近） | 官方/Kaggle 镜像（Kaggle 页实测可访问） |
| **自有采集**（照片/视频抽帧/航拍） | 域适配主力 | 500–1,000 | 本程序导入（M1 后） |
| **垃圾类** | 补齐第四类 | 300–500 | 自采为主；TACO 等通用垃圾数据做预训练/预标注辅助 |

> 磁盘约束：当前可用 27GB，上表合计约 5–6GB（含缩略图与导出），可行。扩容见 08-deployment。

**数据划分**：train/val/test = 7:1.5:1.5，**按路段（GPS 网格或 road_segment）分组划分**，防止同一路段跨集合泄漏。测试集冻结后不参与任何训练/调参。

## 3. 训练配方（首版）

| 项 | 值 | 说明 |
|---|---|---|
| 模型 | **YOLO11s**（备选 YOLO11n 用于边缘） | s 精度优先训练；n 用于导出边缘 |
| 输入 | `imgsz=640`（航拍增强轮次可用 1024） | 16GB 显存下 batch 16 @640 可行 |
| epochs | 100（早停 patience 20） | 小样本易过拟合，监控 val mAP |
| 优化 | AdamW，`lr0=0.001`，cos 退火，warmup 3 | ultralytics 默认即可 |
| 增强 | mosaic 1.0、scale 0.5、translate 0.1、degrees 10、HSV、`fliplr=0.5`、**`flipud=0（禁用）** | ⚠️ 垂直翻转会交换"横/纵"语义，必须关闭 |
| 类别不均衡 | 垃圾类过采样 + copy-paste | 垃圾样本少，否则召回塌陷 |
| 训练框架 | ultralytics（AGPL，内部自用） | 见 ADR-0003 的替换路径 |

**M3 落地时补充的实现约定**（`configs/train.yaml` + `train/runner.py`）：

| 项 | 落地方式 |
|---|---|
| 硬约束 | `augment.flipud ≠ 0` 直接抛配置错误（要开必须显式 `allow_vertical_flip: true`），见 ADR-0001/roadmap |
| 幂等 | `run_key = sha256(数据集清单哈希 + 架构 + 初始权重指纹 + 全部超参 + 增强)`；重复提交复用既有运行，正在运行则 409 |
| 断点 | `resume_from` 指向 `last.pt` 时按 ultralytics 的 `resume=True` 续训；指向 `best.pt`/第三方权重时当初始化权重用 |
| 日志 | ultralytics 日志逐行落盘 `data/logs/runs/train-<run_id>-*.log`，每个 epoch 把指标增量写进 `runs.metrics_json.progress`（前端/CLI 可轮询） |
| data.yaml | ultralytics 用自身 `datasets_dir` 解析相对的 `path:`，因此在 `data/runs/_data/` 生成绝对路径副本，冻结数据集本身不改（保持不可变） |
| 训练分辨率 | 记录在 `model_versions.metrics_json.train_params`，评估/导出默认沿用同一 imgsz（否则指标不可比） |
| 受限容器 | `/dev/shm` 不可写时 ultralytics 线程池建信号量会失败 → `train/compat.py` 自动退化为串行扫描（`RDINSPECT_SERIAL_SCAN=1` 可强制） |
| 中断自愈 | 服务重启后遗留的 `running` 在启动对账时置 `failed`（`rdinspect runs --reconcile`），同配置可重新提交 |

**训练阶段（两段式）**：
1. **域预训练**：用 RDD2022/CRDDC 公开数据训练 3 类（横裂/纵裂/坑洞）→ 得到基础权重。
2. **域适配微调**：加入自有数据 + 垃圾类 → 小学习率微调（`lr0=0.0005`，30–50 epochs），固定测试集评估。

## 4. 细裂缝与航拍：分辨率策略

- **切片推理（SAHI）**：tile 1024 / overlap 0.2，NMS IoU 0.5；航拍大图（8K）必开。
- **行车记录仪**：1080p 抽帧后整帧 640 推理；若漏检高，改 2×2 tile（每片 640）再合并。
- **训练侧**：对细裂缝样本可额外做 1024 轮次（`imgsz=1024`，batch 8），并在验证时分别报告 640/1024 指标。
- **小目标桶评估**：按框面积分为 small(<32²)/medium/large 三桶报告召回，定位"细裂缝漏检"问题。

## 5. 模型辅助标注所用模型

| 用途 | 模型 | 理由 | 资源 |
|---|---|---|---|
| 检测预标注 | 现役 YOLO11s（生产权重） | 与最终模型一致，误差可控 | GPU < 4GB |
| 裂缝掩膜 | **SAM2 tiny/small**（框提示） | 免训练、交互式、掩膜质量高 | GPU < 3GB |
| 垃圾零样本 | **GroundingDINO / OWLv2**（文本提示 "garbage, trash, debris"） | 无专用数据集时先出草稿 | GPU < 6GB，可按需加载 |
| 冷启动基线 | 公开 CRDDC 预训练 YOLO 权重 | 在自有数据不足时先验证全链路 | 同上 |

所有辅助模型均可关闭；关闭后标注台退化为纯人工，不影响流程。

### 5.1 可直接使用的公开权重（M2 实测）

| 用途 | 权重 | 来源 | 许可 | 实测 |
|---|---|---|---|---|
| 病害检测（冷启动） | `yolo12s_RDD2022_best.pt`（RDD2022 训练，5 类：D00/D10/D20/D40/Repair） | HF `rezzzq/yolo12s-road-damage-rdd2022` | MIT | 10 张真实路面图 → **18 个候选，映射率 100%**（修复通道顺序后复测；`Repair` 为未收录类，若出现会被丢弃并计数） |
| 通用检测（打通链路） | `yolo11n.pt`（COCO 80 类） | ultralytics assets | AGPL-3.0 | 仅验证适配器，类别会全部落入 `unmapped` |
| 裂缝掩膜 | `sam2.1_t.pt` | ultralytics assets | Apache-2.0 | 裂缝候选 → 掩膜宽度 16.6/55.0 px，长度 251/299 px |

> **下载提示（国内网络）**：GitHub 大文件直连常超时，可经镜像前缀加速，例如
> `https://ghfast.top/https://github.com/ultralytics/assets/releases/download/v8.4.0/sam2.1_t.pt`；
> HuggingFace 用 `https://hf-mirror.com/<repo>/resolve/main/<file>`。下载后放入 `data/weights/` 即可
> （程序会把裸文件名权重解析到该目录，见 `prelabel/service.py:_localize_weights`）。

## 6. 评估协议与指标目标

**评估对象**：冻结测试集（`dataset_versions.status='frozen'` 的 test 划分）。
**报告内容**：分类别 P/R/mAP50/mAP50-95、混淆矩阵、大小桶召回、SAHI 开关对比、失败样例 20 张导出。

| 指标 | 首版目标（YOLO11s@640） | 说明 |
|---|---|---|
| mAP50（总体） | ≥ 0.60 | 小样本现实目标 |
| 坑洞 mAP50 / 召回@0.3 | ≥ 0.70 / ≥ 0.85 | 安全优先级最高 |
| 横向裂缝 mAP50 | ≥ 0.55 | 中等难度 |
| 纵向裂缝 mAP50 | ≥ 0.50 | 细长、最难 |
| 垃圾 mAP50 | ≥ 0.60 | 样本少，先保证不误报过多 |
| 推理时延（GPU, 640） | ≤ 15ms/帧 | YOLO11s，含预处理 |
| 边缘时延（CPU 4 线程, 640, YOLO11n ONNX） | ≥ 15 FPS | 满足 FR-8.5 |

### 6.1 M3 实测口径与结果（120 张合成数据，yolo11n@320，CPU）

| 口径 | mAP50 | mAP50-95 | P | R | 说明 |
|---|---|---|---|---|---|
| ultralytics `val`（门禁主判据） | 0.9125 | 0.7249 | 1.000 | 0.912 | 只在**有真值的类别**上取平均 |
| 内部匹配（`train/matching.py`，VOC2010 全点 AP） | 0.9167 | — | 0.579 | 0.524 | 固定 conf=0.25 下的 P/R；AP 按"仅有真值的类"平均 |
| 内部匹配（按类别表全量平均） | 0.7333 | — | — | — | 缺真值的类记 0，是保守口径，不要与官方数直接比 |

- 两套实现相差 **0.004**（0.9167 vs 0.9125），互为交叉验证；差距来源是官方对每类做 101 点插值、内部做全点插值。
- 分类别（官方）：纵向 0.995 / 坑洞 0.995 / 垃圾 0.995 / 横向 0.665；网裂在本次 val 划分中无实例（报告里显示 `—`，不是 0）。
- 大小桶召回：medium 0.75、large 1.00、small 无真值（`null`）；合成数据的缺陷偏大，**真实细裂缝必须重新按 small 桶验收**。
- **切片开关对比：关闭 0.9167 → 开启 0.047（384px 窗口，召回 0.76 → 0.05）**。原因是本数据集以大目标为主，切片把目标切碎。结论：切片不是"默认更好"，必须按数据类型开关（ADR-0006）；细裂缝/航拍数据要单独做对照再决定。
- 失败样例导出：本次 val 仅 2 张含失败样例，叠加图落在 `data/runs/eval-<run_id>-*/failures/`。

> 指标目标（§6 表）是**真实 3–5k 数据 + YOLO11s@640** 的目标；上表是打通链路的合成小样本结果，不能当作真实路况精度。

## 7. 导出与边缘部署

| 项 | 方案 |
|---|---|
| 导出格式 | ONNX（opset 17，dynamic batch，simplify）；边缘可选 TensorRT FP16/INT8 |
| 导出包内容 | `model.onnx` + `labels.yaml`（类别顺序、阈值）+ `preprocess.json`（letterbox/归一化/切片参数）+ `manifest.json`（版本、指标、哈希、schema_version） |
| 后处理一致性 | NMS/切片合并/坐标还原代码在**工作站与边缘共用同一模块**，只由 `preprocess.json` 参数化 |
| 量化 | 首选 FP16；INT8 需用冻结验证集校准并复测（目标掉点 ≤ 2 mAP50） |
| 版本门禁 | 新权重在冻结测试集上不低于现役（容差 ±0.005 mAP50）；未通过仅可标 `candidate`，不得进 `production` |
| M3 实际导出包 | `model.onnx` + `labels.txt`（类别 code，行号=类别下标）+ `preprocess.json`（letterbox/归一化/NMS 阈值）+ `manifest.json`（数据集清单哈希、训练/导出 run、门禁结论、权重与 ONNX 的 sha256）+ `parity.json` + `README.md`；目录 `data/exports/<name>-<version>/` |
| M3 一致性验收 | 同一份居中 letterbox 张量分别喂 torch 与 onnxruntime：**逐框归一化坐标最大误差 ≤ 1e-3**（实测 1e-06，容差 1e-3），且两侧不得有独有框；结果写进 `parity.json` 与 `manifest.parity` |
| 预处理一致性 | ultralytics `predict` 默认 `rect=True`（stride 对齐**非居中**填充）会与导出包不一致（同图分数差 ~2%）→ 工作站检测器显式 `rect=False`，实测与导出包逐框误差 0.0 |

## 8. 主动学习与迭代节奏

| 轮次 | 触发条件 | 动作 |
|---|---|---|
| R0 | 无自有数据 | 用公开权重预标注自有样本，人工修正 300–500 张 |
| R1 | 已有 500 张 approved | 训练首版（3 类 + 垃圾冷启动），导出基线 |
| R2 | 累计 1,500 张 | 加入不确定样本（低置信/混淆类），重训并门禁 |
| R3+ | 每 1,000 张新标注或月度 | 增量重训；若门禁连续两次不通过则回看标注规范与类别定义 |

**选样策略**：50% 不确定性（置信度 0.25–0.45 或 top1-top2 差值 < 0.1）+ 30% 错误驱动（混淆矩阵高发类别）+ 20% 多样性（pHash 聚类每簇 1 张）。

## 9. 风险与缓解

| 风险 | 影响 | 缓解 |
|---|---|---|
| 域差异（国外路况 vs 本地） | 公开数据训练的模型在本地位漏检 | 两段式训练：公开数据预训练 + 自有数据微调；优先用中国子集 |
| 横向接缝/修补块误标 | 标签噪声 | 标注手册明确边界 + `difficult` 标记 + 复核抽检 |
| 细裂缝漏检 | 关键指标不达标 | SAHI 切片 + 1024 轮次 + 小目标桶评估 + 召回优先阈值 |
| 垃圾类样本稀少 | 该类塌陷 | 零样本预标注 + 过采样 + copy-paste + 单独报告（不与裂缝混算） |
| 预标注错误被批量采纳 | 训练集被污染 | 预标注仅作候选、来源可区分、采纳率监控、新标注员全检 |
| AGPL/GPL 许可 | 将来闭源分发受限 | 见 ADR-0003：训练框架可换 RTMDet/YOLOX，标注辅助可换自建掩膜工具 |
| 磁盘不足 | 无法扩数据 | 小样本起步 + 硬链接导出 + 外接盘/NAS 方案（08-deployment） |
