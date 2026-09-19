# road-inspect · 轻量级道路灾害巡查程序

> 识别道路上 **横向裂缝 / 纵向裂缝 / 坑洞 / 垃圾** 四类灾害（可扩展），支持**人工标注**与**模型辅助标注**，
> 覆盖「采集 → 标注 → 训练 → 导出 → 边缘离线推理」全链路。
> **当前状态：M0（架构文档）· M1（采集标注闭环）· M2（模型辅助标注）· M3（训练闭环与门禁）· M4（边缘离线推理）· M5（主动学习与类别扩展）均已交付并在本机实测通过。**

## 为什么做这个

现有开源方案要么是重型标注平台（CVAT/Label Studio），要么是零散的训练脚本：
**没有人把"数据治理 + 模型辅助标注 + 权重门禁 + 边缘导出"串成一条轻量闭环**。
本项目就是这条闭环，且在单机（RTX 5060 Ti 16GB / RAM 15GB / 磁盘有限）上跑得起来。

## 设计要点

| 主题 | 结论 | 详见 |
|---|---|---|
| 标注形态 | 检测框为主，裂缝可选 SAM 掩膜 | [ADR-0001](docs/adr/0001-detection-first-mask-optional.md) |
| 标注工具 | 自建轻量 Web 标注台；X-AnyLabeling 仅作外部工具（格式互通） | [ADR-0002](docs/adr/0002-self-hosted-lightweight-annotation-ui.md) |
| 模型 | YOLO11n/s（预标注 + 训练），ONNX 导出边缘 | [ADR-0003](docs/adr/0003-model-choice-and-license.md) |
| 存储 | SQLite(WAL) + 文件目录 + SHA256/pHash 双哈希去重 | [ADR-0004](docs/adr/0004-sqlite-and-file-storage.md) |
| 可复现 | 数据集冻结 + 清单哈希 + 模型门禁 | [ADR-0005](docs/adr/0005-dataset-freeze-and-hash.md) |
| 细裂缝 | 内置切片推理（SAHI 策略） | [ADR-0006](docs/adr/0006-sahi-sliced-inference.md) |
| 迭代 | 主动学习选样 + 门禁防劣化 | [ADR-0007](docs/adr/0007-active-learning-and-model-gate.md) |

## 目录

```
docs/            架构设计文档集（12 份 + 7 份 ADR + 格式样例）
db/schema.sql    数据库 DDL（可直接建库）
configs/         默认/预标注/训练/边缘 四份配置
src/rdinspect/   实现代码（M1 + M2）
```

## 快速阅读顺序

0. [docs/11-training-guide.md](docs/11-training-guide.md) — **M3 训练闭环操作手册**（训练→评估→门禁→导出）
0. [docs/12-edge-inference.md](docs/12-edge-inference.md) — **M4 边缘推理操作手册**（离线打包→推理→排障）
0. [docs/13-active-learning.md](docs/13-active-learning.md) — **M5 主动学习操作手册**（选样→A/B→扩类→报表）
1. [docs/00-overview.md](docs/00-overview.md) — 目标、范围、术语
2. [docs/02-architecture.md](docs/02-architecture.md) — 架构、数据流、时序、扩展点
3. [docs/03-data-model.md](docs/03-data-model.md) + [db/schema.sql](db/schema.sql) — 数据模型与建库
4. [docs/04-annotation-workflow.md](docs/04-annotation-workflow.md) — 标注与模型辅助流程
5. [docs/05-model-plan.md](docs/05-model-plan.md) — 模型方案与指标目标
6. [docs/09-roadmap.md](docs/09-roadmap.md) — 里程碑与验收

## 自检（当前已通过）

一键自检：`python3 scripts/check_docs.py`（退出码 0 表示全通过）

| 检查 | 覆盖内容 | 结果 |
|---|---|---|
| 数据库 DDL | `db/schema.sql` 建库、表/索引/视图/触发器、CHECK 约束生效 | 14 表 / 22 索引 / 2 视图 / 2 触发器 / 5 类别 |
| OpenAPI 草案 | `docs/openapi.yaml` 结构、状态码、`$ref` 完整性 | 25 路径 / 29 操作 / 17 schema / 29 引用 |
| 格式样例 | YOLO ↔ COCO ↔ LabelMe 归一化坐标零误差 | 4 类 × 3 格式往返一致 |
| 配置与文档 | 4 份 YAML 可解析、文档内相对链接可达 | 60 条链接全部可达 |

## 约束（来自实测环境）

- 磁盘仅剩 27GB（无第二块盘）→ 首版小样本 3–5k 张（约 5–6GB），扩容见 [docs/08-deployment.md](docs/08-deployment.md)
- GPU RTX 5060 Ti 15.5GB / 内存 15GB → 训练 batch ≤ 16 @640、dataloader workers ≤ 4
- 许可：内部自用，可用 ultralytics(AGPL) 与 X-AnyLabeling(GPLv3)；闭源分发路径见 ADR-0003

> ⚠️ 本项目的输出用于道路养护辅助判断，**不替代人工现场核验**；不存在"零漏检"承诺。

## License

本项目采用 [MIT License](LICENSE)。文档中引用的第三方数据集、模型与工具遵循其各自许可（见 [docs/10-oss-survey.md](docs/10-oss-survey.md)）；
其中 ultralytics（AGPL-3.0）与 X-AnyLabeling（GPLv3）仅在**本地训练/标注**环节使用，替换路径见 [ADR-0003](docs/adr/0003-model-choice-and-license.md)。

## 状态

- **M0 ✅**：架构与文档交付，文档自检全绿。
- **M1 ✅**：采集与标注闭环——导入（照片/视频抽帧/航拍切片、SHA256+pHash 去重、EXIF/GPS）、Web 标注台、任务租约、标注全量提交（差异+审计）、复核、数据集冻结（清单哈希）、三格式导出与往返校验。
- **M2 ✅**：模型辅助标注——检测器抽象 + **切片推理（SAHI 策略）**、批量/单图预标注（候选落库 `source=model`）、**候选逐条采纳/忽略**、**SAM 裂缝掩膜**（宽度/长度派生指标）、**预标注质量看板**（采纳率、模型-人工 IoU）、类别扩展 CLI；未装 ML 依赖时端点返回 501，人工流程不受影响。
- **M3 ✅（当前）**：训练闭环与门禁——ultralytics 微调 runner（**日志流 + 断点 + `flipud=0` 硬约束 + run_key 幂等**）、**分类别评估**（官方 val 口径 + 混淆矩阵 + 大小桶召回 + **切片开关对比** + 失败样例叠加图）、**模型门禁**（`candidate→validated→production`，劣化权重 409 拒绝并给出 delta）、**ONNX 导出包**（labels/preprocess/manifest/parity，**一致性验收 ≤1e-3**）、后台训练 API 与 `train`/`runs`/`model` CLI；未装 ML 依赖时返回 501。
- **M4 ✅（当前）**：边缘离线推理——`rdinspect infer`（目录/单图/视频/RTSP、**JSONL+CSV**、**断点续跑**、切片开关、命中快照、**性能基准**）、**启动前导出包校验**（schema 版本/模型 sha256/类别顺序/输出通道/**模型内嵌 names**，任一不匹配退出码 6 拒绝启动）、**离线安装包**（`scripts/build_edge_bundle.py` 生成 wheels+模型+脚本，`--verify` 在临时 venv 里离线安装并冒烟推理）；推理链路**不依赖 torch/ultralytics/数据库**。
- **M5 ✅（当前）**：主动学习与类别扩展——三类选样策略（**不确定性 50% + 错误驱动 30% + 多样性 20%**，ADR-0007）写回 `tasks.priority` 与 `active_queue` 明细、`--package` 时用 ONNX 取 **top1−top2** 证据、**同预算 A/B 效果对比**记录、**连续两次门禁失败 → 回看标注规范**告警、**新增类别不改代码**跑通全链路、统计报表（时间趋势/批次对比/类别覆盖/**GIS GeoJSON+CSV 导出**）。

## 快速开始（M1 + M2 + M3 + M4）

```bash
# 1) 安装（Python 3.11+；不污染其他项目环境）
python3 -m venv .venv && .venv/bin/pip install -e .
# 国内网络可用镜像：-i https://pypi.tuna.tsinghua.edu.cn/simple

# 1b) 模型辅助标注 + 训练闭环所需依赖（可选；CPU 版约 1.6GB，GPU 版把 index-url 换成 .../cu130）
.venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install ultralytics opencv-python-headless onnx onnxruntime onnxslim \
    -i https://pypi.tuna.tsinghua.edu.cn/simple

# 2) 启动工作站（默认 127.0.0.1:8787，标注台在 /，API 在 /api）
.venv/bin/rdinspect serve

# 3) 导入素材（目录必须在 configs/default.yaml 的 paths.allowed_roots 白名单内）
.venv/bin/rdinspect import --kind photo --input ./data/inbox/photos
.venv/bin/rdinspect import --kind video --input ./data/inbox/patrol.mp4 --fps 2
.venv/bin/rdinspect import --kind aerial --input ./data/inbox/aerial --tile   # 航拍切片

# 4) 浏览器打开 http://127.0.0.1:8787 标注 →「保存并提交」；复核走 /api/tasks/{id}/review
#    标注台右侧可「预标注本图 / 批量预标注」，候选可逐条采纳或忽略，并显示预标注质量面板

# 5) 模型辅助标注（CLI 等价入口）
.venv/bin/rdinspect prelabel --limit 200 [--model data/weights/yolo12s_rdd.pt] [--sam]
.venv/bin/rdinspect prelabel-metrics          # 采纳率 / 模型-人工一致性
.venv/bin/rdinspect models                    # 模型注册表（预标注权重）

# 6) 数据集：草稿 → 冻结（冻结后不可变；训练只接受 frozen）
.venv/bin/rdinspect dataset create --name ds-2026w38 --review-status approved
.venv/bin/rdinspect dataset freeze --name ds-2026w38
.venv/bin/rdinspect export --name ds-2026w38 --formats yolo,coco,labelme

# 7) 训练闭环：微调 → 评估 → 门禁 → 提升 → 导出 ONNX
.venv/bin/rdinspect train --dataset ds-2026w38 --arch data/weights/yolo11n.pt \
    --epochs 30 --imgsz 320 --batch 8 --device cpu        # 幂等：同参数重复提交直接复用既有 run
.venv/bin/rdinspect models                                # 状态 / mAP50 / 门禁结论一览
.venv/bin/rdinspect model evaluate --id 1 --split val     # 报告 + 失败样例叠加图
.venv/bin/rdinspect model validate --id 1                 # 门禁：未通过 → 退出码 5，状态保持 candidate
.venv/bin/rdinspect model promote  --id 1                 # validated → production（旧生产模型自动归档）
.venv/bin/rdinspect model export   --id 1 --imgsz 320     # ONNX 导出包 + ONNX↔.pt 一致性验收

# 8) 边缘离线推理（M4）：打包 → 离线安装 → 对目录/视频/流推理
python3 scripts/build_edge_bundle.py --package data/exports/<name>-<version>-320 \
    --out /tmp/edge-bundle --python-version 3.12 --verify      # 含目标机模拟（临时 venv 离线安装+冒烟推理）
.venv/bin/rdinspect infer --package data/exports/<name>-<version>-320 \
    --input /mnt/sd/photos --out ./edge-out --resume --threads 4
.venv/bin/rdinspect infer --package <导出包> --input patrol.mp4 --fps 2 --out ./edge-out   # 视频抽帧
.venv/bin/rdinspect infer --package <导出包> --input /mnt/sd --out ./bench --benchmark 60  # 性能基准

# 9) 主动学习：选样排队 → 标注台按队列标 → 同预算 A/B 对比
.venv/bin/rdinspect active queue --limit 100 --strategy hybrid --package data/exports/<name>-<version>-320
.venv/bin/rdinspect active show                       # 选样明细（得分/理由/证据）
.venv/bin/rdinspect active alerts --threshold 2        # 连续两次门禁不通过 → 回看标注规范
.venv/bin/rdinspect active compare --baseline 3 --candidate 4 --labeled 80   # 单位标注量的 mAP 增量

# 10) 统计报表（时间趋势 / 批次对比 / 类别覆盖 / GIS 导出）
.venv/bin/rdinspect report --format markdown --days 30
.venv/bin/rdinspect report --format gis-geojson --classes transverse_crack --out /tmp/hits.geojson

# 11) 类别扩展（无需改代码；顺序变化需新建数据集版本）
.venv/bin/rdinspect classes add --code water_puddle --zh 积水 --en "Water Puddle" --order 10

# 12) 统计与自检
.venv/bin/rdinspect stats --export csv
.venv/bin/rdinspect check
```

> 训练/评估/门禁/导出的完整操作手册见 [docs/11-training-guide.md](docs/11-training-guide.md)；
> 边缘打包与推理见 [docs/12-edge-inference.md](docs/12-edge-inference.md)；主动学习与扩类见 [docs/13-active-learning.md](docs/13-active-learning.md)。

权重下载（国内网络建议走镜像，详见 [docs/05-model-plan.md](docs/05-model-plan.md#51-可直接使用的公开权重m2-实测)）：
病害检测用 `rezzzq/yolo12s-road-damage-rdd2022`（MIT，RDD2022 训练）放 `data/weights/`；裂缝掩膜用
`sam2.1_t.pt` 或 `mobile_sam.pt`（可直接经 `ghfast.top` 前缀加速）。

无真实素材时可生成合成数据做回归（M1–M3 验收即用此方式）：

```bash
python3 scripts/make_sample_data.py --out /tmp/rd-sample --count 500   # 500 张 + 1 段视频
python3 scripts/e2e_m1.py --photos 500 --per-class 50                  # M1 端到端验收
python3 scripts/e2e_m2.py                                              # M2 端到端验收（含真实权重）
.venv/bin/python scripts/e2e_m3.py --photos 120 --epochs 30 --imgsz 320 # M3 端到端验收（真机训练，约 95 秒）
.venv/bin/python scripts/e2e_m4.py --package <导出包目录>                 # M4 端到端验收（离线/续跑/性能，约 1 分钟）
.venv/bin/python scripts/e2e_m5.py --photos 160 --epochs 15                # M5 端到端验收（选样/A-B/扩类/报表，约 2.5 分钟）
```

## 测试

```bash
PYTHONPATH=src:tests .venv/bin/python -m unittest discover -s tests -t .   # 301 项单测
python3 scripts/check_docs.py                                              # 文档/DDL/OpenAPI/样例自检
python3 scripts/e2e_m1.py                                                  # M1 端到端验收（合成素材）
python3 scripts/e2e_m2.py                                                  # M2 端到端验收（预标注/采纳/掩膜）
.venv/bin/python scripts/e2e_m3.py                                        # M3 端到端验收（真机训练/门禁/导出）
node scripts/check_annotator_ui.mjs                                        # 标注台静态检查（63 断言）
```

| 测试层 | 覆盖 | 结果 |
|---|---|---|
| 单测（301 项，1 跳过） | 几何换算、pHash 去重、EXIF/质量指标、导入幂等、任务租约、标注差异+审计、复核状态机、数据集划分/冻结/哈希稳定、三格式往返、HTTP 契约（含 400/404/409/501 语义）、预标注幂等与 NMS/切片、类别映射、采纳指标、掩膜派生指标、**检测匹配/AP/混淆矩阵/大小桶**、**训练硬约束与幂等键**、**门禁判定与状态机**、**导出包与 ONNX 预处理几何**、**边缘包校验/输出契约/断点续跑**、**主动学习打分与队列**、**报表口径（趋势/批次/GIS）** | 全部通过 |
| 端到端 M1 | 500 张合成照片 + 1 段视频 → 导入 → 四类各 50 张标注复核 → 冻结导出 → 往返零误差 → 哈希可复现 | 通过 |
| 端到端 M2 | 200 张预标注 → 采纳 120/忽略 40/待处理 40（采纳率 0.60、模型-人工 IoU 0.8626）→ 冻结导出 → 无模型时 501 降级 → 真实 RDD 权重映射率 1.00（18 候选） → SAM 掩膜宽度/长度派生 | 6/6 通过 |
| 端到端 M5 | 160 张图 → 种子模型 → 76 候选选 40（写优先级+明细+`strategy=active` 视图）→ **同预算 A/B：随机 0.166 vs 主动 0.209（Δ +0.043）** → **新增 water_puddle 全链路（包内 6 类）** → 连续两次门禁失败告警 → 报表/GIS 导出 | 6/6 通过 |
| 端到端 M4 | 无网代理（死端口）下 200 张图推理 → 断点续跑（100+100，重复 0）→ 与工作站逐框误差 **1e-06** → 四类篡改包**退出码 6 拒绝启动** → 640@4线程 **43.7 FPS**（目标 ≥15）→ 离线安装包 `--verify` 目标机模拟通过（整包 59MB） | 8/8 通过 |
| 端到端 M3 | 120 张合成图（202 真值框）冻结 → 30 epoch 微调（CPU 66s，末轮 mAP50 0.9345）→ val 评估官方 mAP50 **0.9125** / 内部口径 0.9167 → 首个模型按绝对下限过门禁并提升 production → **COCO 劣化权重被 409 拒绝**（整体 −0.9125 + 4 类类别塌陷）→ ONNX 导出**逐框坐标误差 1e-06**（容差 1e-3）→ HTTP 契约（注册表/运行详情/门禁 409）→ 同参数重复提交复用 run（0.0s） | 13/13 通过 |

## 目录结构

```
src/rdinspect/
├── config.py            配置加载 + allowed_roots 路径白名单 + 运行时目录本地化
├── cli.py               serve / import / tasks / dataset / export / stats / thumbs /
│                        check / prelabel / prelabel-metrics / models / classes
├── errors.py            领域异常（映射 404/409）
├── storage/{db,files,repo}.py     SQLite(WAL)+迁移、文件原子写/缩略图、仓储与审计
├── core/                 hashing(pHash) · geometry(坐标) · images(EXIF/质量/切片) ·
│                         ingest(导入去重) · formats(YOLO/COCO/LabelMe) · datasets(划分/冻结)
├── prelabel/             detector(协议+Ultralytics) · sahi(切片+NMS) · sam(掩膜) ·
│                         service(预标注服务) · metrics(采纳率/一致性)
├── train/                matching(纯函数匹配/AP) · runner(微调+幂等+硬约束) ·
│                         evaluate(分类别指标/混淆矩阵/切片对比/失败样例) ·
│                         gate(门禁状态机) · export_onnx(导出包+一致性验收) ·
│                         service(后台任务) · compat(受限容器兼容)
├── active/               scoring(选样打分) · service(队列/告警编排)
├── report.py             统计报表（趋势/批次/覆盖/GIS 导出）
├── edge/                 package(启动前校验) · sources(目录/视频/流) ·
│                         writers(JSONL/CSV/断点状态) · infer(编排+基准) ·
│                         onnx_runtime(letterbox/解码/NMS)
└── api/                  FastAPI 应用 + 静态标注台（单文件 1644 行、无构建、零外部依赖）
tests/                    301 项单测（unittest）
scripts/                  make_sample_data.py · e2e_m1.py · e2e_m2.py · e2e_m3.py ·
                          e2e_m4.py · e2e_m5.py · build_edge_bundle.py · check_docs.py ·
                          check_annotator_ui.mjs
```

> 说明：标注台按 ADR-0002 自建；设计文档里提到的 Vite+React+Konva 方案在 M1 用了「单文件原生 Canvas」实现（零构建、零外部依赖、体积更小），功能对齐 `docs/07-ui-spec.md` 的标注台要求（框选/8 向缩放/快捷键/预标注候选/草稿/亮度对比度）。
