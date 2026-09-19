# road-inspect · 轻量级道路灾害巡查程序

> 识别道路上 **横向裂缝 / 纵向裂缝 / 坑洞 / 垃圾** 四类灾害（可扩展），支持**人工标注**与**模型辅助标注**，
> 覆盖「采集 → 标注 → 训练 → 导出 → 边缘离线推理」全链路。
> **当前状态：M0（架构文档）· M1（采集标注闭环）· M2（模型辅助标注）均已交付并在本机实测通过。**

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
| 数据库 DDL | `db/schema.sql` 建库、表/索引/视图/触发器、CHECK 约束生效 | 13 表 / 18 索引 / 2 视图 / 2 触发器 / 5 类别 |
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
- **M2 ✅（当前）**：模型辅助标注——检测器抽象 + **切片推理（SAHI 策略）**、批量/单图预标注（候选落库 `source=model`）、**候选逐条采纳/忽略**、**SAM 裂缝掩膜**（宽度/长度派生指标）、**预标注质量看板**（采纳率、模型-人工 IoU）、类别扩展 CLI；未装 ML 依赖时端点返回 501，人工流程不受影响。
- **M3 起**：训练与权重门禁 → 边缘离线推理（见 [docs/09-roadmap.md](docs/09-roadmap.md)）。

## 快速开始（M1 + M2）

```bash
# 1) 安装（Python 3.11+；不污染其他项目环境）
python3 -m venv .venv && .venv/bin/pip install -e .
# 国内网络可用镜像：-i https://pypi.tuna.tsinghua.edu.cn/simple

# 1b) 模型辅助标注所需依赖（可选；CPU 版约 1.6GB，GPU 版把 index-url 换成 .../cu130）
.venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install ultralytics opencv-python-headless -i https://pypi.tuna.tsinghua.edu.cn/simple

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

# 6) 数据集：草稿 → 冻结 → 导出
.venv/bin/rdinspect dataset create --name ds-2026w38 --review-status approved
.venv/bin/rdinspect dataset freeze --name ds-2026w38
.venv/bin/rdinspect export --name ds-2026w38 --formats yolo,coco,labelme

# 7) 类别扩展（无需改代码；顺序变化需新建数据集版本）
.venv/bin/rdinspect classes add --code water_puddle --zh 积水 --en "Water Puddle" --order 10

# 8) 统计与自检
.venv/bin/rdinspect stats --export csv
.venv/bin/rdinspect check
```

权重下载（国内网络建议走镜像，详见 [docs/05-model-plan.md](docs/05-model-plan.md#51-可直接使用的公开权重m2-实测)）：
病害检测用 `rezzzq/yolo12s-road-damage-rdd2022`（MIT，RDD2022 训练）放 `data/weights/`；裂缝掩膜用
`sam2.1_t.pt` 或 `mobile_sam.pt`（可直接经 `ghfast.top` 前缀加速）。

无真实素材时可生成合成数据做回归（M1/M2 验收即用此方式）：

```bash
python3 scripts/make_sample_data.py --out /tmp/rd-sample --count 500   # 500 张 + 1 段视频
python3 scripts/e2e_m1.py --photos 500 --per-class 50                  # M1 端到端验收
python3 scripts/e2e_m2.py                                              # M2 端到端验收（含真实权重）
```

## 测试

```bash
PYTHONPATH=src:tests .venv/bin/python -m unittest discover -s tests -t .   # 67 项单测
python3 scripts/check_docs.py                                              # 文档/DDL/OpenAPI/样例自检
python3 scripts/e2e_m1.py                                                  # M1 端到端验收（合成素材）
python3 scripts/e2e_m2.py                                                  # M2 端到端验收（预标注/采纳/掩膜）
node scripts/check_annotator_ui.mjs                                        # 标注台静态检查（63 断言）
```

| 测试层 | 覆盖 | 结果 |
|---|---|---|
| 单测（67 项，1 跳过） | 几何换算、pHash 去重、EXIF/质量指标、导入幂等、任务租约、标注差异+审计、复核状态机、数据集划分/冻结/哈希稳定、三格式往返、HTTP 契约（含 400/404/409/501 语义）、预标注幂等与 NMS/切片、类别映射、采纳指标、掩膜派生指标 | 全部通过 |
| 端到端 M1 | 500 张合成照片 + 1 段视频 → 导入 → 四类各 50 张标注复核 → 冻结导出 → 往返零误差 → 哈希可复现 | 通过 |
| 端到端 M2 | 200 张预标注 → 采纳 120/忽略 40/待处理 40（采纳率 0.60、模型-人工 IoU 0.8626）→ 冻结导出 → 无模型时 501 降级 → 真实 RDD 权重映射率 0.895 → SAM 掩膜宽度/长度派生 | 6/6 通过 |

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
└── api/                  FastAPI 应用 + 静态标注台（单文件 1644 行、无构建、零外部依赖）
tests/                    67 项单测（unittest）
scripts/                  make_sample_data.py · e2e_m1.py · e2e_m2.py ·
                          check_docs.py · check_annotator_ui.mjs
```

> 说明：标注台按 ADR-0002 自建；设计文档里提到的 Vite+React+Konva 方案在 M1 用了「单文件原生 Canvas」实现（零构建、零外部依赖、体积更小），功能对齐 `docs/07-ui-spec.md` 的标注台要求（框选/8 向缩放/快捷键/预标注候选/草稿/亮度对比度）。
