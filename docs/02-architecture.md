# 02 · 架构设计（Architecture）

## 1. 架构原则

1. **单机优先、零重型依赖**：SQLite + 文件目录 + 单进程服务；不引入消息队列/对象存储/容器编排。
2. **同一契约贯穿全链路**：类别定义、标注格式、后处理代码在标注、训练、导出、边缘推理四处复用一份实现，杜绝格式漂移。
3. **数据不可变、版本可回放**：原始文件只增不改；标注以事件形式留痕；训练只吃"冻结数据集"。
4. **模型可替换**：推理层通过统一 `Detector` 接口屏蔽 YOLO/RTMDet/ONNX 差异；类别扩展不改代码。
5. **边缘无缝**：工作站训练 → ONNX 导出 → 边缘 CLI，后处理与阈值来自导出包，避免"两套逻辑"。

## 2. 上下文图（系统边界）

```
        ┌──────────────┐  照片/视频/航拍大图   ┌───────────────────────────────┐
        │ 巡检员（采集） │ ───────────────────▶ │                               │
        └──────────────┘                      │        road-inspect           │
        ┌──────────────┐   标注/复核（浏览器）  │  ┌─────────────────────────┐  │
        │ 标注员/复核员 │ ◀──────────────────▶ │  │ 工作站形态 Web 应用      │  │
        └──────────────┘                      │  └─────────────────────────┘  │
        ┌──────────────┐   冻结/训练/导出      │  ┌─────────────────────────┐  │
        │ 算法/运维     │ ◀──────────────────▶ │  │ 边缘形态 推理 CLI        │  │
        └──────────────┘                      │  └─────────────────────────┘  │
        ┌──────────────┐   数据/模型下载(可选)  └───────────────┬───────────────┘
        │ 公开数据集源  │ ────────────────────────────────────▶ │
        │ (RDD2022/HF) │                                       ▼
        └──────────────┘                          ┌───────────────────────┐
        ┌──────────────┐   模型/结果文件拷贝       │ 文件系统：原图/抽帧/标注/ │
        │ 巡检车/随身设备│ ◀─────────────────────▶ │ 掩膜/导出包 + SQLite   │
        └──────────────┘                          └───────────────────────┘
```

## 3. 组件图

```
┌──────────────────────────── 工作站（单机，浏览器访问） ─────────────────────────────┐
│                                                                                  │
│  web/                          api/ (FastAPI)                                     │
│  ┌────────────────┐   REST    ┌──────────────────────────────────────────────┐   │
│  │ 导入台          │ ◀────────▶ │ routers: ingest/tasks/annotations/review/    │   │
│  │ 标注台（Canvas）│           │          datasets/train/models/export/stats  │   │
│  │ 复核台          │           └───────┬──────────────────────────────────────┘   │
│  │ 数据集/模型/统计 │                  │                                          │
│  └────────────────┘                  ▼                                          │
│                       ┌──────────────────────── core/ ────────────────────────┐  │
│                       │ ingest（EXIF/抽帧/切片/去重）  taxonomy（类别注册表）   │  │
│                       │ tasks（状态机/锁）             annotations（CRUD/审计）  │  │
│                       │ datasets（选样/冻结/导出）      metrics（评估/一致性）    │  │
│                       └───────┬───────────────────────┬───────────────────────┘  │
│                               │                       │                          │
│             ┌─────────────────▼────────┐   ┌──────────▼───────────────────────┐  │
│             │ prelabel/ 预标注服务      │   │ train/ 训练与导出                │  │
│             │ · YOLO11 检测（GPU/CPU）  │   │ · ultralytics 微调              │  │
│             │ · SAM 提示式掩膜          │   │ · 评估/混淆矩阵/门禁             │  │
│             │ · 零样本垃圾检测          │   │ · ONNX 导出 + 导出包            │  │
│             │ · SAHI 切片合并（共用）    │   └──────────┬───────────────────────┘  │
│             └──────────────────────────┘              │                          │
│                                                       │ 导出包（模型+labels+配置）│
│   ┌────────────────────────────── storage/ ───────────┴───────────────────────┐  │
│   │ SQLite（WAL）：images/tasks/annotations/classes/dataset_versions/         │  │
│   │                model_versions/runs/audit_log                              │  │
│   │ 文件目录：raw/ frames/ tiles/ masks/ exports/ datasets/  + 哈希索引         │  │
│   └───────────────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────────────┘
                                        │ 拷贝导出包（或用 rsync/U 盘）
┌────────────────────── 边缘（车载/巡检，无网无数据库） ─────────────────────────────┐
│  rdinspect infer --input <dir|video|rtsp> --out <dir>                            │
│  → onnxruntime/TensorRT + 同一份后处理（NMS/切片合并/阈值）                        │
│  → results.jsonl / results.csv（含类别、置信度、框、时间戳、GPS 若可读）             │
└──────────────────────────────────────────────────────────────────────────────────┘
```

## 4. 组件职责

| 组件 | 职责 | 关键技术 |
|---|---|---|
| `api` | HTTP 契约、鉴权（首版仅本机）、任务编排、错误处理 | FastAPI + Pydantic |
| `web` | 导入台、标注台、复核台、数据集/模型/统计页 | Vite + React + Konva（或原生 Canvas） |
| `core/ingest` | EXIF 解析、视频抽帧、大图切片、去重（SHA256+pHash） | Pillow、ffmpeg、imagehash |
| `core/taxonomy` | 类别注册表（DB + `taxonomy.yaml` 双写校验） | — |
| `core/tasks` | 任务状态机、领取/释放、重试、幂等（按文件哈希） | SQLite 事务 + 行锁 |
| `core/annotations` | 标注 CRUD、来源标记、掩膜存储（RLE/PNG）、审计 | Pydantic 校验 |
| `core/datasets` | 选样、划分（按路段分组避免泄漏）、冻结、导出 | 生成 `data.yaml` + 哈希清单 |
| `prelabel` | 预标注：检测 / 掩膜 / 零样本；切片与合并 | ultralytics、SAM2、SAHI 策略 |
| `train` | 微调、评估、混淆矩阵、门禁、ONNX 导出 | ultralytics + onnx |
| `edge` | 离线推理 CLI、结果导出 | onnxruntime（GPU 可选 TensorRT） |
| `storage` | SQLite 仓储层与文件仓储层（唯一写入入口） | SQLite WAL + 原子写（临时文件 + rename） |

## 5. 推荐目录结构（实现阶段落地）

```
road-inspect/
├── pyproject.toml
├── README.md
├── configs/
│   ├── default.yaml            # 服务与路径
│   ├── prelabel.yaml           # 预标注阈值/模型
│   ├── train.yaml              # 训练配方
│   └── edge.yaml               # 边缘推理参数
├── db/schema.sql               # 建库 DDL（本设计已交付）
├── src/rdinspect/
│   ├── api/            # FastAPI 路由与依赖
│   ├── core/           # ingest/taxonomy/tasks/annotations/datasets/metrics
│   ├── storage/        # db.py（SQLite 仓储）、files.py（文件仓储/原子写）
│   ├── prelabel/       # detector.py、sam.py、zeroshot.py、sahi.py
│   ├── train/          # runner.py、evaluate.py、gate.py、export_onnx.py
│   ├── edge/           # infer.py（CLI 主体）
│   └── cli.py          # rdinspect serve|import|prelabel|train|export|infer
├── web/                # 前端工程（标注台等）
├── tests/              # 单测 + 端到端小样本测试
├── data/               # 运行期数据根（可指向外接盘）
│   ├── raw/ frames/ tiles/ masks/ thumbs/
│   ├── datasets/<version>/  exports/
│   └── app.db
└── docs/               # 本设计文档集
```

## 6. 数据流（主链路）

```
上传/目录 ──▶ ingest ──▶ 去重&归一化 ──▶ images 表 + 文件入库
                                   │
                                   ├─▶ 任务生成（tasks: pending）
                                   ▼
                        （可选）预标注 prelabel ──▶ 候选框/掩膜（source=model）
                                   ▼
                        标注台（人工修正）──▶ annotations（source=human/model_edited）──▶ 审计
                                   ▼
                        复核（抽检/低置信度优先）──▶ approved / rejected
                                   ▼
                        数据集选样 ──▶ dataset_version（draft）──▶ 冻结（frozen + 哈希清单）
                                   ▼
                        训练 run ──▶ 评估 ──▶ 门禁 ──▶ model_version（candidate→validated→production）
                                   ▼
                        导出包（ONNX + labels + preprocess）──▶ 边缘 infer ──▶ results.jsonl
                                   ▼
                        （可选）结果回灌工作站 ──▶ 复核 ──▶ 主动学习选样
```

## 7. 关键时序

### 7.1 导入 + 预标注

```
巡检员 → POST /api/ingest/batches (文件/目录)
api    → ingest: EXIF/时间/GPS → SHA256 查重 → pHash 近似查重
       → 大图切片（可选）/ 视频抽帧（异步任务）
       → 写 images + tasks(pending)，返回 batch 摘要
标注员 → POST /api/tasks/{id}/prelabel（或批量 /api/prelabel/batches）
prelabel → 载入现役模型 → 切片推理（若大图）→ 合并 → 落库候选（source=model）
       → 失败则标记 prelabel_failed（不阻塞人工）
```

### 7.2 标注 + 复核

```
标注员 → GET /api/tasks?status=prelabeled&limit=50（领取 → annotating）
       → PUT /api/tasks/{id}/annotations（全量覆盖式提交，服务端 diff 写审计）
       → POST /api/tasks/{id}/submit → annotated
复核员 → GET /api/reviews?strategy=low_conf|random
       → POST /api/reviews/{id}/approve | reject(reason) → approved / annotating
```

### 7.3 训练 + 导出 + 门禁

```
算法 → POST /api/datasets {filter...} → draft
     → POST /api/datasets/{id}/freeze → 固化清单 + 生成 data.yaml + 哈希
     → POST /api/train/runs {dataset, …} → 训练任务（后台线程；202 + 轮询 GET /api/train/runs/{id}）
train → 训练 → 评估（分类别指标）→ 写 runs/metrics
gate → 与现役 production 权重对比（同测试集）→ pass/fail
     → pass: model_version = candidate→validated；人工确认 → production
     → POST /api/models/{id}/export → exports/<name>-onnx/（模型+labels+preprocess+清单）
```

## 8. 部署拓扑

| 形态 | 进程 | 存储 | 说明 |
|---|---|---|---|
| 工作站 | `rdinspect serve`（systemd user 服务）+ 训练子进程 | `<data>/app.db` + 文件目录 | 浏览器访问 `127.0.0.1:8787`；局域网可选放开 |
| 边缘 | `rdinspect infer`（一次性 CLI，无守护） | 只读模型目录 + 输出目录 | 无 Python Web 依赖；ONNXRuntime 可离线安装 |

资源预算：工作站服务 CPU < 5% 空闲占用、RAM < 300MB；预标注 GPU 峰值 < 4GB（YOLO11s + 1024 tile）；训练 GPU < 14GB（640, batch 16, YOLO11s）。

## 9. 扩展点

| 扩展需求 | 扩展点 | 改动范围 |
|---|---|---|
| 新增灾害类别 | `taxonomy.yaml` + `classes` 表注册 | 无需改代码；数据集与导出自动携带 |
| 更换检测模型 | `prelabel/detector.py` 的 `Detector` 接口（`load/predict`） | 仅新增一个适配器 |
| 新增标注形态（多边形/关键点） | `annotations.kind` 扩展 + 前端工具条插件 | 数据库字段已预留 `geometry_json` |
| 新增采集源（RTSP/新相机） | `ingest/sources/*` 适配器 | 仅新增适配器 |
| 多模型并行（坑洞专用模型） | `model_versions.task` 字段 + 路由策略 | 推理层按 className 路由 |
| 结果接入 GIS/工单 | `stats` 导出字段（经纬度/路段/方位）+ Webhook | 预留字段，不改核心表 |

## 10. 可靠性与一致性设计

- **写入原子性**：所有文件先写 `.tmp` 再 `rename`；SQLite 使用 WAL + 事务，标注提交为单事务。
- **幂等**：导入以文件内容哈希为幂等键；预标注/训练任务带 `run_key`，重复提交返回既有结果。
- **任务锁**：`tasks.lease_until` + 领取时 `UPDATE ... WHERE status='pending' AND lease_until < now`，避免重复领取。
- **失败降级**：预标注失败 → 纯人工；SAM 失败 → 仅框标注；训练失败 → 保留日志与断点权重。
- **可复现**：冻结数据集哈希 + 训练配置 + 随机种子 + 产出权重哈希，四下齐备才算一次可复现实验。

## 11. 技术选型与替代

| 领域 | 选型 | 理由 | 备选（若许可/性能变化） |
|---|---|---|---|
| 服务框架 | FastAPI + Pydantic v2 | 轻、类型安全、自带 OpenAPI | Flask（更简但需手写校验） |
| 存储 | SQLite(WAL) + 文件目录 | 零运维、单机够用 | PostgreSQL（多用户时） |
| 检测模型 | ultralytics YOLO11n/s | 生态成熟、导出 ONNX 便捷、预标注快 | RTMDet / YOLOX（Apache-2.0，闭源分发时） |
| 掩膜 | SAM2（提示式分割） | 裂缝掩膜只需点/框提示，无需训练 | EfficientSAM / MobileSAM（更轻） |
| 垃圾类冷启动 | 开放词表检测（GroundingDINO/OWLv2） | 无专用数据集时先出草稿 | YOLO-World |
| 切片推理 | SAHI 策略 | 细裂缝/航拍小目标召回显著提升 | 自研 tile+NMS（逻辑相同） |
| 视频抽帧 | ffmpeg | 系统已有、稳定 | OpenCV VideoCapture |
| 前端 | Vite + React + Konva | 画布标注成熟、构建产物轻 | 原生 Canvas（更小但开发慢） |
| 边缘运行时 | onnxruntime（可选 TensorRT） | 跨平台、无 Python-UI 依赖 | OpenVINO（Intel 设备） |

## 12. 与现有环境集成

- **不污染既有环境**：项目自建 `uv venv`（Python 3.11）；训练复用 `torch` CUDA 轮子即可，无需改动 DiffSynth 的 venv。
- **复用系统能力**：ffmpeg（抽帧/转码）、系统 `onnxruntime 1.28`（边缘推理可直接用）。
- **服务端口**：默认 `127.0.0.1:8787`，与 DSH GUI（3080）不冲突；需局域网访问时显式配置并给出防火墙提示。
- **可选集成**：后续可提供 DSH 插件（`road_inspect_*` 工具）让 Agent 触发导入/预标注/统计，非本期范围。

## 13. 架构决策索引

| ADR | 主题 |
|---|---|
| [ADR-0001](./adr/0001-detection-first-mask-optional.md) | 检测优先 + 裂缝掩膜可选 |
| [ADR-0002](./adr/0002-self-hosted-lightweight-annotation-ui.md) | 自建轻量标注台，不部署 CVAT/Label Studio |
| [ADR-0003](./adr/0003-model-choice-and-license.md) | YOLO11 首选与 AGPL 边界/替换路径 |
| [ADR-0004](./adr/0004-sqlite-and-file-storage.md) | SQLite + 文件存储 + 双哈希去重 |
| [ADR-0005](./adr/0005-dataset-freeze-and-hash.md) | 数据集冻结与哈希可复现 |
| [ADR-0006](./adr/0006-sahi-sliced-inference.md) | 切片推理内置 |
| [ADR-0007](./adr/0007-active-learning-and-model-gate.md) | 主动学习与权重门禁 |
