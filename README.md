# road-inspect · 轻量级道路灾害巡查程序

> 识别道路上 **横向裂缝 / 纵向裂缝 / 坑洞 / 垃圾** 四类灾害（可扩展），支持**人工标注**与**模型辅助标注**，
> 覆盖「采集 → 标注 → 训练 → 导出 → 边缘离线推理」全链路。
> **当前状态：架构与文档（M0）已交付，业务实现从 M1 开始。**

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

- **M0（当前）**：架构与文档交付，自检全绿。
- **M1 起**：采集与标注闭环 → 模型辅助标注 → 训练门禁 → 边缘离线推理（见 [docs/09-roadmap.md](docs/09-roadmap.md)）。
