# road-inspect 文档索引

> 架构设计文档集（M0 交付，随 M1–M3 实现同步更新）。阅读顺序即编号顺序；决策记录见 `adr/`。

| # | 文档 | 一句话 |
|---|---|---|
| 00 | [overview](./00-overview.md) | 项目目标、范围、术语、环境约束 |
| 01 | [requirements](./01-requirements.md) | 角色、场景、FR/NFR、状态机、验收标准 |
| 02 | [architecture](./02-architecture.md) | 上下文/组件/数据流/时序/部署/扩展点 |
| 03 | [data-model](./03-data-model.md) | 数据库与文件布局、格式映射、去重、冻结算法、容量估算 |
| 04 | [annotation-workflow](./04-annotation-workflow.md) | 人工标注 SOP、判定规则、模型辅助、复核与一致性、效率基线 |
| 05 | [model-plan](./05-model-plan.md) | 类别体系、数据来源与规模、训练配方、指标目标、导出与边缘预算 |
| 06 | [api-spec](./06-api-spec.md) + [openapi.yaml](./openapi.yaml) | REST 与 CLI 契约、错误码、幂等语义 |
| 07 | [ui-spec](./07-ui-spec.md) | 页面信息架构、标注台交互、异常态、前端约定 |
| 08 | [deployment](./08-deployment.md) | 工作站/边缘部署、资源预算、备份、安全与合规 |
| 09 | [roadmap](./09-roadmap.md) | M0–M5 里程碑、任务分解、风险登记 |
| 10 | [oss-survey](./10-oss-survey.md) | 开源项目调研与取舍（含许可） |
| 11 | [training-guide](./11-training-guide.md) | M3 训练闭环操作手册：训练→评估→门禁→导出与排障速查 |
| 12 | [edge-inference](./12-edge-inference.md) | M4 边缘离线推理操作手册：离线打包、infer 用法、输出契约、性能与包校验 |
| 13 | [active-learning](./13-active-learning.md) | M5 主动学习与类别扩展：三类选样策略、同预算 A/B、增量重训与告警、报表与 GIS 导出 |
| — | [adr/](./adr/) | ADR-0001 ~ 0007 关键决策 |
| — | [format-samples/](./format-samples/) | YOLO/COCO/LabelMe 三格式样例与对照表 |

## 决策记录（ADR）

| ADR | 决策 |
|---|---|
| [0001](./adr/0001-detection-first-mask-optional.md) | 检测优先，裂缝掩膜可选 |
| [0002](./adr/0002-self-hosted-lightweight-annotation-ui.md) | 自建轻量标注台，不部署 CVAT/Label Studio |
| [0003](./adr/0003-model-choice-and-license.md) | YOLO11 首选、许可边界与替换路径 |
| [0004](./adr/0004-sqlite-and-file-storage.md) | SQLite + 文件目录 + 双哈希去重 |
| [0005](./adr/0005-dataset-freeze-and-hash.md) | 数据集冻结与清单哈希 |
| [0006](./adr/0006-sahi-sliced-inference.md) | 内置切片推理 |
| [0007](./adr/0007-active-learning-and-model-gate.md) | 主动学习与权重门禁 |

## 相关工件

- [`../db/schema.sql`](../db/schema.sql)：可直接建库的 DDL
- [`../configs/`](../configs/)：default / prelabel / train / edge 四份配置
- [`../README.md`](../README.md)：项目入口与自检结果

## 变更记录

| 版本 | 日期 | 变更 |
|---|---|---|
| v1.0 | 2026-09-12 | 初版：M0 架构与文档交付（含 DDL、OpenAPI 草案、格式样例、7 份 ADR） |
| v1.1 | 2026-09-19 | M1 采集标注闭环 + M2 模型辅助标注落地，文档同步实测数据 |
| v1.2 | 2026-09-19 | M3 训练闭环与门禁落地：新增 11-training-guide；05/06/08/09 与 openapi 同步 M3 端点与实测结果 |
| v1.4 | 2026-09-19 | M5 主动学习与类别扩展落地：新增 13-active-learning 与 `active_queue` 表（migration 0002）；06 补 M5 端点 |
| v1.3 | 2026-09-19 | M4 边缘离线推理落地：新增 12-edge-inference；06 §8 改为已实现契约（新增退出码 6）、08/09 同步实测数据 |
