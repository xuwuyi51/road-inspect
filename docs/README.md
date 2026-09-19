# road-inspect 文档索引

> 架构设计文档集（M0 交付）。阅读顺序即编号顺序；决策记录见 `adr/`。

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
