# 10 · 开源项目调研（OSS Survey）

> 目的：不重复造轮子，明确"直接依赖 / 仅借鉴 / 只做格式兼容 / 不采用"四类取舍。
> 许可以调研当日仓库与包管理器声明为准；采用前需复核（尤其数据集许可）。
> **外链可达性实测**：Kaggle 镜像 200；`huggingface.co` 直连超时（用 `hf-mirror.com` 替代，API 实测可用）；Wiley 论文页 403（可能需机构访问权限，可从 DOI 检索其他入口）。

## 1. 数据集

| 项目 | 链接 | 内容 | 许可 | 取舍 |
|---|---|---|---|---|
| **RDD2022**（Road Damage Dataset） | [论文 PDF（Wiley，可能需机构权限）](https://onlinelibrary.wiley.com/doi/pdfdirect/10.1002/gdj3.260) · [Kaggle 镜像（实测 200）](https://www.kaggle.com/datasets/sreekaraditya/rdd2022-yolo-crackscan-v2) | 6 国 47k+ 图，D00 纵向/D10 横向/D20 网裂/D40 坑洞 | 数据集自带声明（需按其要求引用） | **直接使用**：3 类映射 + 类别定义来源 |
| **Unified Road Defect Dataset** | [HF 原始](https://huggingface.co/datasets/TamAko783/Unified_Road_Defect_Dataset)（直连不通）→ 改用 `https://hf-mirror.com/datasets/TamAko783/Unified_Road_Defect_Dataset` | RDD2022 + UAV-PDD2023 + RoadDamageVision 合并为四类 CRDDC，YOLO 格式 | `other / mixed-source-attribution`（需同时引用三个源数据集） | **直接使用**：主训练集；注意按 README 附带 `SOURCES.md` |
| **UAV-PDD2023**（并入上者） | 见上 | 无人机路面病害（LC/TC/AC/PH） | 见上 | 间接使用：航拍切片策略的验证数据 |
| **TACO**（垃圾数据） | GitHub `pedropro/TACO` | 通用垃圾图像检测 | 需查仓库声明（CC BY 4.0 常见） | **借鉴/辅助**：垃圾类冷启动与预训练，不直接混入道路数据训练（域差异大） |

## 2. 检测与分割模型

| 项目 | 许可 | 价值 | 取舍 |
|---|---|---|---|
| **Ultralytics YOLO11** | **AGPL-3.0** | 训练/导出/预标注一条龙，生态最成熟 | **采纳（内部自用）**；闭源分发需替换（ADR-0003） |
| **RTMDet / MMDetection** | Apache-2.0 | 许可宽松、可做替换路径；小目标表现好 | **备选**：闭源分发时的检测骨干 |
| **YOLOX** | Apache-2.0 | 轻量、部署简单 | **备选**：边缘端替代 |
| **SAHI** | MIT | 切片推理与合并，细裂缝/航拍小目标必备 | **采纳**：策略实现（可自研等价逻辑，保持零依赖） |
| **SAM2 / EfficientSAM** | Apache-2.0（Meta） | 提示式分割，裂缝掩膜无需训练 | **采纳（可选模块）** |
| **GroundingDINO / OWLv2** | Apache-2.0 | 开放词表零样本，垃圾类冷启动 | **采纳（可选模块）** |
| **DeepCrack / CrackFormer** | 各自仓库声明 | 裂缝分割经典方案 | **借鉴**：后续若做裂缝分割任务的基线 |
| **LDA-YOLO11 等轻量化改进** | 论文 | 边缘部署的注意力/细节增强思路 | **借鉴**：边缘精度不足时再评估 |

## 3. 标注与数据工具

| 项目 | 许可 | 价值 | 取舍 |
|---|---|---|---|
| **X-AnyLabeling** | **GPLv3** | 桌面标注 + YOLO/SAM 模型辅助，开箱即用 | **不嵌入**；作为**可选外部工具**（格式互通 LabelMe JSON），重标注场景使用 |
| **CVAT** | 开源版（需确认具体版本许可） | 功能完整、支持自动化标注 | **不采用**：容器编排重（违反轻量约束）；仅借鉴其"自动标注 + 人机复核"交互 |
| **Label Studio** | 开源版（需确认） | 通用标注平台 + ML backend | **不采用**：同上；重 |
| **LabelMe / labelImg** | MIT / MIT | 经典格式与轻量工具 | **只做格式兼容**（LabelMe JSON 导入导出） |
| **Roboflow** | SaaS | 数据托管与增强 | **不采用**：数据外传与订阅成本 |

## 4. 工程与部署参考

| 项目 | 价值 | 取舍 |
|---|---|---|
| **FastAPI + Pydantic** | 轻量服务、自动 OpenAPI | 采纳 |
| **SQLite(WAL)** | 单机零运维 | 采纳 |
| **ONNXRuntime** | 跨平台边缘推理 | 采纳（本机已有 1.28） |
| **ffmpeg** | 抽帧/转码 | 采纳（系统已有） |
| **imagehash（pHash）** | 近似去重 | 采纳（或自实现 DCT 哈希，保持零依赖） |
| **Konva** | 画布标注交互 | 采纳 |

## 5. 结论：本项目自研 vs 复用

| 能力 | 结论 |
|---|---|
| 检测/分割模型训练 | **复用** ultralytics（许可已确认可接受） |
| 切片推理 | **复用策略**（SAHI 思路），实现保持零依赖 |
| 掩膜辅助 | **复用** SAM2（可选模块） |
| 垃圾类冷启动 | **复用** 零样本检测模型 |
| 标注台 | **自研**（轻量 Web + 与训练闭环同库），外部工具仅做格式互通 |
| 数据管理/版本/门禁 | **自研**（现有开源平台不覆盖"数据集冻结 + 权重门禁 + 边缘导出"这套组合） |
| 边缘推理 | **自研 CLI** + ONNXRuntime |

> 一句话：**模型与算法尽量复用，流程与数据治理自研**——后者正是本项目的差异化价值，也是轻量约束下最容易被忽视的部分。
