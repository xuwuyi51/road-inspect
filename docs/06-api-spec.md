# 06 · 接口规范（API & CLI）

> 机器可读契约：[`openapi.yaml`](./openapi.yaml)（OpenAPI 3.1，已校验：32 路径 / 36 操作 / 36 schema / 53 引用，无悬空引用）。
> 覆盖范围：M1 采集/标注/复核、M2 模型辅助标注、M3 训练闭环（本文档按**实现现状**编写；
> 契约与实现如有出入，以 `src/rdinspect/api/app.py` 与 `src/rdinspect/cli.py` 为准，§6.5 列出已知差异）。

## 1. 基本约定

| 项 | 约定 |
|---|---|
| Base URL | `http://127.0.0.1:8787`（默认仅本机；需局域网时显式配置监听并提示防火墙） |
| 认证 | 首版无账号；仅监听本机。若放开局域网必须配合反代鉴权（文档中标注为部署前置条件） |
| 内容类型 | JSON（`application/json`）；影像字节流为 `image/jpeg` / `image/png` |
| 时间 | UTC ISO8601 文本 `YYYY-MM-DDTHH:MM:SSZ`（前端按本地时区显示） |
| 坐标 | 归一化 `[0,1]`，明确相对"所属影像"（切片则为该 tile） |
| 分页 | `limit`（1–500，默认 50）+ `cursor`（不透明字符串）；响应 `{items, next_cursor}` |
| 幂等 | **`Idempotency-Key` 请求头未实现**（服务端忽略它）。真实幂等键是数据库唯一键：影像 `sha256`、数据集 `manifest_hash`、长任务 `runs.run_key`（见 §5） |
| 长任务 | 训练：`POST /api/train/runs` → **202**（异步，`wait=false`）/ **200**（同步，`wait=true`，训练已结束），轮询 `GET /api/train/runs/{runId}`（含 `log_tail`），`POST /api/train/runs/{runId}/cancel` 协作式取消（见 §6.1）。`GET /api/runs` 只提供运行记录列表，**没有** `POST`/详情/取消（见 §2） |
| 错误 | 统一 `{code, message}`（`code` 恒为 `http_<status>`）；请求参数/schema 校验失败为 **400**，带 `details.errors`（见 §4） |
| 上传限制 | 单文件 ≤ `ingest.max_file_mb`（默认 50MB，超限 **413**）；`source_dir` 必须落在 `configs/default.yaml` 的 `allowed_roots` 白名单内 |
| 版本 | 契约版本随 `info.version`；不兼容变更升主版本并在导出包 `manifest.json` 中标注 |

## 2. 实现状态（M1–M3 已实现）

| 状态 | 端点 |
|---|---|
| ✅ M1 已实现 | `/api/health`、`/api/classes`(GET/POST)、`/api/ingest/batches`(POST/GET)、`/api/ingest/upload`、`/api/ingest/batches/{id}`、`/api/images`、`/api/images/{id}/file`、`/api/tasks`、`/api/tasks/lease`、`/api/tasks/{id}`、`/api/tasks/{id}/annotations`(PUT)、`/api/tasks/{id}/adopt-candidates`、`/api/tasks/{id}/submit`、`/api/tasks/{id}/review`、`/api/datasets`(GET/POST)、`/api/datasets/{id}/freeze`、`/api/datasets/{id}/export`、`/api/runs`(仅 GET)、`/api/stats/overview`、`/api/stats/export` |
| ✅ M2 已实现 | `/api/tasks/{id}/prelabel`、`/api/prelabel/batches`、`/api/prelabel/metrics`、`DELETE /api/annotations/{id}`（忽略候选）、`POST \| GET /api/annotations/{id}/mask`（SAM 掩膜）、`GET /api/models` |
| ✅ M3 已实现 | `POST /api/train/runs`（异步微调）、`GET /api/train/runs/{id}`（轮询进度）、`POST /api/train/runs/{id}/cancel`、`GET /api/models/registry`、`GET /api/models/{id}`、`POST /api/models/{id}/evaluate`、`POST /api/models/{id}/validate`、`POST /api/models/{id}/promote`、`POST /api/models/{id}/export` |
| 🚧 M4/M5 未实现（**501**） | 边缘推理（`POST /api/infer` 等）、主动学习选样（`POST /api/active-learning/queue` 等）——所有**未注册的 POST 路径**都落到兜底路由 `POST /api/{rest:path}`，返回 501 与中立文案「`/api/{rest}` 未实现：当前版本已实现 M1–M3 端点（导入/标注/复核/预标注/训练/门禁/导出），M4（边缘推理）与 M5（主动学习）规划中；已实现端点清单见 docs/06-api-spec.md」（**不再误报**某条路径属于某个里程碑）；同一路径的 GET 返回 **405**。CLI 侧对应 `rdinspect infer`（子命令未注册，argparse 直接以退出码 2 报错） |
| ⚠️ 历史设计、从未实现 | `POST /api/runs`（启动训练/评估）、`GET /api/runs/{id}`、`POST /api/runs/{id}/cancel` 是 M1 期设计，M3 起由 `/api/train/runs*` 承载。这三者仍未实现：POST 落 501 兜底（文案已中立化）、GET 落 405。`openapi.yaml` 中已标 `deprecated: true` |

> 停用声明：`/api/models/{id}/promote|export` 在 M2 文档里曾被描述为「返回 501」——**已不成立**，M3 起两者都是同步实现的真实端点。

## 3. 端点总览

| Tag | 方法 | 路径 | 用途 |
|---|---|---|---|
| health | GET | `/api/health` | 服务与依赖状态（schema 版本、影像/任务计数） |
| classes | GET / POST | `/api/classes` | 类别注册表读取 / 新增（扩展点） |
| ingest | POST / GET | `/api/ingest/batches` | 创建导入批次（登记目录）/ 批次列表 |
| ingest | POST | `/api/ingest/upload` | 上传文件导入（multipart，异步 202） |
| ingest | GET | `/api/ingest/batches/{batchId}` | 批次详情、导入结果统计与失败清单 |
| images | GET | `/api/images` | 影像查询（批次/来源过滤） |
| images | GET | `/api/images/{imageId}/file` | 原始影像或缩略图（`?thumb=1`） |
| tasks | GET | `/api/tasks` | 任务队列（`strategy=low_conf` 用于主动学习优先队列） |
| tasks | POST | `/api/tasks/lease` | 批量领取任务（写租约，防重复领取） |
| tasks | GET | `/api/tasks/{taskId}` | 任务详情（影像 + 标注 + 预标注候选） |
| annotations | PUT | `/api/tasks/{taskId}/annotations` | 全量提交标注（覆盖式，服务端 diff + 审计） |
| tasks | POST | `/api/tasks/{taskId}/submit` | 提交进入复核 |
| prelabel | POST | `/api/tasks/{taskId}/prelabel` | 单任务预标注 |
| prelabel | POST | `/api/prelabel/batches` | 批量预标注（按条件选任务） |
| reviews | POST | `/api/tasks/{taskId}/review` | 复核决定（approve / reject + 原因码） |
| datasets | GET / POST | `/api/datasets` | 数据集版本列表 / 创建草稿（选样可回放） |
| datasets | POST | `/api/datasets/{datasetId}/freeze` | 冻结（写清单哈希，之后不可变） |
| datasets | POST | `/api/datasets/{datasetId}/export` | 导出 YOLO / COCO / LabelMe |
| runs | GET | `/api/runs` | 运行记录列表（导入/预标注/训练/评估/导出） |
| runs | GET / POST | `/api/runs/{runId}`、`/api/runs/{runId}/cancel`、`POST /api/runs` | **未实现**（M1 历史设计）：GET → 405，POST → 501，见 §2 |
| train | POST | `/api/train/runs` | 提交微调（`wait=false` → 202 异步轮询；`wait=true` → 200 同步） |
| train | GET | `/api/train/runs/{runId}` | 训练详情（状态/进度/指标/日志尾部/`in_process`） |
| train | POST | `/api/train/runs/{runId}/cancel` | 协作式取消（epoch 边界生效） |
| models | GET | `/api/models` | 模型注册表（`model_versions` 原始行） |
| models | GET | `/api/models/registry` | 模型摘要 + 门禁结论 + 主指标（M3） |
| models | GET | `/api/models/{modelId}` | 模型详情（摘要 + 完整评估结果） |
| models | POST | `/api/models/{modelId}/evaluate` | 评估（官方 val 口径 + 内部匹配口径） |
| models | POST | `/api/models/{modelId}/validate` | 评估 + 门禁（不通过 409） |
| models | POST | `/api/models/{modelId}/promote` | 提升 production（同任务旧生产模型自动归档） |
| models | POST | `/api/models/{modelId}/export` | 导出 ONNX 边缘包 + ONNX↔.pt 一致性验收 |
| stats | GET | `/api/stats/overview` | 统计概览（类别分布/任务进度/标注效率） |
| stats | GET | `/api/stats/export` | 统计明细导出（CSV/JSON，含经纬度） |

## 4. 错误语义

### 4.1 现状（与实现一致）

| 来源 | HTTP | 响应体 |
|---|---|---|
| 业务异常（`ConflictError`/`NotFoundError`/`ValueError`/`ConfigError`） | 400 / 404 / 409 | `{"code": "http_<status>", "message": "<中文说明>"}` |
| 请求体/查询参数/路径参数未通过 schema 校验（区间越界、类型错误、JSON 解析失败） | **400** | `{"code": "http_400", "message": "请求参数不合法：<loc>: <msg>；…", "details": {"errors": ["<msg>", …]}}`（`RequestValidationError` 处理器，**已不再是 FastAPI 默认的 422**） |
| 路径存在但方法不允许 | 405 | Starlette 默认 `{"detail": "Method Not Allowed"}`（不经过统一处理器） |
| 未实现端点（M4/M5 及兜底路由） | 501 | `{"code": "http_501", "message": "/api/<rest> 未实现：当前版本已实现 M1–M3 端点（导入/标注/复核/预标注/训练/门禁/导出），M4（边缘推理）与 M5（主动学习）规划中；已实现端点清单见 docs/06-api-spec.md"}` |
| 依赖缺失（未安装 ultralytics/torch）、权重不可读 | 501 | 预标注/训练提交：「未安装 ML 依赖（ultralytics/torch）：pip install -e '.[ml]'（详见 docs/08-deployment.md）」；导出/评估：「未安装 ultralytics/torch，无法导出 ONNX…」/「加载权重失败 …」 |
| 单文件超过上传上限 | 413 | `{"code": "http_413", "message": "<文件名> 超过单文件上限 NMB"}` |
| 未预期异常 | 500 | `{"code": "http_500", "message": "<异常文本>"}` |

要点：

- `code` **恒为** `http_<status>`，不是语义码；`details` 只在**请求校验失败（400）**时出现（`details.errors`）。
- 区间越界（`epochs`/`imgsz`/`batch`/`log_lines`/`opset`/`tolerance`/`parity_images`）与类型错误统一 **400**；
  业务校验失败（train/val 划分为空、`flipud` 非 0）也是 400，但**不带** `details`（按此区分两类 400）。
- 门禁未通过是 **409**，原因与 `delta` 拼在 `message` 里（见 §6.3），不会走 400/500。

### 4.2 语义错误码（规划，尚未实现）

| code | HTTP | 含义与处理 |
|---|---|---|
| `invalid_request` | 400 | 参数/坐标/格式非法（含 bbox 越界、类别不存在） |
| `unsupported_media` | 415 | 不支持的文件类型 |
| `path_forbidden` | 400 | `source_dir` 不在白名单内 |
| `not_found` | 404 | 资源不存在 |
| `conflict` | 409 | 状态不允许（如对非 draft 数据集重复冻结） |
| `dataset_name_taken` | 409 | 数据集名重复（需新建版本） |
| `dataset_not_frozen` | 409 | 用未冻结数据集训练 |
| `gate_failed` | 409 | 权重门禁未通过（`details.baseline` / `details.delta` 给出对比） |
| `prelabel_disabled` | 503 | 预标注被配置关闭或模型未加载（**当前实现**：ML 依赖缺失统一 501） |
| `model_not_ready` | 503 | 指定模型不存在/加载失败（**当前实现**：权重加载失败为 501） |
| `lease_conflict` | 409 | 任务被他人领取（租约未过期） |
| `quota_exceeded` | 429 | 批次体积/请求频率超限（**当前实现**：单文件超限为 413） |
| `internal` | 500 | 未预期错误（规划附 `details.trace_id`） |

## 5. 幂等与并发语义

- **导入**：以文件 `sha256` 为幂等键（`uq_images_sha256` 唯一索引）；重复文件记 `dup_sha`，感知哈希近似的记 `dup_phash`，均不重复建任务。
- **标注提交**：`PUT` 覆盖式；服务端按 `(task_id, annotation_id)` diff，只写增量并记 `audit_log`。重复提交同一份内容得到 `changed=0`（天然幂等），但**请求头 `Idempotency-Key` 被忽略**。
- **任务领取**：`POST /api/tasks/lease` 原子领取（租约到期可被重新领取，`attempts+1`）；租约默认 1800s。
- **长任务**：`runs.run_key` 为唯一索引；训练幂等键见 §6.2。预标注的 `run_key` 由 `(kind, target, 参数摘要)` 派生，同键复用既有 run。
- **取消**：协作式——当前 epoch/批次结束后停止；训练保留最近一次 checkpoint（`best.pt`/`last.pt` 可续训）。

## 6. 训练闭环与门禁（M3）

### 6.1 异步训练与轮询

```
POST /api/train/runs                      → 202（异步，默认）
  {"run": {..., "status": "running"}, "request": {...}, "splits": {"train":140,"val":30,"test":30},
   "cached": false, "run_id": 12, "message": "训练已在后台启动，请轮询 GET /api/train/runs/{run_id}"}

POST /api/train/runs  {..., "wait": true}  → 200（同步，训练已结束）
  {"run": {..., "status": "succeeded"}, "request": {...}, "splits": {...}, "model": {...,"status":"candidate"},
   "cached": false, "run_id": 12, "status": "succeeded"}      # 无 message

GET  /api/train/runs/12?log_lines=40      → 200（**扁平**结构，无 run 包装）
  {"id": 12, "kind": "train", "status": "running", ..., "metrics": {...}, "progress": {"epoch": 3, ...},
   "epochs_done": 3, "final": null, "weights_path": null, "work_dir": "...",
   "log_tail": ["..."], "model": null, "in_process": true}

POST /api/train/runs/12/cancel            → 200
  {"run_id": 12, "status": "running", "cancel_requested": true,
   "note": "将在当前 epoch 结束时停止；已生成的 best.pt/last.pt 仍可用于续训"}
```

- 默认 `wait=false`：训练跑在服务进程的**后台线程**里，`run` 的真实状态一律以 `runs` 表为准。
- **僵尸运行对账**：`status=running` 但已不在本进程（上次服务被杀/重启遗留）的运行，会被收尾为
  `failed`（`error=进程中断（服务重启或被杀），运行未完成；已允许按同一配置重新提交`，并记 `metrics.interrupted_at`）——
  服务**启动时**自动做一次（对账失败不影响启动），CLI 可随时用 `runs --reconcile` 手动补做，
  同参数重新提交时也会先收尾再重跑。因此 `in_process=false` 且 `status=running` 基本不会长期存在；
  真出现（如对账异常）时取消**必然 409**，需直接终止训练进程。
- `wait=true`：同步跑完再返回 **200**；响应多出 `status` 与 `model`，但没有 `message`。
- 训练日志落盘 `data/logs/runs/train-<run_id>-<slug>.log`；每个 epoch 把 `results.csv` 末行（列名归一化为
  `map50`/`map50_95`/`precision`/`recall`/各 `loss`/`lr`/`epoch`）写入 `runs.metrics_json.progress` 与 `epochs_done`，
  结束时写入 `final`/`weights_path`/`last_weights`/`work_dir`/`train_params`。
- **依赖缺失在请求期就失败**：`start_training()` 在**建 run 之前**预检 ML 依赖，缺 ultralytics/torch 时
  `POST /api/train/runs` 直接返回 **501**（`wait` 两种取值一致），**不会留下 run 记录**；
  只有「依赖正常、但训练过程本身出错」（数据异常、初始权重损坏、CUDA OOM…）才在**异步分支**表现为
  `202` 之后该 run 变为 `failed`（原因在 `error`，形如 `RuntimeError: …`）；同步分支则直接在 200 响应里
  给出 `status=failed`。注意与**导出**路径的区别：导出的权重加载失败是请求期 **501**（见 §6.4）。
- 前置条件：数据集必须 `frozen`（否则 409，见 ADR-0005）；`train`/`val` 划分非空（否则 400）；响应里的
  `splits` 是各划分的图片数。

### 6.2 幂等（run_key）

```
run_key = sha256("train" + 数据集名 + manifest_hash + arch + 初始权重路径 + 初始权重 sha256
                 + resume + 全部超参(imgsz/epochs/batch/device/…) + 增强参数)[:32]
```

| 既有运行状态 | 行为 |
|---|---|
| `succeeded` | 复用该运行：`cached=true`、返回既有 `run_id`，不重复占用 GPU |
| `running` 且确实在当前进程内 | **409**「同一配置的训练已在运行中（run #N）；如需中止请调用取消接口」 |
| `running` 但已不在本进程（僵尸） | 先收尾为 `failed`（`error=进程中断…`），再复用同一 run 行重跑 |
| `failed` / `canceled` | 复用同一 run 行重跑（写入 `restarted_at`） |

数据集内容变化 → 新 `manifest_hash` → 新 `run_key`（ADR-0005：冻结即不可变）。

### 6.3 模型状态机与门禁

```
训练成功 → candidate ──validate 通过──▶ validated ──promote──▶ production
                │                          │                      │
                └─ validate 未通过(409，状态退回 candidate)        └─ 同任务旧 production 自动 → archived
```

- `candidate → validated`：`POST /api/models/{id}/validate`（内部先 evaluate、再 gate；响应含 `gate` 与 `metrics`）。
- `validated → production`：`POST /api/models/{id}/promote`。**只有 validated 可提升**，且必须存在**通过的门禁记录**；
  同任务旧 production 自动归档为 `archived`（响应 `archived` 列出被归档的 id）。已是 production 时返回 `changed=false`。
- 门禁判据（阈值见 `configs/train.yaml` 的 `gate`，默认 `map50_tolerance=0.005`、`per_class_tolerance=0.02`、
  `min_map50=0.0`、`min_class_recall=null`，基线默认取现役 production）：
  - 整体 mAP50 相对基线下降超过 `map50_tolerance`；
  - 任一类别 mAP50 相对基线下降超过 `per_class_tolerance`（由 >0 掉到 0 单列为「类别塌陷」）；
  - 候选 mAP50 低于绝对下限 `min_map50`；任一类召回低于 `min_class_recall`（非 null 时）。
  - 无基线（首个模型）时只按绝对下限判定，并在 `notes` 里显式标注。
- **门禁未通过返回 409**，`message` 形如
  「门禁未通过：整体 mAP50 下降 0.3425 超过容差 0.0050（0.7234 → 0.3809）；类别塌陷：pothole mAP50 由 0.5100 掉到 0.0000；delta={...}」；
  完整结论（`passed`/`reasons`/`notes`/`delta`/`baseline`/`candidate`/`thresholds`/`checked_at`）同时写入
  `model_versions.gate_json`，可用 `GET /api/models/{id}` 的 `gate` 字段回看。

### 6.4 ONNX 导出与一致性验收

只有 `validated`/`production` 模型可导出（`model_versions.onnx_path` 仅在验收通过后更新）。包目录
`exports/<name>-<version>/`：

| 文件 | 内容 |
|---|---|
| `model.onnx` | opset 17（可调 11–20）、可选 dynamic batch / simplify / FP16 |
| `labels.txt` | 类别 code，行号 = 模型类别下标（顺序必须与冻结数据集一致，否则 409） |
| `preprocess.json` | letterbox（居中、pad 114、keep aspect）、`/255`、RGB、NCHW、后处理 `conf`/`iou`/`max_detections`/NMS 类型 |
| `manifest.json` | 溯源：模型 sha256/大小、权重路径与 sha256、数据集名与清单哈希、`train_run_id`/`export_run_id`、门禁结论、`parity` 摘要 |
| `parity.json` | 仅 `verify=true`：ONNX↔.pt 一致性验收结果 |
| `README.md` | 边缘端最少用法示例 |

一致性验收（`verify=true`，默认）判据：同一 letterbox 张量分别喂 torch 与 onnxruntime，比较原始输出
（`max_raw_delta`，仅报告），再用**同一套**解码 + NMS 比较逐框**归一化坐标**——
`max_bbox_delta ≤ tolerance`（默认 1e-3）**且**两侧独有框数为 0（`unmatched_pt == unmatched_onnx == 0`）。
不通过时：包仍落盘（便于排查）、`parity.passed=false`、`registered=false`，且**不更新**模型的 `onnx_path`。

响应 `package` 的键集**稳定**：无论 `verify` 取值都返回
`dir`/`model_path`/`labels`/`manifest_path`/`preprocess_path`/`parity_path`/`model_sha256`/`model_size_bytes`/`manifest`
（`verify=false` 时 `parity_path=null`、`manifest` 照常返回，只是不含 parity 摘要）。

导出 409 的两种情形（都发生在建 export run **之前**，不消耗导出算力、不留 run）：
状态非 validated/production；模型类别顺序与当前类别表不一致（需按新顺序重新冻结数据集训练）。
**依赖缺失 / 权重不可读 → 501**：未安装 ultralytics/torch（导出与一致性验收两处都预检）、或权重文件损坏
无法加载，都返回 501（CLI 对应退出码 3）。权重加载失败发生在 run 创建之后，因此会留下一条
`status=failed` 的 export 运行记录，便于排查。

### 6.5 仍存在的实现与契约差异（截至 M3，2026-09-19 逐条实测核对）

| # | 现象 | 影响 |
|---|---|---|
| 1 | 错误体 `code` 恒为 `http_<status>`（无 §4.2 的语义码）；405 由 Starlette 直接产出，体为 `{"detail": "Method Not Allowed"}` | 依赖语义错误码或统一错误体的客户端需要额外分支 |
| 2 | `POST /api/runs`、`GET /api/runs/{id}`、`POST /api/runs/{id}/cancel` 从未实现（POST→501 兜底、GET→405）；501 文案已中立化，但端点仍是 M1 期遗留 | M1 期文档遗留（见 §2）；训练请走 `/api/train/runs*` |
| 3 | `openapi.yaml` 对 **M1/M2 部分端点**仍是设计形态，未按实现回填：缺 `POST /api/ingest/upload`、`DELETE /api/annotations/{id}`、`GET|POST /api/annotations/{id}/mask`、`POST /api/tasks/{id}/adopt-candidates`、`GET /api/prelabel/metrics` 的条目；`prelabel`/`export` 等写的是 `202 + Run`，实现是 `200 + 业务负载` | 以本文档 §2/§3/§7 与源码为准；openapi 的 M1/M2 部分是**待清理的历史偏差**（M3 新增的 7 个路径已与实现一致） |
| 4 | 权重加载失败发生在 export run 创建**之后** → 会留下一条 `status=failed` 的 export 运行记录 | **有意保留**（审计价值）；对照：状态/类别顺序类 409 不留下任何 run |

### 6.6 已在源码侧修复（曾列入差异，2026-09-19 复测通过）

| 原现象 | 现状 |
|---|---|
| `wait=true` 同步分支也返回 202 | **已修复**：同步分支返回 **200**（202 只表示"已接受、稍后完成"）；异步分支仍为 202 |
| 导出响应的 `package` 键集随 `verify` 变化（`verify=false` 时缺 `parity_path`/`manifest`） | **已修复**：键集稳定——`verify=false` 时也返回 `parity_path`（`null`）与 `manifest`，调用方无需分支 |
| 幂等命中（`cached=true`）时 `message` 谎称「训练已在后台启动」 | **已修复**：命中时 `message` 为「命中幂等：复用既有运行，未启动新训练」 |
| 参数越界返回 422 + FastAPI 默认 `{"detail":[…]}` | **已修复**：新增 `RequestValidationError` 处理器 → **400**，体为 `{"code":"http_400","message":"请求参数不合法：<loc>: <msg>；…","details":{"errors":[…]}}`（路径参数类型错、JSON 解析失败同样 400） |
| 幂等命中（`cached=true`）时响应 `run_id=null` | **已修复**：命中时同样返回既有运行的 `run_id`，与后台分支一致 |
| `promote` 在「已是 production」时缺 `gate`/`weights_path` | **已修复**：两次调用形状一致（含 `gate`/`weights_path`），仅 `changed=false`、`archived=[]` 并多一个 `message` |
| 类别顺序不一致的 409 在 ONNX 导出**之后**才判定 | **已修复**：`_class_records` 前置到建 export run 之前，报错不再消耗导出算力、不留下 run |
| 权重损坏/不可读时 `export` 返回 500 | **已修复**：`_run_export` 把加载失败转为 `DetectorUnavailable` → **501**（CLI 退出码 3）；`evaluate` 同样是 501 |
| 缺 ultralytics 时 `POST /api/models/{id}/export` 返回 409 | **已修复**：`export_onnx`/`verify_onnx_parity` 改抛 `DetectorUnavailable` → **501**，CLI 退出码 3；不创建 export run |
| `POST /api/train/runs` 缺依赖时先 202、再由后台线程把 run 置 failed | **已修复**：`start_training()` 在建 run 前预检依赖 → **501**，不留 run 记录；训练期真实错误仍是 202 后 `failed` |
| 兜底 501 文案误报「属于 M4/M5」 | **已修复**：改为中立文案「… 未实现：当前版本已实现 M1–M3 端点…；M4（边缘推理）与 M5（主动学习）规划中」 |
| 服务重启后遗留的 `running` 僵尸运行会一直显示"训练中" | **已修复**：服务启动时自动对账（`reconcile_stale_runs`）收尾为 `failed`；CLI `runs --reconcile` 可手动补做；同参数重新提交时也会先收尾再重跑 |

## 7. CLI 契约（与 REST 同源，供巡检与自动化使用）

全局选项：`rdinspect [--config <path>] [--data-dir <dir>] [--json] <command> …`

```
# ── M1 采集 / 标注 / 复核 / 数据集 ──
rdinspect serve    [--host 127.0.0.1] [--port 8787] [--reload]
rdinspect import   --input <dir|file> [--kind photo|video|aerial|external] [--fps 2]
                   [--max-frames N] [--tile | --no-tile] [--limit N] [--note ...]
rdinspect tasks    [--status pending] [--limit 20] [--lease N] [--assignee cli]
rdinspect dataset  list | show --name ds-xxx
rdinspect dataset  create --name ds-xxx [--review-status approved|annotated|any]
                   [--class <code>]... [--batch <id>]...
                   [--split-train 0.7 --split-val 0.15 --split-test 0.15 --seed 42]
rdinspect dataset  freeze --name ds-xxx
rdinspect export   --name ds-xxx [--formats yolo,coco,labelme] [--no-copy-images]
rdinspect classes  list [--all] | add --code c --zh 名称 --en Name [--color #e6194b]
                   [--order 100] [--is-crack] | disable|enable --code c
rdinspect stats    [--export csv|json]
rdinspect thumbs                  # 重建缩略图
rdinspect check                   # 自检：文档/DDL/OpenAPI/样例 + 数据库连通性

# ── M2 模型辅助标注 ──
rdinspect prelabel --limit 50 [--status pending] [--task <id>]... [--model <路径|名>]
                   [--conf 0.25] [--iou 0.5] [--device auto|cpu|cuda] [--sam] [--sam-limit 2]
rdinspect prelabel-metrics [--json]
rdinspect models   [--task detection] [--status candidate|validated|production|archived]

# ── M3 训练闭环 ──
rdinspect train    --dataset ds-xxx [--name yolo11s-road] [--version 2026.09.19-a]
                   [--arch yolo11s.pt] [--resume-from <weights.pt|last.pt>]
                   [--epochs 100] [--imgsz 640] [--batch 16] [--device auto|cpu|cuda] [--no-register]
                   # 前台同步训练，逐 epoch 向 stderr 打印 mAP50 / mAP50-95；
                   # 日志落 data/logs/runs/train-<run_id>-<slug>.log；--no-register 只训练不登记模型
rdinspect runs     [--kind train|evaluate|export|prelabel|ingest] [--status <状态>] [--limit 20]
                   [--show <id> [--tail 30]] [--cancel <id>] [--reconcile]
                   # --show 打印 run 详情与日志尾部；--cancel 请求中止训练（epoch 边界生效）
                   # --reconcile 把上次进程中断遗留的 running 运行收尾为 failed（幂等，输出 {"reconciled":[…]}
                   #   服务启动时也会自动做一次同样的对账）
rdinspect model    list
rdinspect model    show --id N
rdinspect model    evaluate --id N [--split val]
rdinspect model    validate --id N [--split val]        # 评估 + 门禁；不通过 → 退出码 5
rdinspect model    promote  --id N                      # 仅 validated 可提升；失败 → 退出码 5
rdinspect model    export   --id N [--opset 17] [--imgsz 640] [--tolerance 1e-3]
                   [--dynamic-batch] [--half] [--no-verify]
                   # 一致性验收未通过或前置冲突 → 退出码 5，registered=false

# ── M4 边缘推理（当前未实现） ──
rdinspect infer    --model exports/<pkg>/ --input <dir|video|rtsp://...> --out ./out
                   [--imgsz 640] [--conf 0.25] [--iou 0.5] [--tile 1024 --overlap 0.2]
                   [--device cpu|gpu] [--jsonl out.jsonl]
```

退出码：`0` 成功；`2` 参数错误（argparse 拒绝或子命令显式校验，如缺 `--name`/`--id`）；
`3` 依赖缺失（未安装 ML 依赖：`prelabel`、`train`，以及 `model evaluate/validate/promote/export`）；
`4` 运行失败（训练未成功、数据集未冻结、模型不存在、未捕获的 `ConfigError`/`ValueError` 等）；
`5` **门禁未通过**（`model validate` 门禁失败、`model promote` 被拒、`model export` 一致性验收未通过或状态冲突）。

## 8. 边缘推理输出格式（M4 设计，当前未实现）

```json
{"image":"frames/000123.jpg","ts":"2026-09-12T02:13:05Z","gps":{"lat":31.2304,"lon":121.4737},
 "model":{"name":"yolo11s-road","version":"2026.09.12-a","imgsz":640},
 "detections":[{"class":"transverse_crack","conf":0.71,"bbox":[0.12,0.34,0.58,0.41]},
               {"class":"pothole","conf":0.88,"bbox":[0.62,0.55,0.79,0.72]}],
 "tiles":2,"elapsed_ms":41}
```

配套 `results.csv`（便于表格软件）：`image,ts,lat,lon,class,conf,x1,y1,x2,y2`。
导出包 `manifest.json` 记录 `rdinspect_version`、类别顺序（`labels`/`classes`）与阈值/切片参数；
边缘端加载时 `load_export_package` 会校验 `manifest.json`、模型文件与 `preprocess.json`/`labels.txt` 是否存在
（**注意**：当前没有 `schema_version` 字段，也没有版本不匹配时拒绝启动的逻辑——这是 M4 的待补项）。

## 9. 契约演进策略

- 新增字段：向后兼容，客户端忽略未知字段。
- 类别顺序变化：**不改变端点**，但会使既有导出包与新数据集不兼容 → 通过数据集新版本 + 重新导出解决（ADR-0005）。
- 破坏性变更（如标注坐标改像素制）：升 `info.version` 主版本，并在 `db/migrations` 提供数据迁移脚本。
