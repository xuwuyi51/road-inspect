# 06 · 接口规范（API & CLI）

> 机器可读契约：[`openapi.yaml`](./openapi.yaml)（OpenAPI 3.1，已校验：25 路径 / 29 操作 / 17 schema / 29 引用，无悬空引用）。

## 1. 基本约定

| 项 | 约定 |
|---|---|
| Base URL | `http://127.0.0.1:8787`（默认仅本机；需局域网时显式配置监听并提示防火墙） |
| 认证 | 首版无账号；仅监听本机。若放开局域网必须配合反代鉴权（文档中标注为部署前置条件） |
| 内容类型 | JSON（`application/json`）；影像字节流为 `image/jpeg` / `image/png` |
| 时间 | UTC ISO8601 文本 `YYYY-MM-DDTHH:MM:SSZ`（前端按本地时区显示） |
| 坐标 | 归一化 `[0,1]`，明确相对"所属影像"（切片则为该 tile） |
| 分页 | `limit`（1–500，默认 50）+ `cursor`（不透明字符串）；响应 `{items, next_cursor}` |
| 幂等 | 写操作支持 `Idempotency-Key` 头；长任务用 `runs.run_key` |
| 长任务 | 统一返回 `run` 资源（`202`），客户端轮询 `GET /api/runs/{id}`（含 `log_tail`），可 `POST /api/runs/{id}/cancel` |
| 错误 | 统一 `{code, message, details}`；HTTP 状态语义化（400/404/409/429/503） |
| 上传限制 | 单文件 ≤ 50MB；单批次 ≤ 2GB；`source_dir` 必须落在 `configs/default.yaml: allowed_roots` 白名单内 |
| 版本 | 契约版本随 `info.version`；不兼容变更升主版本并在 `manifest.json` 中标注 |

## 2. 端点总览

| Tag | 方法 | 路径 | 用途 |
|---|---|---|---|
| health | GET | `/api/health` | 服务与依赖状态（现役模型、GPU、schema 版本） |
| classes | GET / POST | `/api/classes` | 类别注册表读取 / 新增（扩展点） |
| ingest | POST / GET | `/api/ingest/batches` | 创建导入批次（上传或登记目录）/ 批次列表 |
| ingest | GET | `/api/ingest/batches/{batchId}` | 批次详情、导入结果统计与失败清单 |
| images | GET | `/api/images` | 影像查询（批次/来源/时间过滤） |
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
| runs | GET / POST | `/api/runs` | 运行记录列表 / 启动训练或评估 |
| runs | GET | `/api/runs/{runId}` | 运行详情（指标 + 日志尾部） |
| runs | POST | `/api/runs/{runId}/cancel` | 取消运行 |
| models | GET | `/api/models` | 模型注册表 |
| models | POST | `/api/models/{modelId}/promote` | 晋级（门禁不通过返回 409） |
| models | POST | `/api/models/{modelId}/export` | 导出边缘包（ONNX + labels + preprocess + manifest） |
| stats | GET | `/api/stats/overview` | 统计概览（类别分布/任务进度/标注效率/模型指标） |
| stats | GET | `/api/stats/export` | 统计明细导出（CSV/JSON，含经纬度与路段号） |

## 3. 错误码表

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
| `prelabel_disabled` | 503 | 预标注被配置关闭或模型未加载 |
| `model_not_ready` | 503 | 指定模型不存在/加载失败 |
| `lease_conflict` | 409 | 任务被他人领取（租约未过期） |
| `quota_exceeded` | 429 | 批次体积/请求频率超限 |
| `internal` | 500 | 未预期错误（附带 `details.trace_id`，日志可查） |

## 4. 幂等与并发语义

- **导入**：以文件 `sha256` 为幂等键；同批重复导入返回 `dup_sha` 计数，不重复建任务。
- **标注提交**：`PUT` 覆盖式 + `Idempotency-Key`；服务端按 `(task_id, annotation_id)` diff，写入 `audit_log`；同一 key 重复提交返回首次结果。
- **任务领取**：`POST /api/tasks/lease` 使用 `UPDATE ... WHERE status=? AND (lease_until IS NULL OR lease_until < now)` 原子领取；租约默认 1800s，超时自动回到可领取状态（`attempts+1`）。
- **预标注/训练**：`run_key = sha256(kind|target|params)`；同 key 返回既有 run（避免重复烧 GPU）。
- **取消**：`cancel` 为协作式——当前 epoch/批次结束后停止；训练保留最近一次 checkpoint。

## 5. CLI 契约（与 REST 同源，供巡检与自动化使用）

```
rdinspect serve    [--host 127.0.0.1] [--port 8787] [--data-dir ./data]
rdinspect import   --kind photo|video|aerial|external --input <dir|file> [--fps 2] [--tile 1024 --overlap 0.2] [--track track.csv]
rdinspect prelabel --status pending --limit 200 [--conf 0.25] [--sam] [--zeroshot]
rdinspect tasks    --status prelabeled --limit 50 [--out tasks.json]
rdinspect dataset  create --name ds-xxx --filter filter.json
rdinspect dataset  freeze --name ds-xxx
rdinspect dataset  export --name ds-xxx --format yolo|coco|labelme [--out dir]
rdinspect train    --dataset ds-xxx --config configs/train.yaml [--resume <weights.pt>]
rdinspect evaluate --model <name:version> --dataset ds-xxx [--out metrics.json]
rdinspect export   --model <name:version> [--opset 17] [--out exports/]
rdinspect infer    --model exports/<pkg>/ --input <dir|video|rtsp://...> --out ./out
                   [--imgsz 640] [--conf 0.25] [--iou 0.5] [--tile 1024 --overlap 0.2] [--device cpu|gpu] [--jsonl out.jsonl]
rdinspect stats    [--csv out.csv]
```

退出码：`0` 成功；`2` 参数错误；`3` 依赖缺失（模型/GPU/ffmpeg）；`4` 运行失败（详情写日志与 `runs`）；`5` 门禁未通过。

## 6. 边缘推理输出格式（`results.jsonl`，每行一条）

```json
{"image":"frames/000123.jpg","ts":"2026-09-12T02:13:05Z","gps":{"lat":31.2304,"lon":121.4737},
 "model":{"name":"yolo11s-road","version":"2026.09.12-a","imgsz":640},
 "detections":[{"class":"transverse_crack","conf":0.71,"bbox":[0.12,0.34,0.58,0.41]},
               {"class":"pothole","conf":0.88,"bbox":[0.62,0.55,0.79,0.72]}],
 "tiles":2,"elapsed_ms":41}
```

配套 `results.csv`（便于表格软件）：`image,ts,lat,lon,class,conf,x1,y1,x2,y2`。
`manifest.json`（导出包内）声明 `schema_version`、类别顺序、阈值与切片参数；边缘端版本不匹配时拒绝启动并提示重新导出。

## 7. 契约演进策略

- 新增字段：向后兼容，客户端忽略未知字段。
- 类别顺序变化：**不改变端点**，但会使既有导出包与新数据集不兼容 → 通过数据集新版本 + 重新导出解决。
- 破坏性变更（如标注坐标改像素制）：升 `info.version` 主版本，并在 `db/migrations` 提供数据迁移脚本。
