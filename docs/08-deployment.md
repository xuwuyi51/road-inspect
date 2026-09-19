# 08 · 部署与运维（Deployment）

## 1. 形态一：工作站（首版主形态）

### 1.1 依赖与安装

| 组件 | 版本/来源 | 说明 |
|---|---|---|
| Python | 3.11（项目自建 venv，`uv venv`） | 不污染 DiffSynth/系统环境 |
| CUDA 轮子 | torch 2.13+cu130（复用本机已有） | 训练与 GPU 预标注 |
| 推理运行时 | onnxruntime 1.28（系统已有）或 `onnxruntime-gpu` | 边缘/CPU 预标注 |
| 媒体 | ffmpeg（系统已有） | 抽帧/转码 |
| 关键库 | fastapi、pydantic、ultralytics、opencv-python-headless、pillow、imagehash、pyyaml | 见 `pyproject.toml`（实现阶段补） |
| 前端 | 预构建 `web/dist`（无需 Node 运行时） | 服务端单端口托管 |

```bash
# 目标形态（实现阶段）
uv venv .venv && source .venv/bin/activate
uv pip install -e .
rdinspect serve --host 127.0.0.1 --port 8787 --data-dir ./data
# 浏览器打开 http://127.0.0.1:8787
```

### 1.2 目录与权限

```
/opt/road-inspect/        # 代码与 venv（只读运行）
/var/lib/road-inspect/    # 数据根 data/（app.db + 文件）、备份目标
/etc/road-inspect/        # configs/*.yaml
```
- 服务运行用户仅需 `data/` 读写权限；`allowed_roots`（可登记的上传源目录）显式白名单，默认仅 `data/inbox/`。
- 建议 `systemd --user` 服务（与 DSH 同机共存，端口 8787 不与 3080 冲突）：

```ini
[Unit]
Description=road-inspect workstation service
[Service]
WorkingDirectory=/opt/road-inspect
ExecStart=/opt/road-inspect/.venv/bin/rdinspect serve --host 127.0.0.1 --port 8787 --data-dir /var/lib/road-inspect/data
Restart=on-failure
[Install]
WantedBy=default.target
```

### 1.3 资源预算（实测环境 RTX 5060 Ti 15.5GB / RAM 15GB）

| 场景 | CPU | 内存 | 显存 | 磁盘 |
|---|---|---|---|---|
| 空闲服务 | < 5% | < 300MB | 0 | — |
| 人工标注（浏览器） | 一个核 | 浏览器侧 | 0 | 缩略图缓存 |
| 预标注（YOLO11s 640） | 1–2 核 | < 1.5GB | < 4GB | — |
| 预标注 + SAM 掩膜 | 2 核 | < 2.5GB | < 6GB | 掩膜 PNG |
| 微调 YOLO11s 640 batch16 | 4–6 核 | < 8GB | < 14GB | 权重 + 日志 |
| 边缘推理（CPU 4 线程） | 4 核 | < 500MB | 0 | — |

## 2. 形态二：边缘（车载/巡检，离线）

| 项 | 方案 |
|---|---|
| 运行环境 | Linux x86_64（车载主机/NUC）或 Windows；Python 3.11 + onnxruntime（CPU 版约 40MB） |
| 依赖 | 仅 `onnxruntime` + `pillow`/`opencv-headless` + `numpy`；**无数据库、无浏览器、无 ffmpeg 依赖**（视频输入时需 ffmpeg 或内置解码） |
| 安装 | 拷贝导出包 + `rdinspect` wheel（离线 `pip install ./wheels/*.whl`） |
| 运行 | `rdinspect infer --model ./exports/yolo11s-road-onnx --input /mnt/sd --out ./out --device cpu` |
| 加速 | 可选 TensorRT（NVIDIA 设备）、OpenVINO（Intel）；INT8 需在冻结验证集上复测掉点 |
| 存储 | 结果 JSONL/CSV < 1MB/千帧；可选的现场截图按命中类别保存（默认关闭） |
| 断点续跑 | `--resume` 记录已处理文件哈希；重复运行跳过已完成项 |
| 无网升级 | 导出包内含 `manifest.json`（版本/类别/阈值/哈希），不匹配时拒绝启动并提示重新导出 |

### 2.1 性能目标（边缘）

| 模型 | 设备 | 输入 | 目标 |
|---|---|---|---|
| YOLO11n ONNX FP16 | CPU 4 线程 | 640 | ≥ 15 FPS |
| YOLO11n ONNX FP16 | GPU | 640 | ≥ 100 FPS |
| YOLO11s ONNX FP16 | GPU | 640 | ≥ 60 FPS |
| SAHI 切片（1024 tile） | GPU | 4K 航拍 | ≤ 1.5s/张 |

## 3. 备份、恢复与容量

| 项 | 策略 |
|---|---|
| 备份内容 | `app.db`（含 WAL 检查点）+ `raw/`、`masks/`、`datasets/`（可硬链接，实际只备份一次原图）+ `configs/` |
| 频率 | 每日增量（rsync/硬链接快照）；冻结数据集与生产模型导出后立即备份 |
| 恢复演练 | 每次里程碑执行一次：在临时目录恢复 → `rdinspect serve` → 校验 `manifest_hash` 与模型哈希一致 |
| 容量 | 首版小样本约 5–6GB（见 03 文档容量估算）；扩到 5 万张约需 60–80GB |
| 磁盘不足的处理 | ① 清理 `thumbs/` 与 `exports/` 旧版本；② 数据集导出用硬链接（同分区零拷贝）；③ 外接盘/NAS：把 `data/` 迁出后 `ln -s /mnt/nas/road-inspect-data ./data`（SQLite 需落在同一分区，或改为 `data/db/` 独立目录） |

## 4. 安全与合规

- **仅本机监听**是默认值；放开局域网必须：绑定内网地址 + 前置反代鉴权 + 防火墙仅放行办公网段（与 DSH 局域网访问相同的注意事项）。
- 上传：类型白名单、大小限制、路径穿越防护（拒绝 `..`、符号链接指向白名单外）、解压炸弹防护（不自动解压上传的压缩包）。
- 数据隐私：巡查影像可能包含车牌/人脸，默认不外传；如需云端训练必须显式导出并脱敏（文档标注为后续功能）。
- 许可：首版依赖 ultralytics（AGPL-3.0）与可选 X-AnyLabeling（GPLv3）——**仅内部使用**；若将来闭源分发，按 ADR-0003 切换到 RTMDet/YOLOX + 自建标注台，标注数据格式无需变更。
- 数据来源合规：RDD2022 / Unified Road Defect Dataset（HF）/ TACO 均需按其授权引用来源；文档列出引用要求，导出数据集时附带 `SOURCES.md`。

## 5. 可观测性与排障

| 项 | 做法 |
|---|---|
| 日志 | `data/logs/app.log`（按天轮转）；训练日志 `data/logs/runs/<run_id>.log` |
| 运行视图 | `/api/runs` 提供状态与日志尾部；失败 run 保留 `error` 与 trace_id |
| 健康检查 | `/api/health` 返回 schema 版本、现役模型、GPU 可用性；systemd `ExecStartPost` 可做自检 |
| 常见故障 | ① 模型未加载 → 检查 `model_versions.status='production'` 是否存在；② 预标注超时 → 降低 tile 尺寸或关 SAM；③ 训练 OOM → 降 batch/imgsz；④ 磁盘满 → 见 §3 |
| 保留策略 | `thumbs/` 可重建（可随时删除）；`raw/` 与冻结数据集不可删；`exports/` 保留最近 3 个版本 |

## 6. 与现有环境共存

- 端口：DSH GUI 3080 / searxng 8888 / road-inspect 8787，互不冲突。
- GPU 竞争：训练与预标注会占用显存；与本地大模型/H3 视频生成同时使用时需排队（文档建议：训练放在空闲时段，或通过 `runs` 队列串行化）。
- Python 环境：项目自建 venv；复用系统 ffmpeg 与已有 CUDA 轮子，不修改 DiffSynth venv 与系统 python 包。
- 可选集成（非本期）：以 DSH 插件形式暴露 `road_inspect_*` 工具（导入/预标注/统计），便于在 DSH GUI 中用自然语言驱动；契约已由 `openapi.yaml` 固定。
