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

### 1.4 模型辅助标注（M2）的依赖与目录

```bash
# CPU 版（约 1.6GB，适合预标注/单机验证；本机实测 200 张 640px 预标注 15.7s）
.venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install ultralytics opencv-python-headless -i https://pypi.tuna.tsinghua.edu.cn/simple
# GPU 版（本机有 RTX 5060 Ti 时更快）：把上面第一条换成
#   .venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

- 权重放 `data/weights/`（裸文件名如 `yolo11n.pt` 会被解析到该目录，不会下载到进程 CWD）。
- ultralytics/matplotlib 的配置目录被程序收进 `data/ultralytics`、`data/mpl`（`Config.export_runtime_env()` 设置
  `YOLO_CONFIG_DIR`/`MPLCONFIGDIR`）；这样在受限文件系统（只读 home、容器）里也能导入 ultralytics。
- 未安装 ML 依赖时：预标注端点返回 **501** 并给出安装命令，人工标注与数据集流程完全不受影响。

### 1.5 训练闭环（M3）的依赖、目录与容器注意事项

```bash
# 训练/评估/导出的增量依赖（在 §1.4 的 torch+ultralytics 之上）
.venv/bin/pip install onnx onnxruntime onnxslim -i https://pypi.tuna.tsinghua.edu.cn/simple
```

| 项 | 说明 |
|---|---|
| 训练工作目录 | `data/runs/run-<run_id>-<slug>/`（weights、results.csv、args.yaml）；**失败重跑会把旧目录改名 `.prev-<时刻>`**，避免陈旧的 results.csv 污染新指标 |
| 运行日志 | `data/logs/runs/train-<run_id>-<slug>.log`（ultralytics 日志逐行落盘；异常连堆栈一起落盘） |
| 评估产物 | `data/runs/eval-<run_id>-<dataset>-<split>/`：`report-<split>.md`、`metrics-<split>.json`、`failures/`（叠加图 + failures.json） |
| 导出包 | `data/exports/<name>-<version>/`：`model.onnx`、`labels.txt`、`preprocess.json`、`manifest.json`、`parity.json`、`README.md`（该目录整体拷到边缘即可用） |
| 训练数据 | 冻结数据集的 YOLO 导出用**硬链接**（同分区零拷贝），不额外占盘；`data/runs/_data/*.yaml` 只是绝对路径的 data.yaml 副本 |
| GPU 训练 | 装了 CUDA 版 torch 后 `device: auto` 会自动用 GPU；`amp: auto` 只在 CUDA 下开混合精度。CPU 上 120 张 @320、30 epoch 实测 66 秒（YOLO11n） |

⚠️ **容器/沙箱里的两个坑（本机实测）**：

1. **`/dev/shm` 必须可写**。ultralytics 扫描标签时会建 `multiprocessing` 线程池，而线程池初始化
   就要创建 POSIX 信号量（`/dev/shm`）。只读 `/dev/shm` 时训练会以
   `PermissionError: [Errno 13] Permission denied` 失败，且报错点离真实原因很远。
   程序侧已做兼容：`src/rdinspect/train/compat.py` 探测到信号量不可用时，把线程池换成串行扫描
   （结果一致、只是慢一点；正常环境零影响）。Docker 里仍建议 `--shm-size=1g`（或 `--ipc=host`），
   也可用 `RDINSPECT_SERIAL_SCAN=1` 强制串行。
2. **把 `HOME`（或 `XDG_CACHE_HOME`）指向可写目录**。onnxruntime 启动时会持久化遥测设备 ID：
   HOME 不可写时它会在**进程 CWD** 落一个 `:memory:.ses` 文件（本机实测；已加进 `.gitignore`，
   容器里建议 `-e HOME=/tmp` 并挂载可写卷）。
3. **无网环境要预热两个缓存**：ultralytics 首次画图会下载 `Arial.ttf`，且首次导入时会在
   `YOLO_CONFIG_DIR` 写 `settings.json`。离线部署前请在联网机器上先跑一次 `rdinspect train`（或在
   镜像里预置 `data/ultralytics/`），否则启动阶段会卡在字体下载上。

- 训练端点同样是"缺依赖就降级"：未装 ML 依赖时 `POST /api/train/runs` 返回 **501** 并附安装命令，
  标注/复核/数据集流程不受影响。
- 门禁不通过返回 **409**（`model_versions.status` 保持 `candidate`），这是设计上的"拒绝发布"，
  不是服务错误——排障时看 `gate_json.reasons` 与 `delta`。

## 2. 形态二：边缘（车载/巡检，离线）

| 项 | 方案 |
|---|---|
| 运行环境 | Linux x86_64（车载主机/NUC）或 Windows；Python 3.11 + onnxruntime（CPU 版约 40MB） |
| 依赖 | 仅 `onnxruntime` + `pillow`/`opencv-headless` + `numpy`；**无数据库、无浏览器、无 ffmpeg 依赖**（视频输入时需 ffmpeg 或内置解码） |
| 安装 | 拷贝导出包 + `rdinspect` wheel（离线 `pip install ./wheels/*.whl`） |
| 运行 | `rdinspect infer --package ./model/<name>-<version>-<imgsz> --input /mnt/sd --out ./out --resume`（M4 已实现，详见 [12-edge-inference](./12-edge-inference.md)） |
| 依赖 | 只需 `numpy + pillow + onnxruntime + pyyaml + road-inspect`（**不需要** fastapi/uvicorn/torch）；视频/流额外需要 OpenCV 或 ffmpeg；实测 59MB 离线包（wheels 48.7MB + 模型 10.5MB） |
| 启动前校验 | `validate_package()` 校验 schema 版本、模型 sha256、labels 一致性、输出通道数 = 4+nc、**模型内嵌 names/imgsz**；任一不匹配 → 退出码 **6**，不产出结果文件 |
| 性能实测（本机 CPU） | imgsz 640 · 4 线程：p50 22.9ms / **43.7 FPS**（达标 ≥15）；imgsz 320 · 4 线程：p50 6.96ms / **141 FPS**；RSS 峰值 194–438MB |
| 断点续跑 | `results.jsonl` + `.infer-state.json`（原子写）双保险；状态文件丢失也能从 JSONL 重建，不重复处理 |
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
| 运行视图 | `GET /api/runs`（列表）与 `GET /api/train/runs/{id}`（状态 + epoch 进度 + 日志尾部）；失败 run 保留 `error`（含堆栈，落在 `data/logs/runs/*.log`） |
| 健康检查 | `/api/health` 返回 schema 版本、现役模型、GPU 可用性；systemd `ExecStartPost` 可做自检 |
| 常见故障 | ① 模型未加载 → 检查 `model_versions.status='production'` 是否存在；② 预标注超时 → 降低 tile 尺寸或关 SAM；③ 训练 OOM → 降 batch/imgsz；④ 磁盘满 → 见 §3 |
| 保留策略 | `thumbs/` 可重建（可随时删除）；`raw/` 与冻结数据集不可删；`exports/` 保留最近 3 个版本 |

## 6. 与现有环境共存

- 端口：DSH GUI 3080 / searxng 8888 / road-inspect 8787，互不冲突。
- GPU 竞争：训练与预标注会占用显存；与本地大模型/H3 视频生成同时使用时需排队（文档建议：训练放在空闲时段，或通过 `runs` 队列串行化）。
- Python 环境：项目自建 venv；复用系统 ffmpeg 与已有 CUDA 轮子，不修改 DiffSynth venv 与系统 python 包。
- 可选集成（非本期）：以 DSH 插件形式暴露 `road_inspect_*` 工具（导入/预标注/统计），便于在 DSH GUI 中用自然语言驱动；契约已由 `openapi.yaml` 固定。
