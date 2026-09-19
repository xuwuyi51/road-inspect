# 12 · 边缘离线推理操作手册（M4）

> 面向"把模型放到车上/巡检设备上、没有网也要跑"的人。命令可直接复制；每步都有产物路径与失败处置。
> 上游：导出包由 [11-training-guide](./11-training-guide.md) §6 产出；性能目标与依赖见 [08-deployment](./08-deployment.md) §2。

## 1. 全链路一图

```
工作站                                    边缘设备（可完全无网）
────────                                  ──────────────────────
rdinspect model export  ──►  导出包  ──►  rdinspect infer --package <dir> --input <目录/视频/流>
  （含 parity 一致性验收）      + 离线安装包        ├─ results.jsonl（每图一条，含 GPS/时间）
                              （wheels+脚本）      ├─ results.csv（每检测一条，便于表格/GIS）
                                                  └─ .infer-state.json（断点续跑）
```

**推理链路只依赖 `onnxruntime + numpy + pillow`**（视频/流另需 OpenCV 或 ffmpeg），不装 torch/ultralytics、
不连数据库（可用 `python -c "import sys, rdinspect.edge.infer; assert 'torch' not in sys.modules"` 自证）。

## 2. 准备导出包（工作站）

```bash
.venv/bin/rdinspect model export --id 1 --imgsz 320      # 只有 validated/production 模型可导出
# → data/exports/<name>-<version>-<imgsz>/
#    model.onnx  labels.txt  preprocess.json  manifest.json  parity.json  README.md
```

导出时的 **ONNX↔.pt 一致性验收**必须通过（逐框归一化坐标误差 ≤ 1e-3，实测 1e-06）；
不通过时 `registered=false`，模型不会登记 `onnx_path`（见 training-guide §6）。

## 3. 打包与目标机安装（可无网）

```bash
# 工作站：生成离线包（wheels + 导出包 + 脚本）
python3 scripts/build_edge_bundle.py --package data/exports/<name>-<version>-320 \
    --out /tmp/edge-bundle --python-version 3.12
#   --platform manylinux_2_28_aarch64 可交叉为 ARM64 车载设备准备
#   --skip-wheels 只生成骨架（无网的工作站上也能先备好目录与脚本）
tar -czf edge-bundle.tar.gz -C /tmp/edge-bundle .

# 目标机（无网）：
tar -xzf edge-bundle.tar.gz && cd edge-bundle && ./install.sh   # venv + 离线安装 + 冒烟自检
./verify.sh                                                     # 只校验导出包完整性
```

`install.sh` 走 `pip install --no-index --find-links wheels`，不访问网络；自检会真的跑一张图。

## 4. 运行推理

```bash
# 目录（递归，默认跳过已处理项）
.venv/bin/rdinspect infer --package ./model/<name>-<version>-320 --input /mnt/sd/photos --out /mnt/sd/out

# 视频抽帧（2 fps；需 OpenCV 或 ffmpeg）
.venv/bin/rdinspect infer --package <dir> --input patrol.mp4 --fps 2 --out ./out

# 实时视频流（RTSP/RTMP/HTTP；Ctrl-C 停止）
.venv/bin/rdinspect infer --package <dir> --input rtsp://192.168.1.10/stream --out ./out --limit 500

# 细裂缝/航拍大图：开启切片（ADR-0006）
.venv/bin/rdinspect infer --package <dir> --input /mnt/aerial --out ./out --tile 1024

# 命中留证据图 + 只处理前 50 张
.venv/bin/rdinspect infer --package <dir> --input /mnt/sd --out ./out --snapshots --limit 50
```

| 参数 | 说明 |
|---|---|
| `--edge-config` | 边缘配置（默认 `configs/edge.yaml`，可缺省）；**`inference.imgsz` 默认 `null` = 用包内尺寸** |
| `--imgsz` | 显式覆盖推理尺寸；模型输入是静态尺寸时会直接报错，动态尺寸才允许 |
| `--threads` | onnxruntime 算子内线程数（默认 4；边缘 CPU 预算有限时调小） |
| `--resume` / `--no-resume` | 断点续跑开关（默认按配置，`edge.yaml` 里为开） |
| `--expect-labels` | 期望的类别顺序（逗号分隔）；与包不一致则拒绝启动 |
| `--no-hash-check` | 跳过模型 sha256 校验（不推荐；只在超大模型+可信介质时用） |

退出码：`0` 成功 · `2` 参数/输入源错误 · `4` 运行失败 · **`6` 导出包校验失败（拒绝启动）**。

## 5. 输出契约（docs/06 §8）

`results.jsonl` 每行一条影像：

```json
{"image":"sim_00000.jpg","source":"/mnt/sd/photos/sim_00000.jpg","image_sha256":"…",
 "index":0,"frame_index":null,"ts":"2026-09-10T08:00:00","gps":{"lat":31.219167,"lon":121.434167},
 "model":{"name":"yolo11n-road","version":"2026.09.19-r1","imgsz":640,"schema_version":1,"labels":[…]},
 "detections":[{"class":"transverse_crack","conf":0.71,"bbox":[0.12,0.34,0.58,0.41]}],
 "tiles":1,"elapsed_ms":22.877,"width":640,"height":360}
```

`results.csv` 每行一条检测（`image,ts,lat,lon,class,conf,x1,y1,x2,y2`）；**没有检测的图也会占一行**
（class/conf/bbox 为空），便于按图统计"查了多少张、命中多少张"。坐标一律归一化 `[0,1]`。

## 6. 断点续跑（三层保险）

1. **JSONL 即真相**：恢复时先扫已写出的 JSONL 里的 `image_sha256`（视频帧用路径+帧号）；
2. **状态文件** `.infer-state.json`：记录同一集合与累计计数，供快速恢复；
3. **原子写**：状态文件先写临时文件再 `os.replace`，被 `kill -9` 也不会留下半截 JSON。

因此"跑一半断电 → 重新执行同一条命令"不会重复处理，也不会因为状态文件丢失而从头再来。
`--no-resume` 才会重新处理全部输入。

## 7. 性能（实测，本机 Intel Core Ultra 5 230F，CPU）

| 包内 imgsz | 线程 | p50 | p95 | FPS | RSS 峰值 |
|---|---|---|---|---|---|
| 320 | 4 | 6.96 ms | 7.13 ms | **141** | 194 MB |
| 320 | 1 | 14.9 ms | 15.2 ms | 67 | 193 MB |
| 640 | 4 | 22.9 ms | 23.2 ms | **43.7** | 438 MB |

对照 [08-deployment §2.1](./08-deployment.md) 的目标"YOLO11n ONNX @640 CPU 4 线程 ≥ 15 FPS"：**达标（43.7 FPS）**。
复现：`rdinspect infer --package <dir> --input <目录> --out ./bench --benchmark 60 --threads 4`（结果写 `bench/benchmark.json`）。

> 注意：这里的 FPS 是**整条链路**（解码 + letterbox + 推理 + NMS + 写盘）的端到端吞吐，
> 不是纯推理时延；换更小的包内尺寸（320）能到 141 FPS，但细裂缝召回会下降——按路况选尺寸。

## 8. 启动前校验：拒绝"结果不可信"的包

`infer` 在读取任何一张图之前先校验导出包，任一项不匹配 → **退出码 6，不产出任何结果文件**：

| 校验 | 拦下的问题 |
|---|---|
| `schema_version` 存在且 ≤ 运行端支持版本 | 包比 runtime 新（字段语义可能不同） |
| `model.onnx` 的 sha256 == `manifest.model_sha256` | 模型文件被替换/损坏/不完整 |
| `labels.txt` == `manifest.labels` == `manifest.nc` | 文本清单被改 |
| 模型输出形状里存在等于 `4+nc` 的维度 | 类别数量漂移（最危险：类别会整体错位） |
| **模型内嵌 `names` == `labels.txt`** | 文本清单被**整体重排**（连 manifest 一起改）也拦得住 |
| 模型内嵌 `imgsz` == `preprocess.input_size`；静态输入尺寸与之一致 | 预处理契约与模型不匹配 |
| `--expect-labels` 与包内类别顺序一致 | 与工作站的类别表不一致（需重新冻结数据集并重新导出） |

实测四类篡改（schema 升高 / 类别整体重排 / 模型追加字节 / 期望类别不符）全部在启动阶段被拒。

## 9. 排障速查

| 症状 | 原因 | 处理 |
|---|---|---|
| 退出码 6，提示"模型哈希不匹配" | 包被改动或拷贝不完整 | 重新打包传输；不要手改 `model.onnx` |
| 退出码 6，提示"内嵌类别名与 labels.txt 不一致" | 有人改了 `labels.txt`/`manifest.labels` | 用工作站重新导出，别手工改包内文本 |
| 提示"模型输入尺寸固定为 [320,320]，与请求的 imgsz=640 不一致" | 静态尺寸模型 + `--imgsz` 覆盖 | 去掉 `--imgsz`（用包内尺寸）或导出 640 的包 |
| 视频报"需要 OpenCV 或 ffmpeg" | 边缘最小依赖里视频解码是可选的 | `pip install opencv-python-headless` 或装 ffmpeg |
| `pip install --no-index` 报找不到包 | wheels 不齐（交叉下载时平台/版本不匹配） | 重新用 `--platform/--python-version` 生成离线包 |
| CWD 出现 `:memory:.ses` | HOME 不可写，onnxruntime 把遥测 ID 落到当前目录 | 无害（已 gitignore）；把 `HOME`/`XDG_CACHE_HOME` 指到可写目录 |
| 断点续跑后统计数没涨 | 输入没变（全部命中跳过） | 看输出的"跳过 N"；要重跑加 `--no-resume` |

## 10. 自动化验收

```bash
.venv/bin/python scripts/e2e_m4.py --package <导出包目录>          # 复用已有包，约 1 分钟
.venv/bin/python scripts/e2e_m4.py --photos 200                   # 现场训练+导出+全流程验收
```

覆盖 8 项断言：无网代理下 200 张推理 / 推理链路不导入 torch / JSONL+CSV 契约 / 与工作站逐框一致 /
断点续跑不重复 / 四类篡改拒绝启动 / 性能达标 / 离线安装包骨架，
末行打印 `M4 验收：通过 ✅（8/8 项）`，完整报告写 `<work>/e2e_m4_report.json`。
