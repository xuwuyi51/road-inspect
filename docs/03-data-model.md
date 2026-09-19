# 03 · 数据模型（Data Model）

> DDL 已交付并通过自检：`db/schema.sql` 在本机执行后可建 **13 张表 / 18 个索引 / 2 个视图 / 2 个触发器**，内置 5 个类别（含预留的网裂）。

## 1. 实体关系总览

```
batches ──1:N──▶ ingest_items
   │
   └─1:N──▶ images ──1:N──▶ tasks ──1:N──▶ annotations ──N:1──▶ classes
                │            │                │
                │            └─1:N──▶ reviews  └─N:1──▶ model_versions
                │
                └─N:M──▶ dataset_items ──N:1──▶ dataset_versions
                                                     │
                                            runs ◀───┘
                                             │
                                             └──▶ model_versions

audit_log：横切所有实体的写操作留痕（entity/entity_id/action/before/after）
```

## 2. 表说明（关键字段与约束）

| 表 | 作用 | 关键约束 |
|---|---|---|
| `schema_meta` | 模式版本，用于迁移判定 | `schema_version=1` |
| `classes` | **类别注册表**（扩展点） | `order_index` 决定 YOLO 索引；冻结数据集时固化顺序 |
| `batches` / `ingest_items` | 导入批次与逐项导入结果 | `outcome ∈ {added,dup_sha,dup_phash,skipped_type,skipped_size,error}`，可统计与回滚 |
| `images` | 影像唯一真相（只增不改） | `sha256` 唯一；`parent_image_id` 指向视频原帧/航拍原图；`tile_json` 记录切片位置 |
| `tasks` | 处理单元与状态机 | `lease_until` 领取租约；`prelabel_state ∈ {none,queued,done,failed}` |
| `annotations` | 标注（bbox/polygon/mask） | `bbox_*` 为**归一化**坐标；bbox 行必须四个坐标齐全（CHECK）；`deleted_at` 软删；`source` 区分人/机 |
| `reviews` | 复核记录 | `decision ∈ {approve,reject}` + `reason_code` 六类 |
| `dataset_versions` | 数据集版本 | `status ∈ {draft,frozen,archived}`；冻结后写 `class_order_json` 与 `manifest_hash` |
| `dataset_items` | 数据集成员与划分 | 主键 `(dataset_id,image_id)`；`split ∈ {train,val,test}` |
| `runs` | 任务运行记录（导入/预标注/训练/评估/导出） | `run_key` 唯一 → 幂等 |
| `model_versions` | 模型注册表 | `status ∈ {candidate,validated,production,archived}`；`gate_json` 存门禁结论 |
| `audit_log` | 审计 | 标注增删改、数据集冻结、模型晋级全部留痕 |

视图：`v_task_queue`（待办队列，含优先级）、`v_class_counts`（各类别标注数）。

## 3. 文件布局（与数据库的对应关系）

```
data/                              # 数据根（可指向外接盘/NAS，用符号链接接入）
├── app.db                         # SQLite（WAL: app.db-wal / app.db-shm）
├── raw/<YYYYMMDD>/<sha256>.jpg     # 原始影像（按内容哈希命名，天然去重）
├── frames/<video_id>/<ts>.jpg      # 视频抽帧（ts = 毫秒时间戳）
├── tiles/<image_id>/<x>_<y>.jpg    # 航拍切片（坐标写入 images.tile_json）
├── masks/<image_id>/<ann_id>.png   # 裂缝掩膜（PNG 单通道；大掩膜可改 RLE 存 DB）
├── thumbs/<image_id>.webp          # 标注台缩略图（长边 512）
├── datasets/<version>/             # 冻结数据集导出（images/ labels/ data.yaml manifest.sha256）
└── exports/<model>-<version>-onnx/ # 边缘导出包（model.onnx + labels.yaml + preprocess.json + manifest.json）
```

规则：
- 文件名一律用**内容哈希或稳定 ID**，不使用用户原始文件名（避免重名/注入/编码问题）。
- 原图只写一次；重命名/移动不改变数据库中的 `sha256`。
- 所有写入先写 `.tmp` 再 `rename`（同分区原子替换）。

## 4. 坐标与几何约定

| 场景 | 约定 |
|---|---|
| 数据库 bbox | 归一化 `x1,y1,x2,y2 ∈ [0,1]`，相对**所属影像**（切片则为该 tile） |
| 切片 ↔ 原图 | 由 `images.parent_image_id` + `tile_json` 换算：`x_orig = tile.x + x_tile * tile.w` |
| 导出 YOLO | `class cx cy w h`（归一化中心点+宽高，保留 6 位小数） |
| 导出 COCO | `bbox=[x,y,w,h]` 像素坐标 + `category_id`（需映射 `class_order`） |
| 掩膜 | PNG 单通道（0/255），命名 `masks/<image_id>/<ann_id>.png`，与同 task 的 bbox 可共存 |

## 5. 标注格式映射（三格式互通）

| 概念 | YOLO txt | COCO json | LabelMe / X-AnyLabeling json |
|---|---|---|---|
| 影像 | 一行一个 `.txt`，与图片同名 | `images[]: {id,file_name,width,height}` | 每个图片一个 json，含 `imagePath/imageWidth/imageHeight` |
| 类别 | `classes.txt` 顺序 = `class_order` | `categories[]: {id,name}` | `shapes[].label`（字符串名） |
| 目标 | `class cx cy w h` | `annotations[]: {bbox:[x,y,w,h],category_id,image_id}` | `shapes[]: {label,shape_type:'rectangle',points:[[x1,y1],[x2,y2]]}` |
| 来源标记 | 无（丢失） | `attributes.source`（本项目扩展字段） | `flags: {source: 'model'}`（本项目扩展字段） |
| 掩膜 | YOLO-seg：`class x1 y1 x2 y2 ...`（多边形点） | `segmentation: [[x,y,...]]` | `shape_type:'polygon'` |
| 归属 | `data.yaml`: `path/train/val/names` | 单文件 | 逐文件 |

**互转要求**：类别以**名称**为唯一键（数字索引由 `class_order` 决定，顺序变化即视为新数据集版本）；四类往返转换后框坐标误差必须为 0（保留 6 位小数）。样例见 `docs/format-samples/`。

## 6. 去重与幂等

| 层级 | 方法 | 处理 |
|---|---|---|
| 完全重复 | 文件 `sha256` | `images.sha256` 唯一索引；命中则 `ingest_items.outcome='dup_sha'`，不建新任务 |
| 近似重复（连拍/相邻帧） | `phash`（64bit）汉明距离 | 距离 ≤ 阈值（默认 6）标记 `duplicate_of`，默认**仍入库**但优先级降级（可在标注台查看，避免漏检） |
| 任务重复提交 | `runs.run_key` / 标注提交幂等 | 相同 key 直接返回既有结果 |
| 切片重复 | `(parent_image_id, x, y, w, h)` 唯一 | 防止重复切片入库 |

## 7. 数据集冻结算法（可复现的核心）

```
输入：filter_json（类别/批次/时间/复核状态/路段）
 1. 选样：按 filter 查询 images ⋈ annotations（source 不限，deleted_at IS NULL）
 2. 分组划分：按 road_segment（或 GPS 网格）分组，避免同一路段跨 train/test 泄漏
 3. 固化类别顺序：class_order = classes.order_index 排序结果（写入 class_order_json）
 4. 导出：写 datasets/<name>/{images/{train,val,test}, labels/{train,val,test}, data.yaml}
 5. 计算 manifest_hash = sha256(排序后的 (image.sha256, annotations 归一化序列化, class_order))
 6. 落库 dataset_versions.status='frozen', frozen_at, stats_json
 7. 写 audit_log(action='freeze')
```
约定：**训练只允许使用已冻结数据集**；同一 `name` 不可重复冻结（要改就新建版本）。

## 8. 迁移与兼容策略

- `schema_meta.schema_version` 单调递增；迁移脚本命名 `db/migrations/0001_xxx.sql`，只做增量（`ALTER TABLE ADD COLUMN`）不做破坏性变更。
- 类别删除一律用 `active=0`（保留历史标注可解释性）；`order_index` 变更必须新建数据集版本。
- 导出包（边缘端）带 `manifest.json`（含 schema_version / classes / thresholds），边缘端发现不兼容时拒绝启动并提示升级。

## 9. 容量估算（小样本首版）

| 项 | 估算 |
|---|---|
| 图片 | 5,000 张 × 平均 800KB ≈ 4GB（含缩略图 ~0.2GB） |
| 数据库 | 标注 5,000 图 × 平均 3 框 ≈ 15,000 行 → < 20MB |
| 数据集导出 | 与原始图同量级（可用硬链接节省空间） |
| 模型与导出 | YOLO11s 权重 19MB + ONNX 40MB |
| 合计 | **约 5–6GB**，符合当前 27GB 可用空间的约束 |
