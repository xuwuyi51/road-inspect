-- road-inspect schema v1 (SQLite)
-- 用法：python -c "import sqlite3;sqlite3.connect('data/app.db').executescript(open('db/schema.sql').read())"
-- 或 sqlite3 data/app.db < db/schema.sql
--
-- 设计要点
--  * images 只增不改（软删用 duplicate_of / 状态字段），保证数据集可回放
--  * annotations 以"有效行"表达（deleted_at 软删），每次提交写 audit_log
--  * dataset_versions 冻结后不可变：class_order 与 manifest_hash 一并固化
--  * 所有时间使用 UTC ISO8601 文本（'YYYY-MM-DDTHH:MM:SSZ'）

PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;

-- ─────────────────────────── 元信息 ───────────────────────────
CREATE TABLE IF NOT EXISTS schema_meta (
  key    TEXT PRIMARY KEY,
  value  TEXT NOT NULL
);
INSERT OR IGNORE INTO schema_meta(key, value) VALUES ('schema_version', '1');

-- ─────────────────────────── 类别注册表（可扩展） ───────────────────────────
CREATE TABLE IF NOT EXISTS classes (
  code        TEXT PRIMARY KEY,              -- 如 transverse_crack
  name_zh     TEXT NOT NULL,                 -- 横向裂缝
  name_en     TEXT NOT NULL,                 -- Transverse Crack
  color       TEXT NOT NULL DEFAULT '#e6194b',
  parent_code TEXT REFERENCES classes(code) ON DELETE SET NULL,
  is_crack    INTEGER NOT NULL DEFAULT 0 CHECK (is_crack IN (0,1)),
  order_index INTEGER NOT NULL DEFAULT 100,  -- YOLO 类别索引用（冻结时写入数据集）
  active      INTEGER NOT NULL DEFAULT 1 CHECK (active IN (0,1)),
  note        TEXT,
  created_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
);
CREATE INDEX IF NOT EXISTS idx_classes_order ON classes(order_index);

-- ─────────────────────────── 导入批次 ───────────────────────────
CREATE TABLE IF NOT EXISTS batches (
  id         INTEGER PRIMARY KEY AUTOINCREMENT,
  kind       TEXT NOT NULL CHECK (kind IN ('photo','video','aerial','external')),
  source     TEXT,                            -- 目录/设备/文件名备注
  note       TEXT,
  stats_json TEXT NOT NULL DEFAULT '{}',      -- {added,dup_sha,dup_phash,skipped,error}
  created_by TEXT,
  created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
);

CREATE TABLE IF NOT EXISTS ingest_items (
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  batch_id    INTEGER NOT NULL REFERENCES batches(id) ON DELETE CASCADE,
  source_path TEXT NOT NULL,
  outcome     TEXT NOT NULL CHECK (outcome IN ('added','dup_sha','dup_phash','skipped_type','skipped_size','error')),
  reason      TEXT,
  image_id    INTEGER,
  created_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
);
CREATE INDEX IF NOT EXISTS idx_ingest_items_batch ON ingest_items(batch_id);

-- ─────────────────────────── 影像 ───────────────────────────
CREATE TABLE IF NOT EXISTS images (
  id              INTEGER PRIMARY KEY AUTOINCREMENT,
  batch_id        INTEGER REFERENCES batches(id) ON DELETE SET NULL,
  path            TEXT NOT NULL,                       -- 相对 data/ 的路径
  sha256          TEXT NOT NULL,                       -- 内容哈希（幂等键）
  phash           TEXT,                                -- 感知哈希（近似去重）
  width           INTEGER NOT NULL,
  height          INTEGER NOT NULL,
  bytes           INTEGER NOT NULL,
  source_kind     TEXT NOT NULL CHECK (source_kind IN ('photo','video_frame','aerial','aerial_tile','external')),
  parent_image_id INTEGER REFERENCES images(id) ON DELETE SET NULL,  -- 帧/切片指回原图
  tile_json       TEXT,                                -- {"x":..,"y":..,"w":..,"h":..,"overlap":..}
  captured_at     TEXT,                                -- EXIF/帧时间戳
  gps_lat         REAL,
  gps_lon         REAL,
  gps_source      TEXT CHECK (gps_source IN ('exif','track','manual','none')),
  device          TEXT,
  duplicate_of    INTEGER REFERENCES images(id) ON DELETE SET NULL,
  quality_json    TEXT,                                -- 亮度/模糊度等质量指标
  created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
);
CREATE UNIQUE INDEX IF NOT EXISTS uq_images_sha256 ON images(sha256);
CREATE INDEX IF NOT EXISTS idx_images_phash ON images(phash);
CREATE INDEX IF NOT EXISTS idx_images_batch ON images(batch_id);
CREATE INDEX IF NOT EXISTS idx_images_parent ON images(parent_image_id);
CREATE INDEX IF NOT EXISTS idx_images_captured ON images(captured_at);

-- ─────────────────────────── 任务（状态机） ───────────────────────────
CREATE TABLE IF NOT EXISTS tasks (
  id             INTEGER PRIMARY KEY AUTOINCREMENT,
  image_id       INTEGER NOT NULL REFERENCES images(id) ON DELETE CASCADE,
  status         TEXT NOT NULL DEFAULT 'pending'
                 CHECK (status IN ('pending','prelabeling','prelabeled','annotating','annotated','reviewing','approved','rejected','skipped')),
  priority       INTEGER NOT NULL DEFAULT 100,          -- 数值越小越优先
  assignee       TEXT,
  lease_until    TEXT,                                  -- 领取租约（并发防重）
  prelabel_state TEXT NOT NULL DEFAULT 'none'
                 CHECK (prelabel_state IN ('none','queued','done','failed')),
  attempts       INTEGER NOT NULL DEFAULT 0,
  created_at     TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
  updated_at     TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
  submitted_at   TEXT,
  reviewed_at    TEXT
);
CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status, priority);
CREATE INDEX IF NOT EXISTS idx_tasks_image ON tasks(image_id);

-- ─────────────────────────── 标注 ───────────────────────────
CREATE TABLE IF NOT EXISTS annotations (
  id               INTEGER PRIMARY KEY AUTOINCREMENT,
  task_id          INTEGER NOT NULL REFERENCES tasks(id) ON DELETE CASCADE,
  image_id         INTEGER NOT NULL REFERENCES images(id) ON DELETE CASCADE,
  class_code       TEXT NOT NULL REFERENCES classes(code),
  kind             TEXT NOT NULL DEFAULT 'bbox' CHECK (kind IN ('bbox','polygon','mask')),
  -- bbox 使用归一化坐标 [0,1]，便于跨分辨率复用；像素坐标由服务端按需换算
  bbox_x1          REAL,
  bbox_y1          REAL,
  bbox_x2          REAL,
  bbox_y2          REAL,
  geometry_json    TEXT,                       -- polygon/关键点等扩展几何
  mask_path        TEXT,                       -- RLE 或 PNG（裂缝掩膜）
  difficult        INTEGER NOT NULL DEFAULT 0 CHECK (difficult IN (0,1)),
  score            REAL,                       -- 模型置信度（人工标注为 NULL）
  source           TEXT NOT NULL DEFAULT 'human'
                   CHECK (source IN ('human','model','model_edited','external')),
  model_version_id INTEGER REFERENCES model_versions(id) ON DELETE SET NULL,
  deleted_at       TEXT,                       -- 软删，保留审计
  created_at       TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
  updated_at       TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
  CHECK (kind <> 'bbox' OR (bbox_x1 IS NOT NULL AND bbox_y1 IS NOT NULL AND bbox_x2 IS NOT NULL AND bbox_y2 IS NOT NULL))
);
CREATE INDEX IF NOT EXISTS idx_ann_task ON annotations(task_id);
CREATE INDEX IF NOT EXISTS idx_ann_image ON annotations(image_id);
CREATE INDEX IF NOT EXISTS idx_ann_class ON annotations(class_code);
CREATE INDEX IF NOT EXISTS idx_ann_source ON annotations(source);

-- ─────────────────────────── 复核 ───────────────────────────
CREATE TABLE IF NOT EXISTS reviews (
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  task_id     INTEGER NOT NULL REFERENCES tasks(id) ON DELETE CASCADE,
  reviewer    TEXT,
  decision    TEXT NOT NULL CHECK (decision IN ('approve','reject')),
  reason_code TEXT CHECK (reason_code IN ('missing','wrong_class','loose_box','tight_box','mask_bad','other')),
  note        TEXT,
  created_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
);
CREATE INDEX IF NOT EXISTS idx_reviews_task ON reviews(task_id);

-- ─────────────────────────── 数据集版本 ───────────────────────────
CREATE TABLE IF NOT EXISTS dataset_versions (
  id               INTEGER PRIMARY KEY AUTOINCREMENT,
  name             TEXT NOT NULL UNIQUE,        -- 如 ds-2026w38
  status           TEXT NOT NULL DEFAULT 'draft' CHECK (status IN ('draft','frozen','archived')),
  filter_json      TEXT NOT NULL DEFAULT '{}',  -- 选样条件（可回放）
  class_order_json TEXT,                        -- 冻结时固化：["longitudinal_crack",...]
  split_json       TEXT,                        -- {"train":0.8,"val":0.1,"test":0.1,"seed":42,"group_by":"road_segment"}
  stats_json       TEXT,                        -- 各类别/各划分数量
  manifest_hash    TEXT,                        -- 清单哈希（images+annotations+class_order）
  root_path        TEXT,                        -- data/datasets/<name>/
  frozen_at        TEXT,
  created_at       TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
);

CREATE TABLE IF NOT EXISTS dataset_items (
  dataset_id INTEGER NOT NULL REFERENCES dataset_versions(id) ON DELETE CASCADE,
  image_id   INTEGER NOT NULL REFERENCES images(id) ON DELETE CASCADE,
  split      TEXT NOT NULL CHECK (split IN ('train','val','test')),
  PRIMARY KEY (dataset_id, image_id)
);
CREATE INDEX IF NOT EXISTS idx_dataset_items_split ON dataset_items(dataset_id, split);

-- ─────────────────────────── 运行记录（训练/预标注/导出/导入） ───────────────────────────
CREATE TABLE IF NOT EXISTS runs (
  id            INTEGER PRIMARY KEY AUTOINCREMENT,
  kind          TEXT NOT NULL CHECK (kind IN ('ingest','prelabel','train','evaluate','export')),
  status        TEXT NOT NULL DEFAULT 'queued' CHECK (status IN ('queued','running','succeeded','failed','canceled')),
  run_key       TEXT UNIQUE,                    -- 幂等键
  config_json   TEXT NOT NULL DEFAULT '{}',
  dataset_id    INTEGER REFERENCES dataset_versions(id) ON DELETE SET NULL,
  model_id      INTEGER,
  log_path      TEXT,
  metrics_json  TEXT,
  error         TEXT,
  started_at    TEXT,
  finished_at   TEXT,
  created_at    TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
);
CREATE INDEX IF NOT EXISTS idx_runs_status ON runs(status, kind);

-- ─────────────────────────── 模型注册表 ───────────────────────────
CREATE TABLE IF NOT EXISTS model_versions (
  id           INTEGER PRIMARY KEY AUTOINCREMENT,
  name         TEXT NOT NULL,                   -- 如 yolo11s-road
  version      TEXT NOT NULL,                   -- 如 2026.09.12-a
  task         TEXT NOT NULL DEFAULT 'detection' CHECK (task IN ('detection','segmentation','zeroshot')),
  status       TEXT NOT NULL DEFAULT 'candidate' CHECK (status IN ('candidate','validated','production','archived')),
  run_id       INTEGER REFERENCES runs(id) ON DELETE SET NULL,
  dataset_id   INTEGER REFERENCES dataset_versions(id) ON DELETE SET NULL,
  weights_path TEXT,                            -- .pt
  onnx_path    TEXT,                            -- 导出产物
  labels_json  TEXT,                            -- {"names":[...],"nc":4,"thresholds":{...}}
  metrics_json TEXT,                            -- {"mAP50":...,"per_class":{...}}
  gate_json    TEXT,                            -- {"passed":true,"baseline":"...","delta":{...}}
  sha256       TEXT,
  created_at   TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
  UNIQUE (name, version)
);
CREATE INDEX IF NOT EXISTS idx_models_status ON model_versions(status, task);

-- ─────────────────────────── 审计 ───────────────────────────
CREATE TABLE IF NOT EXISTS audit_log (
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  entity      TEXT NOT NULL,                    -- annotation/task/dataset/model/class
  entity_id   TEXT NOT NULL,
  action      TEXT NOT NULL,                    -- create/update/delete/freeze/promote/export
  actor       TEXT,
  before_json TEXT,
  after_json  TEXT,
  at          TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
);
CREATE INDEX IF NOT EXISTS idx_audit_entity ON audit_log(entity, entity_id);

-- ─────────────────────────── 触发器：updated_at ───────────────────────────
CREATE TRIGGER IF NOT EXISTS trg_tasks_updated
AFTER UPDATE ON tasks FOR EACH ROW
BEGIN
  UPDATE tasks SET updated_at = strftime('%Y-%m-%dT%H:%M:%SZ','now') WHERE id = OLD.id;
END;

CREATE TRIGGER IF NOT EXISTS trg_annotations_updated
AFTER UPDATE ON annotations FOR EACH ROW
BEGIN
  UPDATE annotations SET updated_at = strftime('%Y-%m-%dT%H:%M:%SZ','now') WHERE id = OLD.id;
END;

-- ─────────────────────────── 视图：常用查询 ───────────────────────────
CREATE VIEW IF NOT EXISTS v_task_queue AS
SELECT t.id AS task_id, t.status, t.priority, i.id AS image_id, i.path, i.source_kind,
       i.captured_at, i.gps_lat, i.gps_lon
FROM tasks t JOIN images i ON i.id = t.image_id
WHERE t.status IN ('pending','prelabeled','annotating')
ORDER BY t.priority ASC, t.id ASC;

CREATE VIEW IF NOT EXISTS v_class_counts AS
SELECT a.class_code, c.name_zh, COUNT(*) AS n
FROM annotations a JOIN classes c ON c.code = a.class_code
WHERE a.deleted_at IS NULL
GROUP BY a.class_code, c.name_zh
ORDER BY n DESC;

-- ─────────────────────────── 初始类别（可扩展） ───────────────────────────
INSERT OR IGNORE INTO classes(code, name_zh, name_en, color, is_crack, order_index) VALUES
  ('longitudinal_crack', '纵向裂缝', 'Longitudinal Crack', '#1f77b4', 1, 0),
  ('transverse_crack',   '横向裂缝', 'Transverse Crack',   '#d62728', 1, 1),
  ('pothole',            '坑洞',     'Pothole',            '#2ca02c', 0, 2),
  ('garbage',            '垃圾',     'Garbage',            '#9467bd', 0, 3),
  ('alligator_crack',    '网裂（预留）','Alligator Crack', '#ff7f0e', 1, 4);
