-- M5（主动学习与类别扩展）：runs.kind 增加 'active'，并新增主动学习队列表
--
-- 注意：SQLite 不能修改 CHECK 约束，只能重建表并搬数据。重建期间必须关掉外键，
-- 否则 DROP TABLE runs 会把 model_versions.run_id 按 ON DELETE SET NULL 置空（丢溯源信息）。

PRAGMA foreign_keys = OFF;

DROP TABLE IF EXISTS runs_m5;
CREATE TABLE runs_m5 (
  id            INTEGER PRIMARY KEY AUTOINCREMENT,
  kind          TEXT NOT NULL CHECK (kind IN ('ingest','prelabel','train','evaluate','export','active')),
  status        TEXT NOT NULL DEFAULT 'queued' CHECK (status IN ('queued','running','succeeded','failed','canceled')),
  run_key       TEXT UNIQUE,
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
INSERT INTO runs_m5 SELECT * FROM runs;
DROP TABLE runs;
ALTER TABLE runs_m5 RENAME TO runs;

CREATE INDEX IF NOT EXISTS idx_runs_kind_status ON runs(kind, status);
CREATE INDEX IF NOT EXISTS idx_runs_key ON runs(run_key);

PRAGMA foreign_keys = ON;

-- 主动学习队列：记录"哪些任务被选中、为什么、得分多少"，供标注台与复盘使用
CREATE TABLE IF NOT EXISTS active_queue (
  id              INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id          INTEGER REFERENCES runs(id) ON DELETE SET NULL,
  task_id         INTEGER NOT NULL REFERENCES tasks(id) ON DELETE CASCADE,
  image_id        INTEGER NOT NULL REFERENCES images(id) ON DELETE CASCADE,
  strategy        TEXT NOT NULL DEFAULT 'hybrid',
  score           REAL NOT NULL DEFAULT 0,
  priority        INTEGER NOT NULL DEFAULT 100,
  reason          TEXT NOT NULL DEFAULT 'score',      -- score | uncertainty | error | diversity | random
  components_json TEXT NOT NULL DEFAULT '{}',         -- {uncertainty, error, diversity}
  detail_json     TEXT NOT NULL DEFAULT '{}',         -- 证据来源、命中类别等
  created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
);

CREATE INDEX IF NOT EXISTS idx_active_queue_run ON active_queue(run_id);
CREATE INDEX IF NOT EXISTS idx_active_queue_task ON active_queue(task_id);
CREATE INDEX IF NOT EXISTS idx_active_queue_image ON active_queue(image_id);
CREATE INDEX IF NOT EXISTS idx_active_queue_score ON active_queue(score DESC);
