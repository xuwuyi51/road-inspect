"""SQLite 连接、模式初始化与迁移。

约定（见 docs/03-data-model.md、ADR-0004）：
  * 单写者：服务进程内串行化写操作，WAL 允许并发读；
  * 文件写入用「临时文件 + rename」保证原子性（见 storage/files.py）；
  * 模式版本记录在 schema_meta，迁移脚本位于 db/migrations/*.sql（按文件名顺序执行）。
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCHEMA_PATH = PROJECT_ROOT / "db" / "schema.sql"
MIGRATIONS_DIR = PROJECT_ROOT / "db" / "migrations"


def connect(db_path: str | Path) -> sqlite3.Connection:
    """打开连接并设置 WAL / 外键 / busy timeout。"""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, isolation_level=None, timeout=5.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.execute("PRAGMA busy_timeout = 5000")
    return conn


@contextmanager
def transaction(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """显式事务；异常回滚。sqlite3 在 isolation_level=None 时不自动开事务。"""
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield conn
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise


def schema_version(conn: sqlite3.Connection) -> str:
    row = conn.execute("SELECT value FROM schema_meta WHERE key='schema_version'").fetchone()
    return row["value"] if row else "0"


def apply_schema(conn: sqlite3.Connection, schema_path: Path | None = None) -> None:
    """执行 db/schema.sql（幂等：全部 CREATE ... IF NOT EXISTS）。"""
    path = schema_path or SCHEMA_PATH
    conn.executescript(path.read_text(encoding="utf-8"))


def apply_migrations(conn: sqlite3.Connection, migrations_dir: Path | None = None) -> list[str]:
    """按文件名顺序执行尚未应用的迁移（记录在 schema_meta.applied_migrations）。"""
    directory = migrations_dir or MIGRATIONS_DIR
    if not directory.exists():
        return []
    applied_row = conn.execute(
        "SELECT value FROM schema_meta WHERE key='applied_migrations'"
    ).fetchone()
    applied = set((applied_row["value"] if applied_row else "").split(",")) - {""}
    done: list[str] = []
    for sql_file in sorted(directory.glob("*.sql")):
        if sql_file.name in applied:
            continue
        conn.executescript(sql_file.read_text(encoding="utf-8"))
        applied.add(sql_file.name)
        done.append(sql_file.name)
    if done:
        conn.execute(
            "INSERT INTO schema_meta(key, value) VALUES('applied_migrations', ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (",".join(sorted(applied)),),
        )
    return done


def init_db(db_path: str | Path) -> sqlite3.Connection:
    """连接 + 建表 + 迁移，返回可用连接。"""
    conn = connect(db_path)
    apply_schema(conn)
    apply_migrations(conn)
    return conn
