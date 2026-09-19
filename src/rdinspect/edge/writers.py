"""边缘输出：`results.jsonl` / `results.csv` / 断点续跑状态（M4）。

契约见 docs/06-api-spec.md §8（JSONL 每行一条影像记录，CSV 每行一条检测）。

断点续跑的三层保险（都实现，互为兜底）：

1. **JSONL 即为真相**：恢复时先扫已存在的 JSONL 里的 `image_sha256`/`source`，得到已完成集合；
2. **状态文件**（`.infer-state.json`）记录同样的键 + 计数，便于无 JSONL 时快速恢复；
3. **原子写**：状态文件写临时文件再 `os.replace`，被 kill 时不会留下半截 JSON。

因此 `--resume` 既不重复处理，也不会因为"状态文件丢了"而重跑全部。
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

CSV_HEADER = ["image", "ts", "lat", "lon", "class", "conf", "x1", "y1", "x2", "y2"]

#: JSONL 每行固定字段（新增字段向后兼容，客户端应忽略未知字段）
JSONL_FIELDS = ("image", "source", "image_sha256", "index", "frame_index", "ts", "gps",
                "model", "detections", "tiles", "elapsed_ms", "width", "height")


def record_for(*, image: str, source: str, image_sha256: str, index: int, frame_index: int | None,
               ts: str | None, gps: dict[str, float] | None, model: dict[str, Any],
               detections: Sequence[dict[str, Any]], tiles: int, elapsed_ms: float,
               width: int, height: int) -> dict[str, Any]:
    """构造一条 JSONL 记录（字段与 docs/06 §8 一致，另加溯源字段）。"""
    return {
        "image": image,
        "source": source,
        "image_sha256": image_sha256,
        "index": int(index),
        "frame_index": frame_index,
        "ts": ts,
        "gps": gps,
        "model": model,
        "detections": [{"class": item["class"], "conf": round(float(item["conf"]), 6),
                        "bbox": [round(float(value), 6) for value in item["bbox"]]}
                       for item in detections],
        "tiles": int(tiles),
        "elapsed_ms": round(float(elapsed_ms), 3),
        "width": int(width),
        "height": int(height),
    }


class JsonlWriter:
    """追加写 JSONL（每行 flush，断电最多丢最后一行）。"""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a", encoding="utf-8")

    def write(self, record: dict[str, Any]) -> None:
        self._handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._handle.flush()

    def close(self) -> None:
        self._handle.close()

    def __enter__(self) -> "JsonlWriter":
        return self

    def __exit__(self, *exc: Any) -> bool:
        self.close()
        return False


class CsvWriter:
    """追加写 CSV（表头只在文件为空时写；每行一条检测，无检测的图也占一行）。"""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        new_file = not self.path.exists() or self.path.stat().st_size == 0
        self._handle = self.path.open("a", encoding="utf-8", newline="")
        self._writer = csv.writer(self._handle)
        if new_file:
            self._writer.writerow(CSV_HEADER)
            self._handle.flush()

    def write(self, record: dict[str, Any]) -> None:
        gps = record.get("gps") or {}
        ts = record.get("ts") or ""
        detections = record.get("detections") or []
        if not detections:
            self._writer.writerow([record.get("image", ""), ts, gps.get("lat", ""), gps.get("lon", ""),
                                   "", "", "", "", "", ""])
        for item in detections:
            x1, y1, x2, y2 = item["bbox"]
            self._writer.writerow([record.get("image", ""), ts, gps.get("lat", ""), gps.get("lon", ""),
                                   item["class"], f"{float(item['conf']):.6f}",
                                   f"{x1:.6f}", f"{y1:.6f}", f"{x2:.6f}", f"{y2:.6f}"])
        self._handle.flush()

    def close(self) -> None:
        self._handle.close()

    def __enter__(self) -> "CsvWriter":
        return self

    def __exit__(self, *exc: Any) -> bool:
        self.close()
        return False


# ─────────────────────────── 断点续跑 ───────────────────────────
@dataclass
class ResumeState:
    """已处理集合 + 统计（键优先用内容哈希，其次用源路径）。"""

    processed: set[str] = field(default_factory=set)
    path: Path | None = None
    records: int = 0
    detections: int = 0

    @staticmethod
    def key_of(image_sha256: str | None, source: str) -> str:
        return str(image_sha256) if image_sha256 else f"path:{source}"

    def mark(self, *, image_sha256: str | None, source: str, detections: int = 0) -> None:
        self.processed.add(self.key_of(image_sha256, source))
        self.records += 1
        self.detections += int(detections)

    def has(self, *, image_sha256: str | None, source: str) -> bool:
        return self.key_of(image_sha256, source) in self.processed

    def as_dict(self) -> dict[str, Any]:
        return {"version": 1, "records": self.records, "detections": self.detections,
                "processed": sorted(self.processed)}


def load_state(state_path: Path, jsonl_path: Path | None = None) -> ResumeState:
    """恢复已完成集合：先读状态文件，再用 JSONL 兜底/补齐（JSONL 是真相）。"""
    state = ResumeState(path=Path(state_path))
    if state.path.exists():
        try:
            payload = json.loads(state.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = {}
        for key in payload.get("processed") or []:
            state.processed.add(str(key))
        state.records = int(payload.get("records") or 0)
        state.detections = int(payload.get("detections") or 0)
    if jsonl_path is not None and Path(jsonl_path).exists():
        seen: set[str] = set()
        records = 0
        detections = 0
        for line in Path(jsonl_path).read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue                     # 断电写坏的最后一行：忽略，不影响其余
            seen.add(ResumeState.key_of(row.get("image_sha256"), str(row.get("source") or row.get("image"))))
            records += 1
            detections += len(row.get("detections") or [])
        state.processed |= seen
        state.records = max(state.records, records)
        state.detections = max(state.detections, detections)
    return state


def save_state(state: ResumeState) -> None:
    """原子写状态文件（临时文件 + os.replace）。"""
    if state.path is None:
        return
    state.path.parent.mkdir(parents=True, exist_ok=True)
    tmp = state.path.with_name(state.path.name + ".tmp")
    tmp.write_text(json.dumps(state.as_dict(), ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, state.path)


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """逐行读 JSONL（坏行跳过）。"""
    if not Path(path).exists():
        return
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            continue


def summarize_records(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """统计 JSONL 结果：图像数、检测数、分类别计数、耗时分布。"""
    images = 0
    detections = 0
    per_class: dict[str, int] = {}
    elapsed: list[float] = []
    for row in records:
        images += 1
        for item in row.get("detections") or []:
            detections += 1
            code = str(item.get("class"))
            per_class[code] = per_class.get(code, 0) + 1
        if isinstance(row.get("elapsed_ms"), (int, float)):
            elapsed.append(float(row["elapsed_ms"]))
    elapsed.sort()

    def _quantile(fraction: float) -> float | None:
        if not elapsed:
            return None
        index = min(len(elapsed) - 1, max(0, int(round(fraction * (len(elapsed) - 1)))))
        return round(elapsed[index], 3)

    return {"images": images, "detections": detections, "per_class": per_class,
            "elapsed_ms": {"mean": round(sum(elapsed) / len(elapsed), 3) if elapsed else None,
                           "p50": _quantile(0.5), "p95": _quantile(0.95),
                           "max": round(elapsed[-1], 3) if elapsed else None}}
