"""测试公共夹具：临时数据目录、合成影像、仓储实例。"""

from __future__ import annotations

import random
from pathlib import Path

from PIL import Image, ImageDraw

from rdinspect.config import Config
from rdinspect.storage.db import init_db
from rdinspect.storage.repo import Repo

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def make_config(tmp_path: Path, **overrides) -> Config:
    """测试用配置：数据根在临时目录，白名单指向 data/inbox。"""
    data_dir = tmp_path / "data"
    config = Config(raw={}, data_dir=data_dir, allowed_roots=(data_dir / "inbox",),
                    thumb_long_edge=128, lease_seconds=600, **overrides)
    config.ensure_dirs()
    return config


def open_repo(config: Config) -> tuple[object, Repo]:
    conn = init_db(config.db_path)
    return conn, Repo(conn)


def synth_images(target: Path, count: int = 6, size: tuple[int, int] = (96, 64),
                 seed: int = 1) -> list[Path]:
    """生成互不相同的合成路面图（含一条裂缝与随机噪点，保证 pHash 可区分）。"""
    target.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    paths: list[Path] = []
    for index in range(count):
        image = Image.new("RGB", size, (70 + index * 7 % 40, 70, 72))
        draw = ImageDraw.Draw(image)
        draw.line([(3, 5 + index * 4 % 40), (size[0] - 3, 9 + index * 6 % 40)], fill=(20, 20, 22), width=2)
        for _ in range(6):
            x, y = rng.randint(0, size[0] - 6), rng.randint(0, size[1] - 6)
            draw.rectangle([x, y, x + 4, y + 4], fill=(rng.randint(30, 200),) * 3)
        path = target / f"img_{index:03d}.jpg"
        image.save(path, format="JPEG", quality=90)
        paths.append(path)
    return paths


def annotate_all(repo: Repo, *, class_code: str = "transverse_crack",
                 bbox: dict[str, float] | None = None, approve: bool = True) -> list[int]:
    """把当前所有 pending 任务标注为同一类别，可选直接复核通过；返回 task ids。"""
    box = bbox or {"x1": 0.1, "y1": 0.2, "x2": 0.4, "y2": 0.35}
    task_ids: list[int] = []
    for task in repo.list_tasks(status="pending", limit=1000):
        task_id = task["id"]
        repo.replace_annotations(task_id, [{"class_code": class_code, "kind": "bbox", "bbox": box}],
                                 actor="test")
        repo.submit_task(task_id, actor="test")
        if approve:
            repo.add_review(task_id, decision="approve", reviewer="test")
        task_ids.append(task_id)
    return task_ids
