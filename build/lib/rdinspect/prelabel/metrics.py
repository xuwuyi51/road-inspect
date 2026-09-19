"""预标注质量看板：采纳率与「模型-人工一致性」。

定义（写入 docs/04-annotation-workflow.md §5.2）：
  * 候选（candidate）  : source = 'model' 且未删除        → 等待人工处理
  * 已采纳（adopted）  : source = 'model_edited'（含随后被微调替换而软删的行）
                         —— "曾经被人工确认"的候选数
  * 已忽略（ignored）  : source = 'model' 且已软删        → 人工判定为误检
  * 采纳率（adoption_rate）= adopted / (adopted + ignored + candidate)
  * 模型-人工一致性    : 同一任务内，把「模型的原始提案」（source=model / model_edited，含被人工
                        微调替换后软删的行）与「最终人工框」按 IoU 贪心配对（IoU ≥ 0.1），
                        取平均 IoU；未配对的人工框说明模型漏检（记录 match_rate）
"""

from __future__ import annotations

from typing import Any

from ..core.geometry import bbox_iou
from ..storage.repo import Repo

MATCH_IOU_FLOOR = 0.1


def _pending_model_count(repo: Repo) -> int:
    """仍未处理（未删除）的模型候选数。"""
    row = repo.conn.execute(
        "SELECT COUNT(*) AS n FROM annotations WHERE source = 'model' AND deleted_at IS NULL"
    ).fetchone()
    return int(row["n"]) if row else 0


def _adopted_count(repo: Repo) -> int:
    """曾经被采纳的候选数（含被人工微调替换后软删的原行）。"""
    row = repo.conn.execute(
        "SELECT COUNT(*) AS n FROM annotations WHERE source = 'model_edited'").fetchone()
    return int(row["n"]) if row else 0


def _deleted_model_count(repo: Repo) -> int:
    row = repo.conn.execute(
        "SELECT COUNT(*) AS n FROM annotations WHERE source = 'model' AND deleted_at IS NOT NULL"
    ).fetchone()
    return int(row["n"]) if row else 0


def _agreement(repo: Repo, *, sample_tasks: int = 200) -> tuple[float | None, float | None, int]:
    """返回 (平均匹配 IoU, 人工框被模型覆盖比例, 参与统计的任务数)。"""
    rows = repo.conn.execute(
        """SELECT task_id,
                  SUM(CASE WHEN source IN ('model','model_edited') THEN 1 ELSE 0 END) AS model_n,
                  SUM(CASE WHEN source = 'human' AND deleted_at IS NULL THEN 1 ELSE 0 END) AS human_n
           FROM annotations WHERE kind = 'bbox'
           GROUP BY task_id
           HAVING model_n > 0 AND human_n > 0
           ORDER BY task_id DESC LIMIT ?""", (sample_tasks,)).fetchall()
    if not rows:
        return None, None, 0

    ious: list[float] = []
    matched_human = 0
    total_human = 0
    for row in rows:
        task_id = row["task_id"]
        # 模型提案含软删行（人工微调会替换掉原提案）；人工框只看未删除的
        proposal_rows = repo.conn.execute(
            """SELECT source, bbox_x1, bbox_y1, bbox_x2, bbox_y2 FROM annotations
               WHERE task_id = ? AND kind = 'bbox'
                 AND ((source IN ('model','model_edited'))
                      OR (source = 'human' AND deleted_at IS NULL))""", (task_id,)).fetchall()
        model_boxes = [{"x1": r["bbox_x1"], "y1": r["bbox_y1"], "x2": r["bbox_x2"], "y2": r["bbox_y2"]}
                       for r in proposal_rows if r["source"] in ("model", "model_edited")]
        human_boxes = [{"x1": r["bbox_x1"], "y1": r["bbox_y1"], "x2": r["bbox_x2"], "y2": r["bbox_y2"]}
                       for r in proposal_rows if r["source"] == "human"]
        total_human += len(human_boxes)
        pairs: list[tuple[float, int, int]] = []
        for human_index, human_box in enumerate(human_boxes):
            for model_index, model_box in enumerate(model_boxes):
                pairs.append((bbox_iou(human_box, model_box), human_index, model_index))
        used_human: set[int] = set()
        used_model: set[int] = set()
        for overlap, human_index, model_index in sorted(pairs, reverse=True):
            if overlap < MATCH_IOU_FLOOR:
                break
            if human_index in used_human or model_index in used_model:
                continue
            used_human.add(human_index)
            used_model.add(model_index)
            ious.append(overlap)
        matched_human += len(used_human)

    mean_iou = round(sum(ious) / len(ious), 4) if ious else None
    match_rate = round(matched_human / total_human, 4) if total_human else None
    return mean_iou, match_rate, len(rows)


def prelabel_metrics(repo: Repo) -> dict[str, Any]:
    """预标注质量看板数据（供 API / CLI / 前端展示）。"""
    candidates = _pending_model_count(repo)          # 未删除的候选
    adopted = _adopted_count(repo)                   # 曾经被采纳（含微调后软删的原行）
    ignored = _deleted_model_count(repo)             # 已忽略（软删）
    denominator = adopted + ignored + candidates
    mean_iou, match_rate, sampled = _agreement(repo)
    return {
        "tasks_prelabeled": repo.tasks_by_prelabel_state().get("done", 0),
        "candidates_total": candidates,
        "adopted": adopted,
        "ignored": ignored,
        "adoption_rate": round(adopted / denominator, 4) if denominator else None,
        "model_human_iou_mean": mean_iou,
        "model_human_match_rate": match_rate,
        "agreement_sampled_tasks": sampled,
        "prelabel_states": repo.tasks_by_prelabel_state(),
    }
