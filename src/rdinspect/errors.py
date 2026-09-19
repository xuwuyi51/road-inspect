"""领域异常：由 HTTP 层映射为标准状态码（见 docs/06-api-spec.md §3）。

* NotFoundError  → 404
* ConflictError  → 409（状态不允许、重名、门禁未通过）
* ValueError     → 400（参数/坐标/类别非法）
"""

from __future__ import annotations


class NotFoundError(KeyError):
    """资源不存在。"""


class ConflictError(ValueError):
    """当前状态不允许该操作（如对非 draft 数据集冻结、重复冻结）。"""
