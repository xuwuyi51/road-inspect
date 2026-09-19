"""受限运行环境兼容层（M3）。

问题：ultralytics 在**扫描/缓存标签**时会创建 ``multiprocessing.pool.ThreadPool``，
而 ``ThreadPool`` 继承 ``Pool``，初始化就要创建 POSIX 信号量（``/dev/shm``）。
在只挂载了工作目录的受限容器/沙箱里 ``/dev/shm`` 不可写，于是训练在
"Fast image access ✅" 之后立刻 ``PermissionError: [Errno 13]`` 挂掉——报错点与真实原因
（信号量）相距很远，排查成本高。

处理方式：**先探测**（尝试建一个 SimpleQueue）。探测失败时，把 ultralytics 里绑定到
各模块的 ``ThreadPool`` 替换成一个纯串行的等价实现（只用 ``imap``/``map`` 接口）。
标签扫描本身是轻量 I/O，串行只是慢一点，结果完全一致；探测成功时本模块什么都不做，
正常环境零影响，也不会掩盖真正的错误。

可用 ``RDINSPECT_SERIAL_SCAN=1`` 强制执行（例如想限制线程数的边缘设备），
``RDINSPECT_SERIAL_SCAN=0`` 禁止该回退。
"""

from __future__ import annotations

import os
import sys
from typing import Any, Callable, Iterable, Iterator

PATCHED_MODULES = ("ultralytics.data.dataset", "ultralytics.data.base", "ultralytics.data.loaders",
                   "ultralytics.utils.downloads", "ultralytics.models.fastsam.predict")

#: 已经打过补丁的进程不再重复探测（探测本身会创建并丢弃一个队列）
_STATE: dict[str, Any] = {"checked": False, "patched": False, "reason": None}


class _SerialPool:
    """``multiprocessing.pool.ThreadPool`` 的最小串行替身（只用上下文管理器 + imap/map）。"""

    def __init__(self, processes: int | None = None, *args: Any, **kwargs: Any) -> None:
        self.processes = processes or 1

    def __enter__(self) -> "_SerialPool":
        return self

    def __exit__(self, *exc: Any) -> bool:
        self.close()
        return False

    def imap(self, func: Callable[[Any], Any], iterable: Iterable[Any],
             chunksize: int = 1) -> Iterator[Any]:
        return (func(item) for item in iterable)

    def imap_unordered(self, func: Callable[[Any], Any], iterable: Iterable[Any],
                       chunksize: int = 1) -> Iterator[Any]:
        return self.imap(func, iterable, chunksize)

    def map(self, func: Callable[[Any], Any], iterable: Iterable[Any],
            chunksize: int = 1) -> list[Any]:
        return [func(item) for item in iterable]

    def starmap(self, func: Callable[..., Any], iterable: Iterable[Any],
                chunksize: int = 1) -> list[Any]:
        return [func(*item) for item in iterable]

    def close(self) -> None:
        return None

    def join(self) -> None:
        return None

    def terminate(self) -> None:
        return None


def semaphores_available() -> bool:
    """POSIX 信号量是否可用（受限容器里 /dev/shm 不可写时为 False）。"""
    try:
        import multiprocessing  # noqa: PLC0415

        queue = multiprocessing.SimpleQueue()
        queue.close()
        return True
    except Exception:  # noqa: BLE001 - 任何失败都按不可用处理
        return False


def apply_compat_patches(*, force: bool | None = None) -> dict[str, Any]:
    """按需给 ultralytics 打上串行扫描补丁；返回诊断信息（幂等）。"""
    if _STATE["checked"]:
        return {"patched": _STATE["patched"], "reason": _STATE["reason"], "checked": True}

    env = os.environ.get("RDINSPECT_SERIAL_SCAN")
    want = bool(force) if force is not None else (env not in (None, "", "0"))
    reason: str | None = None
    patched = False
    if not want and not semaphores_available():
        want = True
        reason = "/dev/shm 不可写：multiprocessing 信号量不可用"
    if want:
        count = 0
        for name in PATCHED_MODULES:
            module = sys.modules.get(name)
            if module is None or not hasattr(module, "ThreadPool"):
                continue
            module.ThreadPool = _SerialPool            # type: ignore[attr-defined]
            count += 1
        patched = count > 0
        reason = reason or ("RDINSPECT_SERIAL_SCAN 显式开启" if env not in (None, "", "0") else reason)

    _STATE.update({"checked": True, "patched": patched, "reason": reason})
    return {"patched": patched, "reason": reason, "checked": True}


def reset_state() -> None:
    """仅供测试：清掉探测缓存。"""
    _STATE.update({"checked": False, "patched": False, "reason": None})


def preload_ultralytics() -> None:
    """先导入 ultralytics 的目标子模块，再打补丁（否则补丁会打在尚未导入的模块上）。"""
    import importlib  # noqa: PLC0415

    for name in PATCHED_MODULES:
        try:
            importlib.import_module(name)
        except Exception:  # noqa: BLE001 - 缺哪个模块就少补一个，不影响训练
            continue
