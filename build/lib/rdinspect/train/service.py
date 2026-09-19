"""训练服务门面：进程内后台任务注册表 + 运行详情（API/CLI 共用）。

为什么用线程而不是队列：工作站是单进程单写者（SQLite WAL）模型，训练本身是长任务，
放进后台线程即可满足「提交后立刻拿到 run_id，前端轮询进度」；真正的状态一律以 ``runs`` 表为准，
本模块的注册表只保存**中断开关**，进程重启后不会假装有任务在跑。
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from ..config import Config
from ..errors import ConflictError, NotFoundError
from ..prelabel.detector import DetectorUnavailable, ml_available
from ..storage.db import init_db
from ..storage.repo import Repo
from .runner import (INTERRUPTED_ERROR, TrainControl, TrainRequest, build_train_request,
                     ensure_trainable, register_trained_model, run_training, slugify,
                     summarize_request)


class TrainingJobs:
    """进程内训练任务注册表（run_id → 中断开关/线程）。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._controls: dict[int, TrainControl] = {}
        self._threads: dict[int, threading.Thread] = {}

    def register(self, run_id: int, control: TrainControl, thread: threading.Thread | None = None) -> None:
        with self._lock:
            self._controls[int(run_id)] = control
            if thread is not None:
                self._threads[int(run_id)] = thread

    def control(self, run_id: int) -> TrainControl | None:
        with self._lock:
            return self._controls.get(int(run_id))

    def thread(self, run_id: int) -> threading.Thread | None:
        with self._lock:
            return self._threads.get(int(run_id))

    def cancel(self, run_id: int, reason: str | None = None) -> bool:
        control = self.control(run_id)
        if control is None:
            return False
        control.cancel(reason)
        return True

    def unregister(self, run_id: int) -> None:
        with self._lock:
            self._controls.pop(int(run_id), None)
            self._threads.pop(int(run_id), None)

    def active(self) -> list[int]:
        with self._lock:
            return sorted(self._controls)

    def is_active(self, run_id: int) -> bool:
        return self.control(run_id) is not None


#: 进程级单例（测试可自行 new 一个）
JOBS = TrainingJobs()


def stale_runs(repo: Repo) -> list[dict[str, Any]]:
    """状态是 running、但本进程里没有对应任务 → 上一次进程留下的僵尸运行。"""
    rows = repo.conn.execute(
        "SELECT * FROM runs WHERE status = 'running' ORDER BY id").fetchall()
    return [dict(row) for row in rows if not JOBS.is_active(int(row["id"]))]


def reconcile_stale_runs(repo: Repo) -> list[int]:
    """把僵尸 running 标成 failed（服务启动时调用；幂等）。"""
    fixed: list[int] = []
    for row in stale_runs(repo):
        repo.update_run(int(row["id"]), metrics={"interrupted_at": summarize_request_stub(row)})
        repo.finish_run(int(row["id"]), status="failed", error=INTERRUPTED_ERROR)
        fixed.append(int(row["id"]))
    return fixed


def summarize_request_stub(row: dict[str, Any]) -> dict[str, Any]:
    """僵尸运行的收尾备注（只留最小信息，完整配置本来就在 config_json 里）。"""
    return {"kind": row.get("kind"), "note": "服务重启时发现该运行仍在 running，已收尾为 failed"}


def _model_for_run(repo: Repo, run_id: int) -> dict[str, Any] | None:
    row = repo.conn.execute(
        "SELECT * FROM model_versions WHERE run_id = ? ORDER BY id DESC LIMIT 1", (int(run_id),)).fetchone()
    return dict(row) if row is not None else None


def start_training(config: Config, *, dataset: str | None = None, name: str | None = None,
                   version: str | None = None, arch: str | None = None, resume_from: str | None = None,
                   epochs: int | None = None, imgsz: int | None = None, batch: int | None = None,
                   device: str | None = None, actor: str = "api", background: bool = True) -> dict[str, Any]:
    """校验参数 → 建 run →（后台线程 | 同步）训练 → 成功后登记 candidate 模型。

    @param background - False 时同步执行（小样本/测试用），返回时 run 已是终止态。
    """
    if not ml_available():
        # 尽早失败：依赖缺失时不要先返回 202、再让后台线程把 run 置为 failed
        raise DetectorUnavailable(
            "未安装 ML 依赖（ultralytics/torch）：pip install -e '.[ml]'（详见 docs/08-deployment.md）")
    config.ensure_dirs()
    conn = init_db(config.db_path)
    repo = Repo(conn)
    try:
        request = build_train_request(config, repo, dataset=dataset, arch=arch, resume_from=resume_from,
                                      epochs=epochs, imgsz=imgsz, batch=batch, device=device)
        splits = ensure_trainable(request)
        existing = repo.conn.execute("SELECT * FROM runs WHERE run_key = ?", (request.run_key,)).fetchone()
        if existing is not None:
            status = str(existing["status"])
            if status == "running" and JOBS.is_active(int(existing["id"])):
                raise ConflictError(
                    f"同一配置的训练已在运行中（run #{existing['id']}）；如需中止请调用取消接口")
            if status == "running":                      # 僵尸运行：收尾后按同一行重跑
                repo.finish_run(int(existing["id"]), status="failed", error=INTERRUPTED_ERROR)
                existing = repo.get_run(int(existing["id"]))
            if status == "succeeded":
                return {"run": dict(existing), "request": summarize_request(request), "splits": splits,
                        "model": _model_for_run(repo, int(existing["id"])), "cached": True,
                        "started": False, "run_id": int(existing["id"])}
        if existing is not None:  # failed / canceled：复用同一行重跑
            run = repo.update_run(int(existing["id"]), metrics={"restarted_at": summarize_request(request)}) or {}
        else:
            run = repo.create_run("train", run_key=request.run_key,
                                  dataset_id=int(request.dataset["id"]), config_json=request.as_dict())
        run_id = int(run["id"])
        control = TrainControl()
        if background:
            thread = threading.Thread(target=_worker, name=f"rdinspect-train-{run_id}", daemon=True,
                                      args=(config, run_id, request, control, name, version, actor))
            JOBS.register(run_id, control, thread)
            thread.start()
            model = None
        else:
            JOBS.register(run_id, control)
            _worker(config, run_id, request, control, name, version, actor)
            model = _model_for_run(repo, run_id)
        payload = {"run": repo.get_run(run_id), "request": summarize_request(request), "splits": splits,
                   "model": model, "cached": False, "started": True, "run_id": run_id,
                   "background": background}
    finally:
        conn.close()
    return payload


def _worker(config: Config, run_id: int, request: TrainRequest, control: TrainControl,
            name: str | None, version: str | None, actor: str) -> None:
    """后台训练线程：自己的数据库连接（SQLite 连接不可跨线程共享）。"""
    conn = init_db(config.db_path)
    repo = Repo(conn)
    try:
        outcome = run_training(config, repo, request, control=control, run_id=run_id)
        if outcome.status == "succeeded":
            model_name = name or f"{slugify(Path(request.arch).stem)}-road"
            register_trained_model(config, repo, outcome, name=model_name, version=version, actor=actor)
    except Exception as exc:  # noqa: BLE001 - 后台线程异常必须落库，否则 run 永远停在 running
        run = repo.get_run(run_id)
        if run is not None and run["status"] == "running":
            repo.finish_run(run_id, status="failed", error=f"{type(exc).__name__}: {exc}")
    finally:
        JOBS.unregister(run_id)
        conn.close()


def cancel_training(repo: Repo, run_id: int, *, reason: str | None = None) -> dict[str, Any]:
    """请求中止训练（epoch 边界生效；已产出的 checkpoint 保留）。"""
    run = repo.get_run(run_id)
    if run is None:
        raise NotFoundError(f"运行 {run_id} 不存在")
    if run["status"] != "running":
        raise ConflictError(f"运行 {run_id} 状态为 {run['status']}，无法取消")
    if not JOBS.cancel(run_id, reason):
        raise ConflictError(f"运行 {run_id} 不在当前进程内（可能是服务重启前的任务），"
                            f"请直接终止对应训练进程")
    return {"run_id": int(run_id), "status": run["status"], "cancel_requested": True,
            "note": "将在当前 epoch 结束时停止；已生成的 best.pt/last.pt 仍可用于续训"}


def run_detail(config: Config, repo: Repo, run_id: int, *, log_lines: int = 40) -> dict[str, Any]:
    """运行详情：状态 + 进度 + 产物 + 日志尾部。"""
    run = repo.get_run(run_id)
    if run is None:
        raise NotFoundError(f"运行 {run_id} 不存在")
    try:
        metrics = json.loads(run.get("metrics_json") or "{}")
    except json.JSONDecodeError:
        metrics = {}
    log_path = run.get("log_path")
    tail: list[str] = []
    if log_path and Path(str(log_path)).exists() and log_lines > 0:
        try:
            lines = Path(str(log_path)).read_text(encoding="utf-8", errors="replace").splitlines()
            tail = lines[-int(log_lines):]
        except OSError:
            tail = []
    detail = dict(run)
    detail["metrics"] = metrics
    detail["progress"] = metrics.get("progress") or (metrics.get("final") or None)
    detail["epochs_done"] = metrics.get("epochs_done")
    detail["final"] = metrics.get("final")
    detail["weights_path"] = metrics.get("weights_path")
    detail["work_dir"] = metrics.get("work_dir")
    detail["log_tail"] = tail
    detail["model"] = _model_for_run(repo, int(run_id))
    detail["in_process"] = JOBS.is_active(int(run_id))
    return detail
