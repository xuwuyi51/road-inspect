"""训练 runner：ultralytics 微调 + 日志流 + 断点 + 硬约束（M3）。

设计要点（与 docs/09-roadmap.md M3 对齐）：

* **幂等**：``run_key`` 由（数据集清单哈希 + 架构 + 初始权重摘要 + 全部超参）决定；
  同一配置重复提交直接返回既有运行，不会重复占用 GPU。
* **硬约束**：``augment.flipud`` 必须为 0——垂直翻转会把「横向裂缝」翻成「纵向裂缝」，
  除非显式 ``allow_vertical_flip=true``，否则直接拒绝（ConfigError）。
* **可观测**：ultralytics 的日志逐行落盘到 ``data/logs/runs/train-<run_id>-*.log``，
  每个 epoch 结束把指标增量写入 ``runs.metrics_json``，前端/CLI 可轮询进度。
* **可中断**：``TrainControl.cancel()`` 后，epoch 回调置 ``trainer.stop_training``，
  已完成的 checkpoint 仍然保留（best.pt/last.pt 可继续用）。

本模块不在导入期引入 torch/ultralytics：未安装 ML 依赖时抛 ``DetectorUnavailable``。
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import re
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from ..config import Config, ConfigError
from ..errors import ConflictError, NotFoundError
from ..prelabel.detector import DetectorUnavailable, ensure_runtime_env, ml_available, resolve_device
from ..storage.files import sha256_file
from ..storage.repo import Repo

#: 进程中断（服务重启/被杀）遗留的 running 运行的收尾说明
INTERRUPTED_ERROR = "进程中断（服务重启或被杀），运行未完成；已允许按同一配置重新提交"

#: flipud 是语义相关的增强，非 0 会破坏「横向/纵向」的标签含义
FLIPUD_CONSTRAINT_MESSAGE = (
    "augment.flipud 必须为 0：垂直翻转会把横向裂缝翻成纵向裂缝（标签语义被破坏）。"
    "如数据中确认不含横/纵向裂缝，可在 train.yaml 显式设置 allow_vertical_flip: true"
)

#: 允许透传给 ultralytics 的增强键（其余键被忽略，避免拼写错误静默生效）
AUGMENT_KEYS = (
    "mosaic", "mixup", "copy_paste", "degrees", "translate", "scale", "shear", "perspective",
    "hsv_h", "hsv_s", "hsv_v", "fliplr", "flipud", "bgr", "erasing", "crop_fraction",
)

#: results.csv 列名 → 归一化指标名
METRIC_ALIASES = {
    "metrics/precision(B)": "precision",
    "metrics/recall(B)": "recall",
    "metrics/mAP50(B)": "map50",
    "metrics/mAP50-95(B)": "map50_95",
    "train/box_loss": "train_box_loss",
    "train/cls_loss": "train_cls_loss",
    "train/dfl_loss": "train_dfl_loss",
    "val/box_loss": "val_box_loss",
    "val/cls_loss": "val_cls_loss",
    "val/dfl_loss": "val_dfl_loss",
    "lr/pg0": "lr",
}

ProgressCallback = Callable[[int, dict[str, Any]], None]
LogSink = Callable[[str], None]


# ─────────────────────────── 控制与结果 ───────────────────────────
@dataclass
class TrainControl:
    """训练中断开关（epoch 边界生效，不丢已完成的 checkpoint）。"""

    canceled: bool = False
    reason: str | None = None

    def cancel(self, reason: str | None = None) -> None:
        self.canceled = True
        self.reason = reason


@dataclass
class TrainRequest:
    """一次训练的完整解析结果（构造过程不依赖 ML 依赖，可单测）。"""

    dataset: dict[str, Any]
    data_yaml: Path
    arch: str
    weights_init: str | None
    resume: bool
    run_name: str
    params: dict[str, Any]
    augment: dict[str, float]
    run_key: str

    @property
    def dataset_name(self) -> str:
        return str(self.dataset["name"])

    @property
    def manifest_hash(self) -> str | None:
        return self.dataset.get("manifest_hash")

    def as_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset_name, "manifest_hash": self.manifest_hash,
            "arch": self.arch, "weights_init": self.weights_init, "resume": self.resume,
            "run_name": self.run_name, "run_key": self.run_key,
            "params": {key: value for key, value in self.params.items() if key != "data"},
            "augment": self.augment, "data_yaml": str(self.data_yaml),
        }


@dataclass
class TrainOutcome:
    """一次训练的结果（写入 runs + model_versions 后返回给调用方）。"""

    run_id: int
    status: str
    dataset: str
    dataset_id: int | None
    manifest_hash: str | None
    weights_path: str | None
    last_weights: str | None
    work_dir: str | None
    log_path: str | None
    metrics: dict[str, Any] = field(default_factory=dict)
    epochs_done: int = 0
    resumed: bool = False
    cached: bool = False
    error: str | None = None
    #: 本次训练的超参（imgsz/batch/epochs/device…）：评估与导出要按它复现同一分辨率
    params: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id, "status": self.status, "dataset": self.dataset,
            "dataset_id": self.dataset_id, "manifest_hash": self.manifest_hash,
            "weights_path": self.weights_path, "last_weights": self.last_weights,
            "work_dir": self.work_dir, "log_path": self.log_path,
            "metrics": self.metrics, "epochs_done": self.epochs_done,
            "resumed": self.resumed, "cached": self.cached, "error": self.error,
            "params": {key: value for key, value in self.params.items() if key != "data"},
        }


# ─────────────────────────── 参数解析 ───────────────────────────
def job_active(run_id: int) -> bool:
    """该运行是否仍是**本进程**内的活任务（本地导入避免与 service 循环依赖）。"""
    try:
        from .service import JOBS  # noqa: PLC0415

        return JOBS.is_active(int(run_id))
    except Exception:  # noqa: BLE001 - 独立调用（无 service 上下文）时按"非活任务"处理
        return False


def _sha256_or_none(path: str | Path | None) -> str | None:
    if not path:
        return None
    candidate = Path(path)
    if candidate.exists() and candidate.is_file():
        return sha256_file(candidate)
    return None


def enforce_hard_constraints(augment: dict[str, float], *, allow_vertical_flip: bool) -> dict[str, float]:
    """硬约束：flipud 必须为 0（除非显式允许）。返回清洗后的增强参数。"""
    cleaned = {key: float(value) for key, value in augment.items() if key in AUGMENT_KEYS}
    flipud = float(cleaned.get("flipud", 0.0) or 0.0)
    if flipud and not allow_vertical_flip:
        raise ConfigError(FLIPUD_CONSTRAINT_MESSAGE)
    cleaned["flipud"] = 0.0 if not allow_vertical_flip else flipud
    return cleaned


def resolve_train_device(prefer: str = "auto") -> str:
    """auto → 有 CUDA 用 cuda，否则 cpu（与预标注一致）。"""
    return resolve_device(prefer or "auto")


def resolve_amp(setting: object, device: str) -> bool:
    """amp=auto 时只有 CUDA 才开启（CPU 上 AMP 检查反而更慢）。"""
    if isinstance(setting, bool):
        return setting
    return device != "cpu"


def dataset_data_yaml(config: Config, repo: Repo, dataset: dict[str, Any]) -> Path:
    """返回数据集的 YOLO data.yaml；缺失时按冻结内容原地补导出（幂等、不改变清单哈希）。"""
    if dataset.get("status") != "frozen":
        raise ConflictError(
            f"数据集 {dataset['name']} 状态为 {dataset.get('status')}，只有 frozen 数据集可训练（见 ADR-0005）")
    root = Path(dataset.get("root_path") or (config.datasets_dir / str(dataset["name"])))
    data_yaml = root / "data.yaml"
    if not data_yaml.exists():
        from ..core import datasets as datasets_mod  # 局部导入避免循环
        from ..core.formats import export_yolo

        items = repo.dataset_items(int(dataset["id"]))
        rows = datasets_mod.fetch_dataset_rows(repo, [item for item in items if item.get("id")])
        abs_rows = [{**row, "abs_path": str(config.abs_data_path(row["image"]["path"]))} for row in rows]
        result = export_yolo(abs_rows, repo.list_classes(), root, copy_images=True)
        data_yaml = Path(result["data_yaml"])
    if not data_yaml.exists():
        raise NotFoundError(f"数据集 {dataset['name']} 缺少 data.yaml，请先 rdinspect export")
    return data_yaml


def absolute_data_yaml(config: Config, data_yaml: Path) -> Path:
    """把 data.yaml 的 ``path`` 改写成绝对路径后再交给 ultralytics。

    坑：ultralytics 对 ``data.yaml`` 里的**相对** ``path`` 是按它自己的 ``datasets_dir`` 解析的
    （不是按 yaml 所在目录），于是 ``path: .`` 会被解析成 ``~/datasets/images/val`` 而报
    "images not found"。冻结数据集保持原样不动，这里在 runs 目录下生成一份绝对路径副本。
    """
    import yaml  # noqa: PLC0415

    raw = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    root = Path(str(raw.get("path") or "."))
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()
    payload = {**raw, "path": str(root)}
    target = config.runs_dir / "_data" / f"{slugify(data_yaml.parent.name)}.yaml"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return target


def _run_key(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:32]


def slugify(value: str) -> str:
    """把数据集/模型名压成可用作目录名的短串。"""
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")
    return slug[:48] or "run"


def build_train_request(config: Config, repo: Repo, *, dataset: str | None = None,
                        arch: str | None = None, resume_from: str | None = None,
                        epochs: int | None = None, imgsz: int | None = None,
                        batch: int | None = None, device: str | None = None,
                        run_name: str | None = None, allow_vertical_flip: bool | None = None,
                        seed: int | None = None, patience: int | None = None) -> TrainRequest:
    """解析训练请求：数据集校验 + 超参合并 + 硬约束 + 幂等键。"""
    train_cfg = config.train
    dataset_name = dataset or train_cfg.dataset
    if not dataset_name:
        raise ValueError("未指定数据集：请传 --dataset 或在 configs/train.yaml 设置 dataset.name")
    row = repo.get_dataset(name=str(dataset_name))
    if row is None:
        raise NotFoundError(f"数据集不存在: {dataset_name}")
    data_yaml = absolute_data_yaml(config, dataset_data_yaml(config, repo, row))

    resolved_arch = str(arch or train_cfg.arch)
    init = resume_from or train_cfg.resume_from
    weights_init = _localize_weights(config, str(init)) if init else _localize_weights(config, resolved_arch)
    resume = bool(init) and is_run_checkpoint(weights_init)

    resolved_device = resolve_device(device or train_cfg.device)
    augment = enforce_hard_constraints(
        train_cfg.augment,
        allow_vertical_flip=train_cfg.allow_vertical_flip if allow_vertical_flip is None else allow_vertical_flip,
    )
    params: dict[str, Any] = {
        "imgsz": int(imgsz or train_cfg.imgsz),
        "epochs": int(epochs or train_cfg.epochs),
        "patience": int(patience if patience is not None else train_cfg.patience),
        "batch": int(batch or train_cfg.batch),
        "workers": int(train_cfg.workers),
        "seed": int(seed if seed is not None else train_cfg.seed),
        "optimizer": str(train_cfg.optimizer),
        "lr0": float(train_cfg.lr0),
        "lrf": float(train_cfg.lrf),
        "warmup_epochs": float(train_cfg.warmup_epochs),
        "cos_lr": bool(train_cfg.cos_lr),
        "close_mosaic": int(train_cfg.close_mosaic),
        "device": resolved_device,
        "amp": resolve_amp(train_cfg.amp, resolved_device),
        "plots": bool(train_cfg.plots),
        "pretrained": bool(train_cfg.pretrained) if not init else False,
        "val": True,
        "save": True,
        "verbose": True,
        "exist_ok": True,
    }
    name = run_name or f"{slugify(row['name'])}-{slugify(Path(resolved_arch).stem)}"
    key = _run_key({
        "kind": "train", "dataset": row["name"], "manifest_hash": row.get("manifest_hash"),
        "arch": resolved_arch, "init": weights_init, "init_sha": _sha256_or_none(weights_init),
        "resume": resume, "params": params, "augment": augment,
    })
    return TrainRequest(dataset=row, data_yaml=data_yaml, arch=resolved_arch, weights_init=weights_init,
                        resume=resume, run_name=name, params=params, augment=augment, run_key=key)


def is_run_checkpoint(weights: str | Path | None) -> bool:
    """判断权重是否为「可续训」的 run checkpoint（ultralytics 的 weights/last.pt）。

    只有 last.pt 才带优化器/epoch 状态；best.pt 或第三方权重一律按「初始化权重」处理。
    """
    if not weights:
        return False
    path = Path(weights)
    return path.name == "last.pt" and path.parent.name == "weights"


def _localize_weights(config: Config, weights: str) -> str:
    """裸文件名（yolo11n.pt）落到 data_dir/weights/，避免下载到进程 CWD。"""
    if not weights or "/" in weights or weights.startswith("~"):
        return weights
    existing = Path(weights)
    if existing.exists():
        return str(existing)
    return str(config.weights_dir / weights)


# ─────────────────────────── 日志与指标 ───────────────────────────
class _FileHandler(logging.Handler):
    """把 ultralytics 的日志逐行落到运行日志文件，并可选转发给 sink（CLI 实时打印）。"""

    def __init__(self, path: Path, sink: LogSink | None = None) -> None:
        super().__init__(level=logging.INFO)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._file = path.open("a", encoding="utf-8")
        self._sink = sink

    def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover - IO
        try:
            line = self.format(record)
            self._file.write(line + "\n")
            self._file.flush()
            if self._sink is not None:
                self._sink(line)
        except Exception:  # noqa: BLE001 - 日志失败不应影响训练
            pass

    def close(self) -> None:  # pragma: no cover - IO
        try:
            self._file.close()
        finally:
            super().close()


def read_results_csv(path: Path) -> list[dict[str, Any]]:
    """读取 ultralytics results.csv（容错：文件不存在/半写入时返回已解析部分）。"""
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            for raw in csv.DictReader(handle):
                row: dict[str, Any] = {}
                for key, value in raw.items():
                    if key is None:
                        continue
                    column = key.strip()
                    try:
                        row[column] = float(str(value).strip())
                    except (TypeError, ValueError):
                        row[column] = str(value).strip()
                rows.append(row)
    except OSError:
        return rows
    return rows


def normalize_metrics(row: dict[str, Any]) -> dict[str, Any]:
    """results.csv 行 → 归一化指标（未识别的列原样保留）。"""
    metrics: dict[str, Any] = {}
    for key, value in row.items():
        metrics[METRIC_ALIASES.get(key, key)] = value
    if "epoch" in metrics:
        try:
            metrics["epoch"] = int(float(metrics["epoch"]))
        except (TypeError, ValueError):
            pass
    return metrics


# ─────────────────────────── 训练主流程 ───────────────────────────
def _build_model(request: TrainRequest) -> Any:
    if not ml_available():
        raise DetectorUnavailable(
            "未安装 ML 依赖（ultralytics/torch）：pip install -e '.[ml]'（详见 docs/08-deployment.md）")
    ensure_runtime_env()
    from .compat import apply_compat_patches, preload_ultralytics  # noqa: PLC0415

    preload_ultralytics()
    apply_compat_patches()
    from ultralytics import YOLO  # noqa: PLC0415

    weights = request.weights_init or request.arch
    try:
        return YOLO(weights)
    except Exception as exc:  # noqa: BLE001 - 权重缺失/损坏
        raise DetectorUnavailable(f"加载初始权重失败 {weights}: {exc}") from exc


def run_training(config: Config, repo: Repo, request: TrainRequest, *, control: TrainControl | None = None,
                 progress: ProgressCallback | None = None, log_sink: LogSink | None = None,
                 run_id: int | None = None) -> TrainOutcome:
    """执行一次微调。

    @param run_id - 调用方预创建的运行 id（API 需要先返回 id、再在后台线程里训练）；
                    None 时按 run_key 幂等创建/复用既有运行。
    """
    if run_id is not None:
        if repo.get_run(int(run_id)) is None:
            raise NotFoundError(f"运行 {run_id} 不存在")
    else:
        existing = repo.conn.execute("SELECT * FROM runs WHERE run_key = ?", (request.run_key,)).fetchone()
        if existing is not None:
            if existing["status"] == "running":
                if job_active(int(existing["id"])):
                    raise ConflictError(f"同一配置的训练正在进行中（run #{existing['id']}），请等待或取消后再提交")
                # 进程重启/被杀后遗留的 running：标记为失败，允许重新提交（否则同一配置永远被挡住）
                repo.finish_run(int(existing["id"]), status="failed", error=INTERRUPTED_ERROR)
                existing = repo.get_run(int(existing["id"]))
            if existing["status"] == "succeeded":
                stored = json.loads(existing["metrics_json"] or "{}")
                return TrainOutcome(
                    run_id=int(existing["id"]), status="succeeded", dataset=request.dataset_name,
                    dataset_id=int(request.dataset["id"]), manifest_hash=request.manifest_hash,
                    weights_path=stored.get("weights_path"), last_weights=stored.get("last_weights"),
                    work_dir=stored.get("work_dir"), log_path=existing["log_path"],
                    metrics=stored.get("final") or {}, epochs_done=int(stored.get("epochs_done") or 0),
                    resumed=bool(stored.get("resumed")), cached=True,
                    params=stored.get("train_params") or {},
                )
            # failed / canceled：允许重跑，复用同一行（保留历史 error 直到本次覆盖）
            run_id = int(existing["id"])
            repo.update_run(run_id, metrics={"restarted_at": _now()})
        else:
            run = repo.create_run("train", run_key=request.run_key, dataset_id=int(request.dataset["id"]),
                                  config_json=request.as_dict())
            run_id = int(run["id"])

    slug = slugify(f"{request.dataset_name}-{Path(request.weights_init or request.arch).stem}")
    log_path = config.train_logs_dir / f"train-{run_id}-{slug}.log"
    work_dir = config.runs_dir / f"run-{run_id}-{slug}"
    if work_dir.exists():
        # 失败重跑会复用同一 run 行：把上一次的产物挪走，避免陈旧的 results.csv/权重污染本次指标
        stale = work_dir.with_name(f"{work_dir.name}.prev-{datetime.now().strftime('%H%M%S')}")
        try:
            work_dir.rename(stale)
        except OSError:
            pass
    repo.update_run(run_id, log_path=str(log_path))

    logger = logging.getLogger("ultralytics")
    handler = _FileHandler(log_path, log_sink)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S"))
    logger.addHandler(handler)
    had_own_level = logger.level
    if logger.level in (logging.NOTSET, logging.WARNING):
        logger.setLevel(logging.INFO)
    own_logger = logging.getLogger("rdinspect.train")
    own_logger.addHandler(handler)

    last_metrics: dict[str, Any] = {}
    epochs_done = 0
    error: str | None = None
    status = "succeeded"
    best_path: str | None = None
    last_path: str | None = None
    try:
        model = _build_model(request)
        params = dict(request.params)
        params.update({"data": str(request.data_yaml), "project": str(config.runs_dir),
                       "name": work_dir.name, **request.augment})
        if request.resume:
            params["resume"] = True
        own_logger.info("[rdinspect] run #%s 开始：dataset=%s manifest=%s arch=%s device=%s epochs=%s imgsz=%s",
                        run_id, request.dataset_name, (request.manifest_hash or "-")[:12],
                        request.arch, params.get("device"), params.get("epochs"), params.get("imgsz"))
        model.add_callback("on_train_epoch_end", _make_epoch_callback(
            run_id=run_id, repo=repo, control=control, progress=progress, logger=own_logger))
        results = model.train(**params)
        save_dir = Path(getattr(results, "save_dir", None) or getattr(model.trainer, "save_dir", work_dir))
        best_path = str(save_dir / "weights" / "best.pt")
        last_path = str(save_dir / "weights" / "last.pt")
        if not Path(best_path).exists():
            best_path = last_path if Path(last_path).exists() else None
        rows = read_results_csv(save_dir / "results.csv")
        if rows:
            last_metrics = normalize_metrics(rows[-1])
            try:
                # results.csv 的 epoch 列是 1 基（第 N 轮 = N），不要 +1
                epochs_done = int(float(last_metrics.get("epoch", len(rows))))
            except (TypeError, ValueError):
                epochs_done = len(rows)
        if control is not None and control.canceled:
            status = "canceled"
            own_logger.info("[rdinspect] run #%s 被取消：%s", run_id, control.reason or "用户取消")
        own_logger.info("[rdinspect] run #%s 结束：status=%s epochs=%s map50=%s weights=%s",
                        run_id, status, epochs_done, last_metrics.get("map50"), best_path)
    except DetectorUnavailable:
        raise
    except Exception as exc:  # noqa: BLE001 - 训练失败要落库
        status = "failed"
        error = f"{type(exc).__name__}: {exc}"
        own_logger.error("[rdinspect] run #%s 失败：%s\n%s", run_id, error, traceback.format_exc())
    finally:
        logger.removeHandler(handler)
        own_logger.removeHandler(handler)
        logger.setLevel(had_own_level)
        handler.close()

    payload = {
        "weights_path": best_path, "last_weights": last_path, "work_dir": str(work_dir),
        "final": last_metrics, "epochs_done": epochs_done, "resumed": request.resume,
        "dataset": request.dataset_name, "manifest_hash": request.manifest_hash,
        "arch": request.arch, "train_params": {key: value for key, value in request.params.items()
                                               if key != "data"},
    }
    repo.finish_run(run_id, status=status, metrics=payload, error=error)

    outcome = TrainOutcome(run_id=run_id, status=status, dataset=request.dataset_name,
                           dataset_id=int(request.dataset["id"]), manifest_hash=request.manifest_hash,
                           weights_path=best_path, last_weights=last_path, work_dir=str(work_dir),
                           log_path=str(log_path), metrics=last_metrics, epochs_done=epochs_done,
                           resumed=request.resume, error=error, params=dict(request.params))
    return outcome


def _make_epoch_callback(*, run_id: int, repo: Repo, control: TrainControl | None,
                         progress: ProgressCallback | None, logger: logging.Logger) -> Callable[[Any], None]:
    """epoch 结束回调：写进度、响应取消。"""

    def _callback(trainer: Any) -> None:
        save_dir = Path(getattr(trainer, "save_dir", "."))
        rows = read_results_csv(save_dir / "results.csv")
        metrics = normalize_metrics(rows[-1]) if rows else {}
        epoch = int(getattr(trainer, "epoch", len(rows) - 1)) + 1
        metrics["epoch"] = epoch
        try:
            repo.update_run(run_id, metrics={"progress": metrics, "epochs_done": epoch})
        except Exception as exc:  # noqa: BLE001 - 进度写库失败不能中断训练
            logger.warning("[rdinspect] 进度写库失败：%s", exc)
        logger.info("[rdinspect] epoch %s/%s map50=%s map50_95=%s", epoch,
                    getattr(trainer, "epochs", "?"), metrics.get("map50"), metrics.get("map50_95"))
        if progress is not None:
            progress(epoch, metrics)
        if control is not None and control.canceled:
            logger.info("[rdinspect] 收到取消请求，本 epoch 后停止训练")
            trainer.stop_training = True

    return _callback


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def dataset_split_counts(data_yaml: Path) -> dict[str, int]:
    """统计 data.yaml 指向的各划分图片数（训练前做「空划分」体检）。"""
    import yaml  # noqa: PLC0415

    try:
        raw = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    except Exception:  # noqa: BLE001 - YAML 缺失/损坏时按「无法体检」处理
        return {}
    root = Path(raw.get("path") or data_yaml.parent)
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()
    counts: dict[str, int] = {}
    for split in ("train", "val", "test"):
        entry = raw.get(split)
        if not entry:
            continue
        target = Path(entry)
        directory = target if target.is_absolute() else (root / target)
        if directory.is_file():  # 允许 txt 清单
            counts[split] = sum(1 for line in directory.read_text(encoding="utf-8").splitlines() if line.strip())
        elif directory.is_dir():
            counts[split] = sum(1 for path in directory.iterdir()
                                if path.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"))
        else:
            counts[split] = 0
    return counts


def ensure_trainable(request: TrainRequest) -> dict[str, int]:
    """训练前体检：train/val 划分不能为空（否则 ultralytics 会以晦涩错误失败）。"""
    counts = dataset_split_counts(request.data_yaml)
    if not counts.get("train"):
        raise ValueError(f"数据集 {request.dataset_name} 的 train 划分为空，无法训练")
    if not counts.get("val"):
        raise ValueError(f"数据集 {request.dataset_name} 的 val 划分为空，无法评估；请调整切分比例")
    return counts


def summarize_request(request: TrainRequest) -> dict[str, Any]:
    """给 CLI/日志的紧凑摘要。"""
    return {"dataset": request.dataset_name, "manifest_hash": request.manifest_hash,
            "arch": request.arch, "weights_init": request.weights_init, "resume": request.resume,
            "run_name": request.run_name, "run_key": request.run_key,
            "epochs": request.params.get("epochs"), "imgsz": request.params.get("imgsz"),
            "batch": request.params.get("batch"), "device": request.params.get("device"),
            "augment": request.augment}


def class_names_for(repo: Repo) -> list[str]:
    """按冻结顺序（order_index, code）取类别 code —— 与数据集 data.yaml 的顺序必须一致。"""
    classes = repo.list_classes()
    return [cls["code"] for cls in sorted(classes, key=lambda c: (c["order_index"], c["code"]))]


def register_trained_model(config: Config, repo: Repo, outcome: TrainOutcome, *, name: str,
                           version: str | None = None, actor: str = "system") -> dict[str, Any]:
    """把训练产物登记为 candidate 模型（幂等：同名同版本直接返回）。"""
    if outcome.status != "succeeded" or not outcome.weights_path:
        raise ConflictError(f"运行 #{outcome.run_id} 状态为 {outcome.status}，没有可登记的权重")
    names = class_names_for(repo)
    resolved_version = version or f"{datetime.now().strftime('%Y.%m.%d')}-r{outcome.run_id}"
    digest = _sha256_or_none(outcome.weights_path)
    model = repo.upsert_model_version(
        name=name, version=resolved_version, task="detection", status="candidate",
        weights_path=outcome.weights_path, run_id=outcome.run_id, dataset_id=outcome.dataset_id,
        labels_json={"names": names, "nc": len(names), "source": "train",
                     "manifest_hash": outcome.manifest_hash, "arch": outcome.metrics.get("arch")},
        metrics_json={"train": outcome.metrics, "epochs_done": outcome.epochs_done,
                      "log_path": outcome.log_path,
                      "train_params": {key: value for key, value in outcome.params.items()
                                       if key != "data"}},
        actor=actor,
    )
    if digest and not model.get("sha256"):
        repo.update_model_version(int(model["id"]), sha256=digest, actor=actor)
        model = repo.get_model_version(int(model["id"])) or model
    return model


def train_and_register(config: Config, repo: Repo, *, dataset: str | None = None, name: str | None = None,
                       version: str | None = None, arch: str | None = None, resume_from: str | None = None,
                       epochs: int | None = None, imgsz: int | None = None, batch: int | None = None,
                       device: str | None = None, control: TrainControl | None = None,
                       progress: ProgressCallback | None = None, log_sink: LogSink | None = None,
                       actor: str = "system") -> dict[str, Any]:
    """训练 + 登记模型（CLI/API 共用的门面）。"""
    request = build_train_request(config, repo, dataset=dataset, arch=arch, resume_from=resume_from,
                                  epochs=epochs, imgsz=imgsz, batch=batch, device=device)
    counts = ensure_trainable(request)
    outcome = run_training(config, repo, request, control=control, progress=progress, log_sink=log_sink)
    model: dict[str, Any] | None = None
    if outcome.status == "succeeded":
        model_name = name or f"{slugify(Path(request.arch).stem)}-road"
        model = register_trained_model(config, repo, outcome, name=model_name, version=version, actor=actor)
    return {"request": summarize_request(request), "outcome": outcome.as_dict(), "model": model,
            "splits": counts}
