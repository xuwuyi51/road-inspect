"""FastAPI 应用：M1 端点集（导入 / 任务 / 标注 / 复核 / 数据集 / 统计）+ 静态标注台。

与 docs/openapi.yaml 保持一致；M1 未实现的端点（prelabel/train/models）留到 M2/M3，
未实现时返回 501 与明确提示，避免前端误判。
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from fastapi import Depends, FastAPI, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ..config import Config, ConfigError, load_config
from ..core import datasets as datasets_mod
from ..core.ingest import import_path
from ..errors import ConflictError, NotFoundError
from ..prelabel.detector import DetectorUnavailable
from ..prelabel.metrics import prelabel_metrics
from ..prelabel.service import attach_mask_for_annotation, build_detector, prelabel_tasks
from ..storage.db import init_db
from ..storage.repo import Repo

STATIC_DIR = Path(__file__).resolve().parent / "static"
UPLOAD_INBOX_NAME = "inbox"


def serialize_annotation(row: dict[str, Any]) -> dict[str, Any]:
    """把数据库行转换为契约形态（docs/openapi.yaml#Annotation）：bbox 收敛为对象，去掉原始列。"""
    return {
        "id": row.get("id"),
        "task_id": row.get("task_id"),
        "image_id": row.get("image_id"),
        "class_code": row.get("class_code"),
        "kind": row.get("kind", "bbox"),
        "bbox": row.get("bbox") or {
            "x1": row.get("bbox_x1"), "y1": row.get("bbox_y1"),
            "x2": row.get("bbox_x2"), "y2": row.get("bbox_y2"),
        },
        "difficult": bool(row.get("difficult", 0)),
        "score": row.get("score"),
        "source": row.get("source"),
        "model_version_id": row.get("model_version_id"),
        "created_at": row.get("created_at"),
        "updated_at": row.get("updated_at"),
    }


# ─────────────────────────── 请求模型 ───────────────────────────
class ClassCreate(BaseModel):
    code: str = Field(pattern=r"^[a-z][a-z0-9_]{2,31}$")
    name_zh: str
    name_en: str
    color: str = "#e6194b"
    parent_code: str | None = None
    is_crack: bool = False
    order_index: int = 100


class IngestRequest(BaseModel):
    kind: str = Field(pattern="^(photo|video|aerial|external)$")
    source_dir: str | None = None
    note: str | None = None
    fps: float | None = None
    max_frames: int | None = None
    tile: bool | None = None
    limit: int | None = None


class LeaseRequest(BaseModel):
    count: int = Field(ge=1, le=200, default=1)
    status: str = "pending"
    assignee: str = "web"
    lease_seconds: int | None = None


class AnnotationInput(BaseModel):
    id: int | None = None
    class_code: str
    kind: str = "bbox"
    bbox: dict[str, float]
    difficult: bool = False
    score: float | None = None
    source: str | None = None


class AnnotationsPut(BaseModel):
    annotations: list[AnnotationInput]


class ReviewRequest(BaseModel):
    decision: str = Field(pattern="^(approve|reject)$")
    reason_code: str | None = None
    note: str | None = None
    reviewer: str | None = None


class DatasetDraftRequest(BaseModel):
    name: str = Field(pattern=r"^ds-[0-9a-zA-Z._-]{3,40}$")
    filter: dict[str, Any] = Field(default_factory=dict)
    split: dict[str, Any] = Field(default_factory=dict)


class PrelabelBatchRequest(BaseModel):
    limit: int = Field(ge=1, le=1000, default=50)
    status: str = "pending"
    model: str | None = None
    conf: float | None = None
    iou: float | None = None
    sam: bool | None = None
    sam_limit: int = Field(ge=0, le=20, default=2)


class DatasetExportRequest(BaseModel):
    formats: list[str] = Field(default_factory=lambda: ["yolo", "coco", "labelme"])
    copy_images: bool = True


# ─────────────────────────── 应用工厂 ───────────────────────────
def create_app(config: Config | None = None, *, detector_factory=None,
               masker_factory=None) -> FastAPI:
    """构建应用。

    @param config - 运行期配置（默认读 configs/default.yaml）
    @param detector_factory - 可注入的检测器工厂 `(config, weights) -> Detector`；
                              默认用 ultralytics 适配器，测试可注入假实现（无需 ML 依赖）
    @param masker_factory  - 可注入的掩膜器工厂 `(config) -> SamMasker`
    """
    cfg = config or load_config()
    cfg.ensure_dirs()

    def make_detector(weights: str, conf: float | None = None, iou: float | None = None):
        if detector_factory is not None:
            return detector_factory(cfg, weights)
        return build_detector(cfg, weights, conf=conf, iou=iou)

    def make_masker():
        if masker_factory is not None:
            return masker_factory(cfg)
        from ..prelabel.sam import SamMasker  # 局部导入：未安装 ML 依赖时不在导入期失败
        from ..prelabel.service import _localize_weights  # noqa: PLC0415

        return SamMasker(_localize_weights(cfg, cfg.prelabel.sam.model), device=cfg.prelabel.device)

    app = FastAPI(title="road-inspect", version="0.1.0",
                  description="轻量级道路灾害巡查：采集 → 标注 → 复核 → 数据集冻结与导出")

    def repo_dependency() -> Iterator[Repo]:
        conn = init_db(cfg.db_path)
        try:
            yield Repo(conn)
        finally:
            conn.close()

    @contextmanager
    def _repo_ctx() -> Iterator[Repo]:
        conn = init_db(cfg.db_path)
        try:
            yield Repo(conn)
        finally:
            conn.close()

    def handle(exc: Exception) -> HTTPException:
        if isinstance(exc, ConflictError):
            return HTTPException(status_code=409, detail=str(exc))
        if isinstance(exc, (NotFoundError, KeyError)):
            return HTTPException(status_code=404, detail=str(exc).strip("'\""))
        if isinstance(exc, (ValueError, ConfigError, json.JSONDecodeError)):
            return HTTPException(status_code=400, detail=str(exc))
        if isinstance(exc, sqlite3.IntegrityError):
            return HTTPException(status_code=409, detail=str(exc))
        return HTTPException(status_code=500, detail=str(exc))

    # ── 健康检查 ──
    @app.get("/api/health")
    def health(repo: Repo = Depends(repo_dependency)) -> dict[str, Any]:
        counts = repo.overview()
        return {
            "status": "ok",
            "schema_version": 1,
            "data_dir": str(cfg.data_dir),
            "images": counts["images"],
            "tasks": counts["tasks"],
            "providers": {"detector": None, "sam": None, "gpu": None},
        }

    # ── 类别 ──
    @app.get("/api/classes")
    def list_classes(active_only: bool = False, repo: Repo = Depends(repo_dependency)) -> list[dict[str, Any]]:
        return repo.list_classes(active_only=active_only)

    @app.post("/api/classes", status_code=201)
    def create_class(payload: ClassCreate, repo: Repo = Depends(repo_dependency)) -> dict[str, Any]:
        try:
            if repo.get_class(payload.code) is not None:
                raise sqlite3.IntegrityError(f"类别已存在: {payload.code}")
            return repo.add_class(payload.code, payload.name_zh, payload.name_en, color=payload.color,
                                  parent_code=payload.parent_code, is_crack=payload.is_crack,
                                  order_index=payload.order_index, actor="api")
        except Exception as exc:  # noqa: BLE001
            raise handle(exc) from exc

    # ── 导入 ──
    @app.post("/api/ingest/batches", status_code=202)
    def create_batch(payload: IngestRequest, repo: Repo = Depends(repo_dependency)) -> dict[str, Any]:
        if not payload.source_dir:
            raise HTTPException(status_code=400, detail="M1 仅支持 source_dir（服务器目录）导入；上传请用 /api/ingest/upload")
        try:
            report = import_path(cfg, repo, payload.source_dir, kind=payload.kind, fps=payload.fps,
                                 max_frames=payload.max_frames, tile=payload.tile, note=payload.note,
                                 actor="api", limit=payload.limit)
        except Exception as exc:  # noqa: BLE001
            raise handle(exc) from exc
        return report.as_dict()

    @app.post("/api/ingest/upload", status_code=202)
    async def upload(files: list[UploadFile], kind: str = "photo",
                     repo: Repo = Depends(repo_dependency)) -> dict[str, Any]:
        inbox = cfg.data_dir / UPLOAD_INBOX_NAME
        inbox.mkdir(parents=True, exist_ok=True)
        saved = 0
        for upload_file in files:
            name = Path(upload_file.filename or "upload.bin").name  # 防路径穿越：只取文件名
            target = inbox / name
            data = await upload_file.read()
            if len(data) > cfg.ingest.max_file_mb * 1048576:
                raise HTTPException(status_code=413, detail=f"{name} 超过单文件上限 {cfg.ingest.max_file_mb}MB")
            target.write_bytes(data)
            saved += 1
        try:
            report = import_path(cfg, repo, inbox, kind=kind, note="upload", actor="api")
        except Exception as exc:  # noqa: BLE001
            raise handle(exc) from exc
        return {**report.as_dict(), "uploaded_files": saved}

    @app.get("/api/ingest/batches")
    def list_batches(limit: int = Query(50, ge=1, le=500), cursor: int | None = None,
                     repo: Repo = Depends(repo_dependency)) -> dict[str, Any]:
        rows = repo.list_batches(limit=limit, cursor=cursor)
        return {"items": rows, "next_cursor": rows[-1]["id"] if len(rows) == limit else None}

    @app.get("/api/ingest/batches/{batch_id}")
    def get_batch(batch_id: int, repo: Repo = Depends(repo_dependency)) -> dict[str, Any]:
        batch = repo.get_batch(batch_id)
        if batch is None:
            raise HTTPException(status_code=404, detail=f"批次 {batch_id} 不存在")
        batch["outcome_counts"] = repo.batch_outcome_counts(batch_id)
        batch["failures"] = repo.batch_failures(batch_id)
        return batch

    # ── 影像 ──
    @app.get("/api/images")
    def list_images(batch_id: int | None = None, source_kind: str | None = None,
                    limit: int = Query(50, ge=1, le=500), cursor: int | None = None,
                    repo: Repo = Depends(repo_dependency)) -> dict[str, Any]:
        rows = repo.list_images(batch_id=batch_id, source_kind=source_kind, limit=limit, cursor=cursor)
        return {"items": rows, "next_cursor": rows[-1]["id"] if len(rows) == limit else None}

    @app.get("/api/images/{image_id}/file")
    def image_file(image_id: int, thumb: bool = False, repo: Repo = Depends(repo_dependency)) -> FileResponse:
        image = repo.get_image(image_id)
        if image is None:
            raise HTTPException(status_code=404, detail=f"影像 {image_id} 不存在")
        if thumb:
            thumb_path = cfg.thumbs_dir / f"{image_id}.webp"
            if thumb_path.exists():
                return FileResponse(thumb_path, media_type="image/webp")
        path = cfg.abs_data_path(image["path"])
        if not path.exists():
            raise HTTPException(status_code=410, detail="影像文件已丢失（可从原始素材重新导入）")
        return FileResponse(path)

    # ── 任务 ──
    @app.get("/api/tasks")
    def list_tasks(status: str | None = None, batch_id: int | None = None, class_code: str | None = None,
                   strategy: str = "fifo", limit: int = Query(50, ge=1, le=500), cursor: int | None = None,
                   repo: Repo = Depends(repo_dependency)) -> dict[str, Any]:
        rows = repo.list_tasks(status=status, batch_id=batch_id, class_code=class_code, strategy=strategy,
                               limit=limit, cursor=cursor)
        return {"items": rows, "next_cursor": rows[-1]["id"] if len(rows) == limit else None}

    @app.post("/api/tasks/lease")
    def lease_tasks(payload: LeaseRequest, repo: Repo = Depends(repo_dependency)) -> dict[str, Any]:
        leased = repo.lease_tasks(count=payload.count, status=payload.status, assignee=payload.assignee,
                                  lease_seconds=payload.lease_seconds or cfg.lease_seconds)
        return {"leased": leased}

    @app.get("/api/tasks/{task_id}")
    def task_detail(task_id: int, repo: Repo = Depends(repo_dependency)) -> dict[str, Any]:
        detail = repo.get_task_detail(task_id)
        if detail is None:
            raise HTTPException(status_code=404, detail=f"任务 {task_id} 不存在")
        detail["annotations"] = [serialize_annotation(row) for row in detail["annotations"]]
        return detail

    @app.put("/api/tasks/{task_id}/annotations")
    def put_annotations(task_id: int, payload: AnnotationsPut,
                        repo: Repo = Depends(repo_dependency)) -> dict[str, Any]:
        try:
            result = repo.replace_annotations(task_id, [item.model_dump() for item in payload.annotations],
                                              actor="api")
        except Exception as exc:  # noqa: BLE001
            raise handle(exc) from exc
        return {"annotations": [serialize_annotation(row) for row in result["annotations"]],
                "changed": result["changed"], "added": result["added"],
                "updated": result["updated"], "deleted": result["deleted"]}

    @app.post("/api/tasks/{task_id}/adopt-candidates")
    def adopt_candidates(task_id: int, repo: Repo = Depends(repo_dependency)) -> dict[str, Any]:
        return {"adopted": repo.adopt_model_candidates(task_id, actor="api")}

    @app.post("/api/tasks/{task_id}/submit")
    def submit_task(task_id: int, repo: Repo = Depends(repo_dependency)) -> dict[str, Any]:
        task = repo.submit_task(task_id, actor="api")
        if task is None:
            raise HTTPException(status_code=404, detail=f"任务 {task_id} 不存在")
        return task

    @app.post("/api/tasks/{task_id}/review")
    def review_task(task_id: int, payload: ReviewRequest,
                    repo: Repo = Depends(repo_dependency)) -> dict[str, Any]:
        try:
            review = repo.add_review(task_id, decision=payload.decision, reason_code=payload.reason_code,
                                     note=payload.note, reviewer=payload.reviewer or "api")
        except Exception as exc:  # noqa: BLE001
            raise handle(exc) from exc
        return {"task": repo.get_task(task_id), "review": review}

    # ── 预标注（M2：模型辅助标注）──

    def _prelabel_deps(payload: "PrelabelBatchRequest", repo: Repo) -> tuple[object | None, object | None]:
        """仅当测试注入了工厂时提前构造检测器/掩膜器；否则交回 service 自行解析权重。"""
        from ..prelabel.service import resolve_weights  # noqa: PLC0415

        detector = None
        masker = None
        if detector_factory is not None:
            weights = payload.model or resolve_weights(cfg, repo, payload.model)
            detector = make_detector(weights, conf=payload.conf, iou=payload.iou)
        if masker_factory is not None:
            masker = make_masker()
        return detector, masker

    @app.post("/api/tasks/{task_id}/prelabel")
    def prelabel_task(task_id: int, payload: PrelabelBatchRequest | None = None,
                      repo: Repo = Depends(repo_dependency)) -> dict[str, Any]:
        options = payload or PrelabelBatchRequest(limit=1)
        detector, masker = _prelabel_deps(options, repo)
        try:
            report = prelabel_tasks(cfg, repo, limit=1, task_ids=[task_id],
                                    model=options.model, conf=options.conf, iou=options.iou,
                                    sam=options.sam, sam_limit=options.sam_limit, actor="api",
                                    detector=detector, masker=masker)
        except DetectorUnavailable as exc:
            raise HTTPException(status_code=501, detail=str(exc)) from exc
        return report.as_dict() | {"task_id": task_id}

    @app.post("/api/prelabel/batches")
    def prelabel_batch(payload: PrelabelBatchRequest, repo: Repo = Depends(repo_dependency)) -> dict[str, Any]:
        detector, masker = _prelabel_deps(payload, repo)
        try:
            report = prelabel_tasks(cfg, repo, limit=payload.limit, status=payload.status,
                                    model=payload.model, conf=payload.conf, iou=payload.iou,
                                    sam=payload.sam, sam_limit=payload.sam_limit, actor="api",
                                    detector=detector, masker=masker)
        except DetectorUnavailable as exc:
            raise HTTPException(status_code=501, detail=str(exc)) from exc
        run = repo.get_run(report.run_id) if report.run_id else None
        return {"run": run, "requested": report.requested, "report": report.as_dict()}

    @app.get("/api/prelabel/metrics")
    def prelabel_quality(repo: Repo = Depends(repo_dependency)) -> dict[str, Any]:
        return prelabel_metrics(repo)

    # ── 标注删除 / 掩膜辅助 ──
    @app.delete("/api/annotations/{annotation_id}")
    def delete_annotation(annotation_id: int, repo: Repo = Depends(repo_dependency)) -> dict[str, Any]:
        deleted = repo.delete_annotation(annotation_id, actor="api")
        if not deleted:
            raise HTTPException(status_code=404, detail=f"标注 {annotation_id} 不存在或已删除")
        return {"deleted": True, "id": annotation_id}

    @app.post("/api/annotations/{annotation_id}/mask")
    def create_mask(annotation_id: int, repo: Repo = Depends(repo_dependency)) -> dict[str, Any]:
        try:
            masker = make_masker() if masker_factory is not None else None
            return attach_mask_for_annotation(cfg, repo, annotation_id, masker=masker)
        except DetectorUnavailable as exc:
            raise HTTPException(status_code=501, detail=str(exc)) from exc
        except (KeyError, ValueError) as exc:
            raise handle(exc) from exc

    @app.get("/api/annotations/{annotation_id}/mask")
    def get_mask(annotation_id: int, repo: Repo = Depends(repo_dependency)) -> FileResponse:
        row = repo.conn.execute("SELECT mask_path FROM annotations WHERE id = ?", (annotation_id,)).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail=f"标注 {annotation_id} 不存在")
        mask_path = row["mask_path"]
        if not mask_path:
            # 便捷回退：用「框标注 id」查询时，返回同任务同类最新掩膜（标注台流程用得到）
            fallback = repo.conn.execute(
                """SELECT a.mask_path FROM annotations a
                   JOIN annotations src ON src.id = ?
                   WHERE a.kind = 'mask' AND a.deleted_at IS NULL AND a.task_id = src.task_id
                     AND a.class_code = src.class_code AND a.mask_path IS NOT NULL
                   ORDER BY a.id DESC LIMIT 1""", (annotation_id,)).fetchone()
            mask_path = fallback["mask_path"] if fallback is not None else None
        if not mask_path:
            raise HTTPException(status_code=404, detail=f"标注 {annotation_id} 没有掩膜")
        path = cfg.abs_data_path(mask_path)
        if not path.exists():
            raise HTTPException(status_code=410, detail="掩膜文件已丢失")
        return FileResponse(path, media_type="image/png")

    # ── 模型注册表（M2：登记预标注权重）──
    @app.get("/api/models")
    def list_models(status: str | None = None, task: str | None = None,
                    repo: Repo = Depends(repo_dependency)) -> list[dict[str, Any]]:
        return repo.list_model_versions(task=task, status=status)

    # ── 数据集 ──
    @app.get("/api/datasets")
    def list_datasets(repo: Repo = Depends(repo_dependency)) -> list[dict[str, Any]]:
        return [datasets_mod.dataset_summary(cfg, repo, row) for row in repo.list_datasets()]

    @app.post("/api/datasets", status_code=201)
    def create_dataset(payload: DatasetDraftRequest, repo: Repo = Depends(repo_dependency)) -> dict[str, Any]:
        try:
            if repo.get_dataset(name=payload.name) is not None:
                raise sqlite3.IntegrityError(f"数据集名已存在: {payload.name}")
            draft = datasets_mod.create_draft(cfg, repo, payload.name,
                                              payload.filter or {"review_status": "approved"},
                                              payload.split or {})
        except Exception as exc:  # noqa: BLE001
            raise handle(exc) from exc
        return datasets_mod.dataset_summary(cfg, repo, draft) | {"stats": draft.get("stats")}

    @app.post("/api/datasets/{dataset_id}/freeze")
    def freeze_dataset(dataset_id: int, repo: Repo = Depends(repo_dependency)) -> dict[str, Any]:
        try:
            frozen = datasets_mod.freeze_dataset(cfg, repo, dataset_id=dataset_id)
        except Exception as exc:  # noqa: BLE001
            raise handle(exc) from exc
        return datasets_mod.dataset_summary(cfg, repo, frozen) | {"exports": frozen.get("exports")}

    @app.post("/api/datasets/{dataset_id}/export")
    def export_dataset(dataset_id: int, payload: DatasetExportRequest,
                       repo: Repo = Depends(repo_dependency)) -> dict[str, Any]:
        dataset = repo.get_dataset(dataset_id=dataset_id)
        if dataset is None:
            raise HTTPException(status_code=404, detail=f"数据集 {dataset_id} 不存在")
        if dataset["status"] != "frozen":
            raise HTTPException(status_code=409, detail="只有 frozen 数据集可导出（见 ADR-0005）")
        try:
            rows = datasets_mod.fetch_dataset_rows(repo, repo.dataset_items(dataset_id))
            classes = repo.list_classes()
            root = cfg.datasets_dir / dataset["name"]
            from ..core.formats import export_coco, export_labelme, export_yolo  # 局部导入避免循环

            abs_rows = [{**row, "abs_path": str(cfg.abs_data_path(row["image"]["path"]))} for row in rows]
            results: dict[str, Any] = {}
            for fmt in payload.formats:
                if fmt == "yolo":
                    results["yolo"] = export_yolo(abs_rows, classes, root, copy_images=payload.copy_images)
                elif fmt == "coco":
                    results["coco"] = export_coco(abs_rows, classes, root / "coco.json", copy_images=False)
                elif fmt == "labelme":
                    results["labelme"] = export_labelme(abs_rows, classes, root / "labelme")
                else:
                    raise HTTPException(status_code=400, detail=f"未知导出格式: {fmt}")
        except HTTPException:
            raise
        except Exception as exc:  # noqa: BLE001
            raise handle(exc) from exc
        return {"dataset": dataset["name"], "root": str(root), "exports": results}

    # ── 运行记录与统计 ──
    @app.get("/api/runs")
    def list_runs(kind: str | None = None, status: str | None = None,
                  limit: int = Query(50, ge=1, le=500), repo: Repo = Depends(repo_dependency)) -> list[dict[str, Any]]:
        return repo.list_runs(kind=kind, status=status, limit=limit)

    @app.get("/api/stats/overview")
    def stats_overview(repo: Repo = Depends(repo_dependency)) -> dict[str, Any]:
        return repo.overview()

    @app.get("/api/stats/export")
    def stats_export(format: str = "csv", repo: Repo = Depends(repo_dependency)) -> Any:
        rows = repo.conn.execute(
            """SELECT i.id AS image_id, i.captured_at, i.gps_lat, i.gps_lon, a.class_code,
                      a.bbox_x1, a.bbox_y1, a.bbox_x2, a.bbox_y2, a.source, a.score
               FROM annotations a JOIN images i ON i.id = a.image_id
               WHERE a.deleted_at IS NULL ORDER BY i.id, a.id""").fetchall()
        if format == "json":
            return JSONResponse([dict(row) for row in rows])
        header = "image_id,captured_at,gps_lat,gps_lon,class_code,x1,y1,x2,y2,source,score\n"
        lines = [header]
        for row in rows:
            lines.append(",".join("" if row[key] is None else str(row[key]) for key in row.keys()) + "\n")
        return PlainTextResponse("".join(lines), media_type="text/csv")

    # ── 未实现（M2/M3）的端点：明确 501，避免前端误判 ──
    @app.post("/api/{rest:path}")
    def not_implemented(rest: str) -> None:  # pragma: no cover - 由更具体路由优先匹配
        raise HTTPException(status_code=501, detail=f"/api/{rest} 属于 M2/M3（预标注/训练），M1 未实现")

    # ── 静态标注台 ──
    index_file = STATIC_DIR / "index.html"
    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", response_class=HTMLResponse)
    def index() -> Any:
        if not index_file.exists():
            return HTMLResponse("<h1>road-inspect</h1><p>标注台文件缺失（src/rdinspect/api/static/index.html）</p>",
                                status_code=500)
        return FileResponse(index_file, media_type="text/html")

    @app.exception_handler(HTTPException)
    async def http_error(_: Request, exc: HTTPException) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content={"code": f"http_{exc.status_code}",
                                                                  "message": str(exc.detail)})

    app.state.config = cfg
    return app


#: uvicorn rdinspect.api.app:app 的默认入口。
#  惰性创建：模块导入阶段不读取配置、不建目录（否则默认配置不可写时连 CLI serve 都无法启动）。
_APP: FastAPI | None = None


def get_app() -> FastAPI:
    """按默认配置（环境变量 ROAD_INSPECT_CONFIG > configs/default.yaml）构建并缓存应用。"""
    global _APP
    if _APP is None:
        _APP = create_app()
    return _APP


class _LazyApp:
    """ASGI 外观：uvicorn 传入的每个连接在首次访问时才真正构建应用。"""

    async def __call__(self, scope, receive, send):  # type: ignore[no-untyped-def]
        await get_app()(scope, receive, send)


app = _LazyApp()
