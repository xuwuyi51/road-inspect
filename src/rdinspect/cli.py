"""命令行入口：rdinspect serve|import|tasks|dataset|export|stats|check。

设计与契约见 docs/06-api-spec.md §5；退出码：0 成功 / 2 参数错误 / 3 依赖缺失 / 4 运行失败 / 5 门禁未通过。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import Config, ConfigError, load_config
from .core import datasets as datasets_mod
from .core.ingest import import_path
from .prelabel.detector import DetectorUnavailable, ml_available
from .prelabel.metrics import prelabel_metrics
from .prelabel.service import DEFAULT_WEIGHTS, prelabel_tasks
from .storage.db import init_db
from .storage.files import rebuild_thumbnails
from .storage.repo import Repo

EXIT_OK, EXIT_ARGS, EXIT_DEP, EXIT_RUN = 0, 2, 3, 4


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rdinspect", description="轻量级道路灾害巡查（M1：采集与标注闭环）")
    parser.add_argument("--config", help="配置文件路径（默认 configs/default.yaml）")
    parser.add_argument("--data-dir", help="数据根目录（覆盖配置）")
    parser.add_argument("--json", action="store_true", help="以 JSON 输出结果")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="启动工作站 Web 服务（标注台 + API）")
    serve.add_argument("--host")
    serve.add_argument("--port", type=int)
    serve.add_argument("--reload", action="store_true")

    imp = sub.add_parser("import", help="导入照片/视频/航拍影像")
    imp.add_argument("--kind", choices=["photo", "video", "aerial", "external"], default=None)
    imp.add_argument("--input", required=True, help="文件或目录（必须在 allowed_roots 内）")
    imp.add_argument("--fps", type=float, help="视频抽帧频率（默认取配置）")
    imp.add_argument("--max-frames", type=int)
    imp.add_argument("--tile", action="store_true", help="强制切片（航拍）")
    imp.add_argument("--no-tile", action="store_true", help="禁用切片")
    imp.add_argument("--limit", type=int, help="最多处理多少个文件（用于小样本试跑）")
    imp.add_argument("--note")

    tasks = sub.add_parser("tasks", help="任务队列操作")
    tasks.add_argument("--status", default=None)
    tasks.add_argument("--limit", type=int, default=20)
    tasks.add_argument("--lease", type=int, help="领取 N 个任务")
    tasks.add_argument("--assignee", default="cli")

    dataset = sub.add_parser("dataset", help="数据集：草稿/冻结/查看")
    dataset.add_argument("action", choices=["list", "create", "freeze", "show"])
    dataset.add_argument("--name")
    dataset.add_argument("--review-status", default="approved", choices=["approved", "annotated", "any"])
    dataset.add_argument("--class", dest="classes", action="append", help="按类别过滤（可重复）")
    dataset.add_argument("--batch", dest="batches", action="append", type=int)
    dataset.add_argument("--split-train", type=float, default=0.7)
    dataset.add_argument("--split-val", type=float, default=0.15)
    dataset.add_argument("--split-test", type=float, default=0.15)
    dataset.add_argument("--seed", type=int, default=42)

    export = sub.add_parser("export", help="导出已冻结数据集（YOLO/COCO/LabelMe）")
    export.add_argument("--name", required=True)
    export.add_argument("--formats", default="yolo,coco,labelme")
    export.add_argument("--no-copy-images", action="store_true")

    prelabel = sub.add_parser("prelabel", help="模型辅助标注：批量生成候选框（可含 SAM 掩膜）")
    prelabel.add_argument("--limit", type=int, default=50)
    prelabel.add_argument("--status", default="pending")
    prelabel.add_argument("--task", dest="tasks", action="append", type=int, help="只处理指定任务（可重复）")
    prelabel.add_argument("--model", help=f"权重路径或名称（默认 {DEFAULT_WEIGHTS} 或现役 production 模型）")
    prelabel.add_argument("--conf", type=float, help="置信度阈值（默认取配置）")
    prelabel.add_argument("--iou", type=float, help="NMS IoU（默认取配置）")
    prelabel.add_argument("--device", help="auto|cpu|cuda")
    prelabel.add_argument("--sam", action="store_true", help="为裂缝候选生成 SAM 掩膜")
    prelabel.add_argument("--sam-limit", type=int, default=2, help="每张图最多生成几个掩膜")

    quality = sub.add_parser("prelabel-metrics", help="预标注质量看板：采纳率与模型-人工一致性")
    quality.add_argument("--json", action="store_true")

    models = sub.add_parser("models", help="模型注册表（M2 预标注权重）")
    models.add_argument("--task", default=None)
    models.add_argument("--status", default=None)

    classes = sub.add_parser("classes", help="类别注册表：查看 / 新增 / 停用（扩展点，无需改代码）")
    classes.add_argument("action", choices=["list", "add", "disable", "enable"])
    classes.add_argument("--code", help="类别 code，如 alligator_crack")
    classes.add_argument("--zh", help="中文名")
    classes.add_argument("--en", help="英文名")
    classes.add_argument("--color", default="#e6194b")
    classes.add_argument("--order", type=int, default=100, help="YOLO 类别顺序（越小越前）")
    classes.add_argument("--is-crack", action="store_true")
    classes.add_argument("--all", action="store_true", help="list 时包含已停用类别")

    stats = sub.add_parser("stats", help="统计概览")
    stats.add_argument("--export", choices=["csv", "json"], help="导出标注明细")

    sub.add_parser("thumbs", help="重建缩略图（可随时执行，缩略图可再生）")
    sub.add_parser("check", help="自检：文档/DDL/OpenAPI/样例 + 数据库连通性")
    return parser


def _load(args: argparse.Namespace) -> Config:
    config = load_config(args.config, data_dir=args.data_dir)
    config.ensure_dirs()
    return config


def _print(args: argparse.Namespace, payload: object, *, text: str | None = None) -> None:
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(text if text is not None else json.dumps(payload, ensure_ascii=False, indent=2))


def _cmd_import(args: argparse.Namespace, config: Config) -> int:
    conn = init_db(config.db_path)
    repo = Repo(conn)
    try:
        tile = True if args.tile else False if args.no_tile else None
        report = import_path(config, repo, args.input, kind=args.kind, fps=args.fps,
                             max_frames=args.max_frames, tile=tile, note=args.note,
                             actor="cli", limit=args.limit)
    finally:
        conn.close()
    payload = report.as_dict()
    text = (f"批次 #{report.batch_id}（{report.kind}）：新增 {report.added}"
            f"（切片 {report.tiles}）｜SHA 重复 {report.dup_sha}｜近似重复 {report.dup_phash}"
            f"｜跳过 {report.skipped_size + report.skipped_type}｜失败 {report.error}")
    _print(args, payload, text=text)
    return EXIT_OK if report.added > 0 or report.dup_sha > 0 else EXIT_RUN


def _cmd_tasks(args: argparse.Namespace, config: Config) -> int:
    conn = init_db(config.db_path)
    repo = Repo(conn)
    try:
        if args.lease:
            leased = repo.lease_tasks(count=args.lease, assignee=args.assignee,
                                      lease_seconds=config.lease_seconds)
            _print(args, {"leased": leased}, text=f"领取 {len(leased)} 个任务")
            return EXIT_OK
        rows = repo.list_tasks(status=args.status, limit=args.limit)
        _print(args, {"items": rows}, text="\n".join(
            f"#{row['id']} image={row['image_id']} status={row['status']} priority={row['priority']}"
            for row in rows) or "（无任务）")
    finally:
        conn.close()
    return EXIT_OK


def _cmd_dataset(args: argparse.Namespace, config: Config) -> int:
    conn = init_db(config.db_path)
    repo = Repo(conn)
    try:
        if args.action == "list":
            rows = [datasets_mod.dataset_summary(config, repo, row) for row in repo.list_datasets()]
            _print(args, rows, text="\n".join(
                f"{row['name']} [{row['status']}] images={row['stats']['images'] if row['stats'] else 0} "
                f"hash={row['manifest_hash'] or '-'}" for row in rows) or "（无数据集）")
            return EXIT_OK
        if args.action == "show":
            dataset = repo.get_dataset(name=args.name) if args.name else None
            if dataset is None:
                print(f"数据集不存在: {args.name}", file=sys.stderr)
                return EXIT_ARGS
            _print(args, datasets_mod.dataset_summary(config, repo, dataset))
            return EXIT_OK
        if args.action == "create":
            if not args.name:
                print("create 需要 --name ds-xxx", file=sys.stderr)
                return EXIT_ARGS
            filters = {"review_status": args.review_status}
            if args.classes:
                filters["class_codes"] = args.classes
            if args.batches:
                filters["batch_ids"] = args.batches
            split = {"train": args.split_train, "val": args.split_val, "test": args.split_test,
                     "seed": args.seed, "group_by": "gps_grid"}
            draft = datasets_mod.create_draft(config, repo, args.name, filters, split)
            _print(args, datasets_mod.dataset_summary(config, repo, draft) | {"stats": draft.get("stats")})
            return EXIT_OK
        # freeze
        frozen = datasets_mod.freeze_dataset(config, repo, name=args.name)
        summary = datasets_mod.dataset_summary(config, repo, frozen)
        _print(args, summary | {"exports": frozen.get("exports")},
               text=f"已冻结 {summary['name']}：{summary['stats']['images']} 图 / "
                    f"{summary['stats']['labels']} 标注 / hash={summary['manifest_hash'][:16]}…")
        return EXIT_OK
    finally:
        conn.close()


def _cmd_export(args: argparse.Namespace, config: Config) -> int:
    conn = init_db(config.db_path)
    repo = Repo(conn)
    try:
        dataset = repo.get_dataset(name=args.name)
        if dataset is None:
            print(f"数据集不存在: {args.name}", file=sys.stderr)
            return EXIT_ARGS
        if dataset["status"] != "frozen":
            print(f"数据集 {args.name} 未冻结（status={dataset['status']}）；先执行 dataset freeze", file=sys.stderr)
            return EXIT_RUN
        rows = datasets_mod.fetch_dataset_rows(repo, repo.dataset_items(dataset["id"]))
        classes = repo.list_classes()
        from .core.formats import export_coco, export_labelme, export_yolo

        root = config.datasets_dir / dataset["name"]
        abs_rows = [{**row, "abs_path": str(config.abs_data_path(row["image"]["path"]))} for row in rows]
        results = {}
        for fmt in [item.strip() for item in args.formats.split(",") if item.strip()]:
            if fmt == "yolo":
                results["yolo"] = export_yolo(abs_rows, classes, root, copy_images=not args.no_copy_images)
            elif fmt == "coco":
                results["coco"] = export_coco(abs_rows, classes, root / "coco.json", copy_images=False)
            elif fmt == "labelme":
                results["labelme"] = export_labelme(abs_rows, classes, root / "labelme")
            else:
                print(f"未知格式: {fmt}", file=sys.stderr)
                return EXIT_ARGS
        _print(args, {"dataset": dataset["name"], "root": str(root), "exports": results})
        return EXIT_OK
    finally:
        conn.close()


def _cmd_prelabel(args: argparse.Namespace, config: Config) -> int:
    if not ml_available():
        print("未安装 ML 依赖：请执行 .venv/bin/pip install -e '.[ml]'（详见 docs/08-deployment.md）", file=sys.stderr)
        return EXIT_DEP
    conn = init_db(config.db_path)
    repo = Repo(conn)
    try:
        if args.device:
            config.prelabel.device = args.device
        report = prelabel_tasks(config, repo, limit=args.limit, status=args.status,
                                task_ids=args.tasks, model=args.model, conf=args.conf, iou=args.iou,
                                sam=args.sam or None, sam_limit=args.sam_limit, actor="cli")
    except DetectorUnavailable as exc:
        print(f"预标注不可用: {exc}", file=sys.stderr)
        return EXIT_DEP
    finally:
        conn.close()
    payload = report.as_dict()
    text = (f"run #{report.run_id}｜任务 {report.processed}/{report.requested}｜候选 {report.candidates}"
            f"｜掩膜 {report.masks}｜切片 {report.tiles}｜未映射类别 {report.unmapped}"
            f"｜跳过 {report.skipped}｜错误 {len(report.errors)}")
    _print(args, payload, text=text)
    return EXIT_OK if report.processed > 0 or report.requested == 0 else EXIT_RUN


def _cmd_prelabel_metrics(args: argparse.Namespace, config: Config) -> int:
    conn = init_db(config.db_path)
    repo = Repo(conn)
    try:
        data = prelabel_metrics(repo)
    finally:
        conn.close()
    if args.json:
        print(json.dumps(data, ensure_ascii=False, indent=2))
    else:
        rate = data["adoption_rate"]
        iou_mean = data["model_human_iou_mean"]
        print(f"已预标注任务 {data['tasks_prelabeled']}｜候选 {data['candidates_total']}｜"
              f"已采纳 {data['adopted']}｜已忽略 {data['ignored']}｜"
              f"采纳率 {'—' if rate is None else f'{rate:.1%}'}｜"
              f"模型-人工 IoU {'—' if iou_mean is None else f'{iou_mean:.2f}'}"
              f"（匹配率 {'—' if data['model_human_match_rate'] is None else f'{data['model_human_match_rate']:.0%}'}）")
        if data["prelabel_states"]:
            print(f"预标注状态分布：{data['prelabel_states']}")
    return EXIT_OK


def _cmd_models(args: argparse.Namespace, config: Config) -> int:
    conn = init_db(config.db_path)
    repo = Repo(conn)
    try:
        rows = repo.list_model_versions(task=args.task, status=args.status)
    finally:
        conn.close()
    _print(args, rows, text="\n".join(
        f"{row['name']}:{row['version']} [{row['status']}] {row['task']} weights={row['weights_path']}"
        for row in rows) or "（模型注册表为空）")
    return EXIT_OK


def _cmd_classes(args: argparse.Namespace, config: Config) -> int:
    conn = init_db(config.db_path)
    repo = Repo(conn)
    try:
        if args.action == "list":
            rows = repo.list_classes(active_only=not args.all)
            _print(args, rows, text="\n".join(
                f"{'✔' if row['active'] else '✘'} {row['order_index']:>3} {row['code']:<22} {row['name_zh']}"
                for row in rows))
            return EXIT_OK
        if args.action == "add":
            if not (args.code and args.zh and args.en):
                print("add 需要 --code/--zh/--en", file=sys.stderr)
                return EXIT_ARGS
            if repo.get_class(args.code) is not None:
                print(f"类别已存在: {args.code}", file=sys.stderr)
                return EXIT_ARGS
            created = repo.add_class(args.code, args.zh, args.en, color=args.color,
                                     is_crack=args.is_crack, order_index=args.order, actor="cli")
            _print(args, created, text=f"已新增类别 {created['code']}（order={created['order_index']}）；"
                                       f"注意：类别顺序变化后需新建数据集版本（见 ADR-0005）")
            return EXIT_OK
        if not args.code:
            print(f"{args.action} 需要 --code", file=sys.stderr)
            return EXIT_ARGS
        active = args.action == "enable"
        ok = repo.set_class_active(args.code, active, actor="cli")
        if not ok:
            print(f"类别不存在: {args.code}", file=sys.stderr)
            return EXIT_ARGS
        print(f"类别 {args.code} 已{'启用' if active else '停用'}（停用不删除历史标注，见 docs/03-data-model.md §8）")
        return EXIT_OK
    finally:
        conn.close()


def _cmd_stats(args: argparse.Namespace, config: Config) -> int:
    conn = init_db(config.db_path)
    repo = Repo(conn)
    try:
        if args.export:
            rows = repo.conn.execute(
                """SELECT i.id AS image_id, i.captured_at, i.gps_lat, i.gps_lon, a.class_code,
                          a.bbox_x1, a.bbox_y1, a.bbox_x2, a.bbox_y2, a.source, a.score
                   FROM annotations a JOIN images i ON i.id = a.image_id
                   WHERE a.deleted_at IS NULL ORDER BY i.id, a.id""").fetchall()
            if args.export == "json":
                _print(args, [dict(row) for row in rows])
            else:
                header = "image_id,captured_at,gps_lat,gps_lon,class_code,x1,y1,x2,y2,source,score"
                print(header)
                for row in rows:
                    print(",".join("" if row[key] is None else str(row[key]) for key in row.keys()))
            return EXIT_OK
        overview = repo.overview()
        _print(args, overview)
        return EXIT_OK
    finally:
        conn.close()


def _cmd_thumbs(args: argparse.Namespace, config: Config) -> int:
    conn = init_db(config.db_path)
    repo = Repo(conn)
    try:
        rows = [(row["id"], row["path"]) for row in repo.conn.execute("SELECT id, path FROM images")]
        done = rebuild_thumbnails(config, rows)
        _print(args, {"images": len(rows), "rebuilt": done},
               text=f"缩略图重建：{done}/{len(rows)}")
        return EXIT_OK
    finally:
        conn.close()


def _cmd_check(args: argparse.Namespace, config: Config) -> int:
    """自检：文档自检脚本 + 数据库可建库/迁移。"""
    import subprocess

    script = Path(__file__).resolve().parents[2] / "scripts" / "check_docs.py"
    result = subprocess.run([sys.executable, str(script)], capture_output=True, text=True)
    print(result.stdout.strip())
    if result.returncode != 0:
        print(result.stderr.strip(), file=sys.stderr)
        return EXIT_RUN
    conn = init_db(config.db_path)
    try:
        repo = Repo(conn)
        overview = repo.overview()
        print(f"数据库 OK：{config.db_path}｜影像 {overview['images']}｜任务 {overview['tasks']}")
    finally:
        conn.close()
    return EXIT_OK


def _cmd_serve(args: argparse.Namespace, config: Config) -> int:
    try:
        import uvicorn
    except ImportError:
        print("缺少 uvicorn：请先 pip install -e . 或 pip install uvicorn", file=sys.stderr)
        return EXIT_DEP
    from .api.app import create_app

    host = args.host or config.host
    port = args.port or config.port
    print(f"road-inspect 工作站：http://{host}:{port}（标注台在 /，API 在 /api）")
    uvicorn.run(create_app(config), host=host, port=port, reload=args.reload, log_level="info")
    return EXIT_OK


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        config = _load(args)
    except ConfigError as exc:
        print(f"配置错误: {exc}", file=sys.stderr)
        return EXIT_ARGS
    handlers = {
        "serve": _cmd_serve, "import": _cmd_import, "tasks": _cmd_tasks,
        "dataset": _cmd_dataset, "export": _cmd_export, "stats": _cmd_stats,
        "thumbs": _cmd_thumbs, "check": _cmd_check, "prelabel": _cmd_prelabel,
        "prelabel-metrics": _cmd_prelabel_metrics, "models": _cmd_models, "classes": _cmd_classes,
    }
    handler = handlers[args.command]
    if args.command == "dataset" and args.action == "freeze" and not args.name:
        print("freeze 需要 --name", file=sys.stderr)
        return EXIT_ARGS
    if args.command == "dataset" and args.action == "create" and not args.name:
        print("create 需要 --name", file=sys.stderr)
        return EXIT_ARGS
    try:
        return handler(args, config)
    except (ConfigError, ValueError) as exc:
        print(f"{args.command} 失败: {exc}", file=sys.stderr)
        return EXIT_RUN
    except KeyError as exc:
        print(f"{args.command} 失败: 找不到 {exc}", file=sys.stderr)
        return EXIT_RUN


if __name__ == "__main__":
    sys.exit(main())
