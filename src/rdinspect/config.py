"""配置加载与路径安全策略。

配置来源优先级：显式 --config > ROAD_INSPECT_CONFIG 环境变量 > configs/default.yaml。
所有对用户输入路径的访问都必须经过 :func:`Config.resolve_source` 的白名单校验。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}


class ConfigError(RuntimeError):
    """配置不合法（缺失字段、路径越界、类型错误）。"""


@dataclass
class DedupConfig:
    phash_hamming_threshold: int = 6
    keep_near_duplicates: bool = True


@dataclass
class VideoConfig:
    default_fps: float = 2.0
    max_frames_per_video: int = 2000


@dataclass
class TileConfig:
    enabled: bool = False
    size: int = 1024
    overlap: float = 0.2


@dataclass
class IngestConfig:
    dedup: DedupConfig = field(default_factory=DedupConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    tile: TileConfig = field(default_factory=TileConfig)
    max_file_mb: int = 50
    max_batch_gb: float = 2.0


@dataclass
class TileInferConfig:
    """推理侧切片参数（ADR-0006）：与导入期切片共用同一套几何。"""

    enabled: object = "auto"      # True | False | "auto"（长边 > 1920 时启用）
    size: int = 1024
    overlap: float = 0.2
    merge_iou: float = 0.5


@dataclass
class SamConfig:
    enabled: bool = False
    model: str = "sam2.1_t.pt"
    min_box_side_px: int = 24


@dataclass
class PrelabelConfig:
    """模型辅助标注配置（configs/prelabel.yaml）。"""

    enabled: bool = True
    model: str | None = None       # None → 使用 model_versions 中 status='production' 的权重
    device: str = "auto"
    imgsz: int = 640
    conf: float = 0.25
    iou: float = 0.5
    max_detections: int = 100
    tile: TileInferConfig = field(default_factory=TileInferConfig)
    sam: SamConfig = field(default_factory=SamConfig)
    #: 检测器类别名（或索引）→ 本项目类别 code 的别名表
    class_aliases: dict[str, str] = field(default_factory=lambda: {
        "d00": "longitudinal_crack", "d10": "transverse_crack",
        "d20": "alligator_crack", "d40": "pothole", "d44": "pothole",
        "longitudinal crack": "longitudinal_crack", "longitudinal_crack": "longitudinal_crack",
        "transverse crack": "transverse_crack", "transverse_crack": "transverse_crack",
        "alligator crack": "alligator_crack", "alligator_crack": "alligator_crack",
        "pothole": "pothole", "manhole": "pothole",
        "garbage": "garbage", "trash": "garbage", "debris": "garbage", "litter": "garbage",
    })


@dataclass
class EvaluateConfig:
    """评估口径（configs/train.yaml: evaluate）。"""

    splits: tuple[str, ...] = ("val", "test")
    export_failures: int = 20
    sahi_ablation: bool = True
    conf: float = 0.25
    iou: float = 0.5
    tiles: int = 0                 # >0 = 用该边长强制切片做对照；0 = prelabel.tile 的 auto 策略


@dataclass
class GateConfig:
    """模型门禁阈值（configs/train.yaml: gate）。"""

    baseline: str = "production"   # production | previous | none
    map50_tolerance: float = 0.005
    per_class_tolerance: float = 0.02
    min_map50: float = 0.0         # 绝对下限（首个模型也要过）
    min_class_recall: float | None = None


@dataclass
class ExportConfig:
    """ONNX 导出与一致性验收（configs/train.yaml: export）。"""

    opset: int = 17
    dynamic_batch: bool = True
    simplify: bool = True
    half: bool = False
    tolerance: float = 1e-3        # 导出包验收：ONNX 与 .pt 逐框坐标最大绝对误差


@dataclass
class TrainConfig:
    """训练配置（configs/train.yaml）。"""

    dataset: str | None = None
    arch: str = "yolo11s.pt"
    pretrained: bool = True
    resume_from: str | None = None
    imgsz: int = 640
    epochs: int = 100
    patience: int = 20
    batch: int = 16
    workers: int = 4
    seed: int = 42
    optimizer: str = "AdamW"
    lr0: float = 0.001
    lrf: float = 0.01
    warmup_epochs: float = 3.0
    cos_lr: bool = True
    device: str = "auto"
    amp: object = "auto"       # auto | True | False（auto = 有 CUDA 才开）
    plots: bool = False
    close_mosaic: int = 10
    augment: dict[str, float] = field(default_factory=dict)
    #: 垂直翻转会交换「横向/纵向」语义（ADR/roadmap 硬约束）。只有显式打开才允许非 0 flipud。
    allow_vertical_flip: bool = False
    evaluate: EvaluateConfig = field(default_factory=EvaluateConfig)
    gate: GateConfig = field(default_factory=GateConfig)
    export: ExportConfig = field(default_factory=ExportConfig)


@dataclass
class Config:
    """运行期配置（最小可用子集，未列出的键保留在 raw 中）。"""

    raw: dict[str, Any]
    data_dir: Path
    host: str = "127.0.0.1"
    port: int = 8787
    allowed_roots: tuple[Path, ...] = ()
    thumb_long_edge: int = 512
    ingest: IngestConfig = field(default_factory=IngestConfig)
    prelabel: PrelabelConfig = field(default_factory=PrelabelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    lease_seconds: int = 1800
    review_sample_ratio: float = 0.2
    config_path: Path | None = None

    # ── 派生路径 ────────────────────────────────────────────────────────────
    @property
    def db_path(self) -> Path:
        return self.data_dir / "app.db"

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def frames_dir(self) -> Path:
        return self.data_dir / "frames"

    @property
    def tiles_dir(self) -> Path:
        return self.data_dir / "tiles"

    @property
    def thumbs_dir(self) -> Path:
        return self.data_dir / "thumbs"

    @property
    def masks_dir(self) -> Path:
        return self.data_dir / "masks"

    @property
    def datasets_dir(self) -> Path:
        return self.data_dir / "datasets"

    @property
    def logs_dir(self) -> Path:
        return self.data_dir / "logs"

    @property
    def runs_dir(self) -> Path:
        """训练工作目录（ultralytics project；checkpoint/results.csv 落在这里）。"""
        return self.data_dir / "runs"

    @property
    def train_logs_dir(self) -> Path:
        return self.data_dir / "logs" / "runs"

    @property
    def exports_dir(self) -> Path:
        """边缘导出包根目录（[M4] rdinspect infer 消费）。"""
        return self.data_dir / "exports"

    @property
    def weights_dir(self) -> Path:
        """模型权重缓存目录（下载的 .pt 落在这里，而不是进程 CWD）。"""
        return self.data_dir / "weights"

    @property
    def ultralytics_dir(self) -> Path:
        """ultralytics 自身配置目录（默认会写 ~/.config/Ultralytics，被沙箱/生产环境拒绝）。"""
        return self.data_dir / "ultralytics"

    @property
    def mpl_dir(self) -> Path:
        """matplotlib 缓存目录（ultralytics 绘图会用到）。"""
        return self.data_dir / "mpl"

    def export_runtime_env(self) -> None:
        """把第三方库的运行期目录收进 data_dir（幂等，不覆盖用户已显式设置的值）。"""
        import os as _os

        _os.environ.setdefault("YOLO_CONFIG_DIR", str(self.ultralytics_dir))
        _os.environ.setdefault("MPLCONFIGDIR", str(self.mpl_dir))

    def ensure_dirs(self) -> None:
        for path in (
            self.data_dir, self.raw_dir, self.frames_dir, self.tiles_dir,
            self.thumbs_dir, self.masks_dir, self.datasets_dir, self.logs_dir,
            self.weights_dir, self.ultralytics_dir, self.mpl_dir,
            self.runs_dir, self.train_logs_dir, self.exports_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)
        self.export_runtime_env()

    # ── 路径安全 ────────────────────────────────────────────────────────────
    def resolve_source(self, path: str | Path) -> Path:
        """校验并解析导入源路径：必须位于 allowed_roots 白名单内。

        白名单为空时，默认只允许 data_dir 下的 inbox/ 目录（安全默认值）。
        """
        candidate = Path(path).expanduser()
        try:
            resolved = candidate.resolve(strict=True)
        except FileNotFoundError as exc:
            raise ConfigError(f"导入源不存在: {candidate}") from exc
        roots = self.allowed_roots or (self.data_dir / "inbox",)
        for root in roots:
            try:
                resolved.relative_to(root.resolve())
                return resolved
            except ValueError:
                continue
        raise ConfigError(
            f"路径越界：{resolved} 不在允许的根目录内 {[str(r) for r in roots]}（见 configs/default.yaml: paths.allowed_roots）"
        )

    def rel_data_path(self, path: Path) -> str:
        """转换为相对 data_dir 的 POSIX 路径（入库存这个相对路径）。"""
        return path.resolve().relative_to(self.data_dir.resolve()).as_posix()

    def abs_data_path(self, rel: str) -> Path:
        """相对路径还原为绝对路径，并阻断 `..` 越界。"""
        candidate = (self.data_dir / rel).resolve()
        candidate.relative_to(self.data_dir.resolve())
        return candidate


def _deep_get(data: dict[str, Any], *keys: str, default: Any = None) -> Any:
    node: Any = data
    for key in keys:
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


def _load_train(config_path: Path, raw: dict[str, Any]) -> TrainConfig:
    """加载训练配置：默认取同目录 train.yaml（存在则合并），顶层 `train:` 键可覆盖。"""
    override = raw.get("train")
    if isinstance(override, dict):
        train_raw = override          # 显式覆盖：overrides={"train": {...}} 直接顶替 train.yaml
    else:
        candidate = config_path.parent / "train.yaml"
        train_raw = yaml.safe_load(candidate.read_text(encoding="utf-8")) if candidate.exists() else {}
    train_raw = train_raw or {}
    model_raw = train_raw.get("model") or {}
    params = train_raw.get("train") or {}
    augment = {str(k): float(v) for k, v in (train_raw.get("augment") or {}).items()
               if isinstance(v, (int, float))}
    eval_raw = train_raw.get("evaluate") or {}
    gate_raw = train_raw.get("gate") or {}
    export_raw = (train_raw.get("export") or {}).get("onnx") or {}
    splits = eval_raw.get("splits") or ["val", "test"]
    min_recall = gate_raw.get("min_class_recall")

    known_params = {
        "imgsz", "epochs", "patience", "batch", "workers", "seed", "optimizer", "lr0", "lrf",
        "warmup_epochs", "cos_lr", "close_mosaic", "device", "amp", "plots",
    }
    return TrainConfig(
        dataset=train_raw.get("dataset", {}).get("name") if isinstance(train_raw.get("dataset"), dict)
        else train_raw.get("dataset"),
        arch=str(model_raw.get("arch", "yolo11s.pt")),
        pretrained=bool(model_raw.get("pretrained", True)),
        resume_from=model_raw.get("resume_from"),
        augment=augment,
        allow_vertical_flip=bool(train_raw.get("allow_vertical_flip", False)),
        evaluate=EvaluateConfig(
            splits=tuple(str(s) for s in splits),
            export_failures=int(eval_raw.get("export_failures", 20)),
            sahi_ablation=bool(eval_raw.get("sahi_ablation", True)),
            conf=float(eval_raw.get("conf", 0.25)),
            iou=float(eval_raw.get("iou", 0.5)),
            tiles=int(eval_raw.get("tiles", 0)),
        ),
        gate=GateConfig(
            baseline=str(gate_raw.get("baseline", "production")),
            map50_tolerance=float(gate_raw.get("map50_tolerance", 0.005)),
            per_class_tolerance=float(gate_raw.get("per_class_tolerance", 0.02)),
            min_map50=float(gate_raw.get("min_map50", 0.0)),
            min_class_recall=None if min_recall is None else float(min_recall),
        ),
        export=ExportConfig(
            opset=int(export_raw.get("opset", 17)),
            dynamic_batch=bool(export_raw.get("dynamic_batch", True)),
            simplify=bool(export_raw.get("simplify", True)),
            half=bool(export_raw.get("half", False)),
            tolerance=float((train_raw.get("export") or {}).get("tolerance", 1e-3)),
        ),
        **{key: params[key] for key in known_params if key in params},
    )


def load_config(path: str | Path | None = None, *, data_dir: str | Path | None = None,
                overrides: dict[str, Any] | None = None) -> Config:
    """加载配置。

    @param path - 配置文件路径；None 时用 ROAD_INSPECT_CONFIG 或 configs/default.yaml
    @param data_dir - 覆盖数据根目录（命令行 --data-dir 优先）
    @param overrides - 顶层键的浅覆盖（测试用）
    """
    config_path = Path(path) if path else Path(os.environ.get("ROAD_INSPECT_CONFIG", DEFAULT_CONFIG_PATH))
    if not config_path.exists():
        raise ConfigError(f"配置文件不存在: {config_path}")
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ConfigError(f"配置文件顶层必须是映射: {config_path}")
    if overrides:
        raw.update(overrides)

    resolved_data_dir = Path(data_dir or _deep_get(raw, "paths", "data_dir", default="./data"))
    if not resolved_data_dir.is_absolute():
        resolved_data_dir = (PROJECT_ROOT / resolved_data_dir).resolve()

    allowed_raw = _deep_get(raw, "paths", "allowed_roots", default=[]) or []
    allowed: list[Path] = []
    for entry in allowed_raw:
        candidate = Path(entry).expanduser()
        allowed.append(candidate if candidate.is_absolute() else (PROJECT_ROOT / candidate).resolve())
    allowed = tuple(allowed)

    ingest_raw = _deep_get(raw, "ingest", default={}) or {}
    ingest = IngestConfig(
        dedup=DedupConfig(**(ingest_raw.get("dedup") or {})),
        video=VideoConfig(**(ingest_raw.get("video") or {})),
        tile=TileConfig(**(ingest_raw.get("tile") or {})),
        max_file_mb=int(ingest_raw.get("limits", {}).get("max_file_mb", 50)),
        max_batch_gb=float(ingest_raw.get("limits", {}).get("max_batch_gb", 2.0)),
    )

    # 预标注配置：默认与 default.yaml 同目录的 prelabel.yaml（存在则合并，便于独立调参）
    prelabel_raw = raw.get("prelabel")
    if not isinstance(prelabel_raw, dict):
        prelabel_path = config_path.parent / "prelabel.yaml"
        prelabel_raw = yaml.safe_load(prelabel_path.read_text(encoding="utf-8")) if prelabel_path.exists() else {}
    prelabel_raw = prelabel_raw or {}
    tile_raw = prelabel_raw.get("tile") or {}
    sam_raw = prelabel_raw.get("sam") or {}
    prelabel = PrelabelConfig(
        enabled=bool(prelabel_raw.get("enabled", True)),
        model=(prelabel_raw.get("detector") or {}).get("model"),
        device=str((prelabel_raw.get("detector") or {}).get("device", "auto")),
        imgsz=int((prelabel_raw.get("detector") or {}).get("imgsz", 640)),
        conf=float((prelabel_raw.get("detector") or {}).get("conf", 0.25)),
        iou=float((prelabel_raw.get("detector") or {}).get("iou", 0.5)),
        max_detections=int((prelabel_raw.get("detector") or {}).get("max_detections", 100)),
        tile=TileInferConfig(
            enabled=tile_raw.get("enabled", "auto"),
            size=int(tile_raw.get("size", 1024)),
            overlap=float(tile_raw.get("overlap", 0.2)),
            merge_iou=float(tile_raw.get("merge_iou", 0.5)),
        ),
        sam=SamConfig(
            enabled=bool(sam_raw.get("enabled", False)),
            model=str(sam_raw.get("model", "sam2.1_t.pt")),
            min_box_side_px=int(sam_raw.get("min_box_side_px", 24)),
        ),
        class_aliases={str(k).lower(): str(v) for k, v in (prelabel_raw.get("class_aliases") or {}).items()}
        or PrelabelConfig().class_aliases,
    )

    return Config(
        raw=raw,
        prelabel=prelabel,
        train=_load_train(config_path, raw),
        data_dir=Path(resolved_data_dir),
        host=str(_deep_get(raw, "server", "host", default="127.0.0.1")),
        port=int(_deep_get(raw, "server", "port", default=8787)),
        allowed_roots=allowed,
        thumb_long_edge=int(_deep_get(raw, "paths", "thumb_long_edge", default=512)),
        ingest=ingest,
        lease_seconds=int(_deep_get(raw, "tasks", "lease_seconds", default=1800)),
        review_sample_ratio=float(_deep_get(raw, "review", "default_sample_ratio", default=0.2)),
        config_path=config_path,
    )
