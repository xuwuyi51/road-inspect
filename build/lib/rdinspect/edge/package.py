"""导出包版本校验（M4）：`manifest.json` 与模型/类别不匹配时必须**拒绝启动**。

设计原则：边缘端只装 ``onnxruntime + numpy + pillow``，因此校验只用 onnxruntime 的会话元信息
（输入/输出形状）与文件哈希，不依赖 ``onnx``/``torch``/``ultralytics``。

校验项与失败语义：

| 校验 | 失败含义 |
|---|---|
| `schema_version` 存在且 ≤ 本端支持版本 | 包比运行时新：拒绝启动（避免解析出错误结果） |
| `model_file` 存在 | 包不完整 |
| `model_sha256` 与文件实际哈希一致 | 文件被替换/损坏 |
| `labels.txt` 行数 == `manifest.nc` == 模型输出通道数 − 4 | 类别顺序/数量漂移（最危险：会把类别标错） |
| `preprocess.json` 关键字段齐全；静态输入尺寸与 `input_size` 一致 | 预处理契约不匹配 |
| 类别名与调用方期望的类别表一致（可选） | 与工作站的数据集类别表不一致 |

哈希校验对 5–20MB 的 ONNX 只需几十毫秒，因此默认开启；``--skip-hash`` 只在超大模型+信任介质时使用。
"""

from __future__ import annotations

import ast
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

#: 本端支持的导出包 schema 版本（manifest.schema_version）
SUPPORTED_SCHEMA_VERSION = 1
REQUIRED_PREPROCESS_KEYS = ("input_size", "channel_order", "scale", "resize", "postprocess")


class PackageError(RuntimeError):
    """导出包校验失败（拒绝启动，不是"运行中出错"）。"""


@dataclass
class PackageInfo:
    """校验通过后的导出包信息（供推理循环使用）。"""

    root: Path
    model_path: Path
    labels: list[str]
    classes: list[dict[str, Any]]
    manifest: dict[str, Any]
    preprocess: dict[str, Any]
    schema_version: int
    name: str
    version: str
    imgsz: int
    conf: float
    iou: float
    max_detections: int
    dynamic_batch: bool
    checks: list[str] = field(default_factory=list)

    @property
    def model_label(self) -> str:
        return f"{self.name}:{self.version}"

    def as_dict(self) -> dict[str, Any]:
        return {"root": str(self.root), "model": str(self.model_path), "model_label": self.model_label,
                "schema_version": self.schema_version, "nc": len(self.labels), "labels": self.labels,
                "imgsz": self.imgsz, "conf": self.conf, "iou": self.iou,
                "max_detections": self.max_detections, "dynamic_batch": self.dynamic_batch,
                "dataset": (self.manifest.get("dataset") or {}).get("manifest_hash"),
                "weights_sha256": self.manifest.get("weights_sha256"),
                "model_sha256": self.manifest.get("model_sha256"),
                "parity": (self.manifest.get("parity") or {}).get("passed"),
                "checks": self.checks}


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _session_shapes(session: Any) -> tuple[list[Any], list[Any], list[str]]:
    inputs = [list(getattr(item, "shape", []) or []) for item in session.get_inputs()]
    outputs = [list(getattr(item, "shape", []) or []) for item in session.get_outputs()]
    providers = list(session.get_providers()) if hasattr(session, "get_providers") else []
    return inputs, outputs, providers


def _literal(value: Any) -> Any:
    """解析 ultralytics 写进 ONNX metadata 的 Python 字面量（`"{0: 'a', 1: 'b'}"`）或 JSON。"""
    if not isinstance(value, str):
        return value
    for parser in (json.loads, ast.literal_eval):
        try:
            return parser(value)
        except Exception:  # noqa: BLE001 - 两种都失败就交给调用方
            continue
    return None


def _metadata_names(session: Any) -> list[str] | None:
    """读取模型内嵌的类别名（ultralytics 导出时会写入 `names`），用于交叉校验 labels.txt。"""
    getter = getattr(session, "get_modelmeta", None)
    if getter is None:
        return None
    try:
        metadata = getattr(getter(), "custom_metadata_map", None) or {}
    except Exception:  # noqa: BLE001 - 没有 metadata 的模型
        return None
    parsed = _literal(metadata.get("names"))
    if isinstance(parsed, dict) and parsed:
        try:
            return [str(parsed[key]) for key in sorted(parsed, key=lambda item: int(item))]
        except (TypeError, ValueError):
            return None
    if isinstance(parsed, (list, tuple)) and parsed:
        return [str(item) for item in parsed]
    return None


def _metadata_imgsz(session: Any) -> list[int] | None:
    getter = getattr(session, "get_modelmeta", None)
    if getter is None:
        return None
    try:
        metadata = getattr(getter(), "custom_metadata_map", None) or {}
    except Exception:  # noqa: BLE001
        return None
    parsed = _literal(metadata.get("imgsz"))
    if isinstance(parsed, (list, tuple)) and len(parsed) == 2:
        try:
            return [int(parsed[0]), int(parsed[1])]
        except (TypeError, ValueError):
            return None
    return None


def _static_int(value: Any) -> int | None:
    """把 ONNX 维度值转成 int；动态维（字符串/None/0）返回 None。"""
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value > 0:
        return value
    return None


def validate_package(package_dir: str | Path, *, session_factory: Callable[[str], Any] | None = None,
                     expect_labels: Sequence[str] | None = None, verify_hash: bool = True,
                     providers: Sequence[str] | None = None) -> PackageInfo:
    """校验导出包并返回 :class:`PackageInfo`；任何不匹配都抛 :class:`PackageError`。

    @param session_factory - 可注入的会话工厂（测试用假对象即可，无需真实 ONNX）
    @param expect_labels   - 期望的类别 code 顺序（如与工作站类别表比对时传入）
    @param verify_hash     - 是否校验 model.onnx 的 sha256（默认校验）
    """
    root = Path(package_dir)
    if not root.is_dir():
        raise PackageError(f"导出包目录不存在: {root}")
    checks: list[str] = []

    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        raise PackageError(f"缺少 manifest.json：{root} 不是导出包（见 docs/12-edge-inference.md）")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PackageError(f"manifest.json 解析失败: {exc}") from exc
    if not isinstance(manifest, dict):
        raise PackageError("manifest.json 顶层必须是对象")

    schema_version = manifest.get("schema_version")
    if schema_version is None:
        raise PackageError("manifest.json 缺少 schema_version：无法判断包格式，拒绝启动"
                           "（请用 rdinspect model export 重新导出）")
    if not isinstance(schema_version, int) or schema_version > SUPPORTED_SCHEMA_VERSION:
        raise PackageError(f"导出包 schema_version={schema_version} 高于本端支持的 "
                           f"{SUPPORTED_SCHEMA_VERSION}：请升级边缘 runtime 或重新导出")
    checks.append(f"schema_version={schema_version}")

    model_path = root / str(manifest.get("model_file") or "model.onnx")
    if not model_path.exists():
        raise PackageError(f"缺少模型文件: {model_path}")
    if verify_hash:
        expected = str(manifest.get("model_sha256") or "")
        if not expected:
            raise PackageError("manifest.json 缺少 model_sha256，无法验证模型完整性")
        actual = sha256_file(model_path)
        if actual != expected:
            raise PackageError(f"模型哈希不匹配：manifest={expected[:12]}… 实际={actual[:12]}…"
                               f"（文件被替换或损坏，拒绝启动）")
        checks.append("model_sha256 一致")

    labels_path = root / "labels.txt"
    if not labels_path.exists():
        raise PackageError("缺少 labels.txt")
    labels = [line.strip() for line in labels_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not labels:
        raise PackageError("labels.txt 为空")
    manifest_labels = [str(item) for item in (manifest.get("labels") or [])]
    if manifest_labels and manifest_labels != labels:
        raise PackageError(f"labels.txt 与 manifest.labels 不一致：{labels} ≠ {manifest_labels}")
    declared_nc = manifest.get("nc")
    if isinstance(declared_nc, int) and declared_nc != len(labels):
        raise PackageError(f"manifest.nc={declared_nc} 与 labels.txt 行数 {len(labels)} 不一致")
    checks.append(f"labels={len(labels)}")

    if expect_labels is not None and list(expect_labels) != labels:
        raise PackageError(f"类别顺序与期望不一致：包内 {labels} ≠ 期望 {list(expect_labels)}"
                           f"（类别顺序变化必须重新冻结数据集并重新导出，见 ADR-0005）")

    preprocess_path = root / "preprocess.json"
    if not preprocess_path.exists():
        raise PackageError("缺少 preprocess.json（预处理契约）")
    try:
        preprocess = json.loads(preprocess_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PackageError(f"preprocess.json 解析失败: {exc}") from exc
    missing = [key for key in REQUIRED_PREPROCESS_KEYS if key not in preprocess]
    if missing:
        raise PackageError(f"preprocess.json 缺少字段 {missing}")
    input_size = preprocess.get("input_size") or []
    if not (isinstance(input_size, (list, tuple)) and len(input_size) == 2
            and all(isinstance(value, int) and value > 0 for value in input_size)):
        raise PackageError(f"preprocess.input_size 非法: {input_size}")
    if str(preprocess.get("channel_order", "")).upper() != "RGB":
        raise PackageError(f"preprocess.channel_order 必须为 RGB，实际 {preprocess.get('channel_order')!r}")
    checks.append(f"preprocess.input_size={list(input_size)}")

    # ── 用 onnxruntime 会话校验形状（不依赖 onnx/torch）──
    if session_factory is None:
        try:
            import onnxruntime  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - 边缘环境缺依赖
            raise PackageError("未安装 onnxruntime：pip install onnxruntime（见 docs/12-edge-inference.md）") from exc
        from .onnx_runtime import disable_telemetry  # noqa: PLC0415

        disable_telemetry(onnxruntime)

        def session_factory(path: str) -> Any:  # type: ignore[misc]
            options = onnxruntime.SessionOptions()
            options.log_severity_level = 3
            available = onnxruntime.get_available_providers()
            chosen = [name for name in (providers or ["CPUExecutionProvider"]) if name in available]
            return onnxruntime.InferenceSession(path, sess_options=options,
                                                providers=chosen or ["CPUExecutionProvider"])

    session = session_factory(str(model_path))
    inputs, outputs, session_providers = _session_shapes(session)
    if not inputs or not outputs:
        raise PackageError("模型没有输入/输出，文件可能不是有效的 ONNX")
    input_shape = inputs[0]
    spatial = [dim for dim in input_shape[-2:]] if len(input_shape) >= 4 else []
    static_side = [_static_int(dim) for dim in spatial]
    if len(static_side) == 2 and all(static_side):
        if static_side != list(input_size):
            raise PackageError(f"模型输入尺寸 {static_side} 与 preprocess.input_size {list(input_size)} 不一致")
        checks.append(f"模型输入尺寸={static_side}")
    batch_dim = input_shape[0] if input_shape else None
    dynamic_batch = _static_int(batch_dim) is None
    checks.append(f"dynamic_batch={dynamic_batch}")

    # YOLO 检测头输出形如 (1, 4+nc, anchors)：anchors 通常是 2100/8400 这类静态大数，
    # 因此不能"取最后一个静态维"当通道数，必须看 **是否有任一维等于 4+nc**。
    output_shape = outputs[0]
    expected_channels = 4 + len(labels)
    static_dims = [dim for dim in output_shape[1:] if isinstance(dim, int) and not isinstance(dim, bool) and dim > 0]
    if not static_dims:
        checks.append("输出形状全动态（跳过通道校验）")
    elif expected_channels in static_dims:
        checks.append(f"输出通道={expected_channels}")
    else:
        raise PackageError(f"模型输出形状 {output_shape} 里没有任何一维等于 4+nc={expected_channels}"
                           f"（labels 共 {len(labels)} 个）：类别顺序/数量漂移，拒绝启动")

    # 模型内嵌类别名（ultralytics 会写 names/imgsz）：这是"文本文件被整体改过"的兜底防线
    embedded = _metadata_names(session)
    if embedded is not None:
        if embedded != labels:
            raise PackageError(f"模型内嵌类别名与 labels.txt 不一致：模型 {embedded} ≠ 包 {labels}"
                               f"（文本清单被改过或包与模型不是同一次导出，拒绝启动）")
        checks.append("模型内嵌类别名一致")
    embedded_size = _metadata_imgsz(session)
    if embedded_size is not None and list(input_size) != embedded_size:
        raise PackageError(f"模型内嵌 imgsz={embedded_size} 与 preprocess.input_size {list(input_size)} 不一致")
    if embedded_size is not None:
        checks.append(f"模型内嵌 imgsz={embedded_size}")

    postprocess = preprocess.get("postprocess") or {}
    classes = manifest.get("classes") or []
    return PackageInfo(
        root=root, model_path=model_path, labels=labels, classes=list(classes), manifest=manifest,
        preprocess=preprocess, schema_version=int(schema_version),
        name=str(manifest.get("name") or root.name), version=str(manifest.get("version") or "unknown"),
        imgsz=int(input_size[0]), conf=float(postprocess.get("conf", 0.25)),
        iou=float(postprocess.get("iou", 0.5)),
        max_detections=int(postprocess.get("max_detections", 100)),
        dynamic_batch=dynamic_batch, checks=[*checks, f"providers={session_providers}"],
    )
