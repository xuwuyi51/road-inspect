#!/usr/bin/env python3
"""构建边缘离线安装包（M4）：wheels + 导出包 + 配置 + 一键安装脚本。

用法：
  python3 scripts/build_edge_bundle.py --package data/exports/yolo11n-road-xxx \
      --out /tmp/edge-bundle --python-version 3.12
  python3 scripts/build_edge_bundle.py --package <dir> --out <dir> --plan   # 只打印计划，不下载

产物结构：

```
<out>/
├── wheels/                 # onnxruntime + pillow + numpy + rdinspect 及其依赖（离线安装用）
├── model/<name>-<version>/ # 导出包（model.onnx / labels.txt / preprocess.json / manifest.json / parity.json）
├── samples/smoke.jpg       # 冒烟样图（install.sh 自检用）
├── configs/edge.yaml       # 边缘配置模板
├── install.sh              # 目标机执行：venv + 离线安装 + 自检
├── verify.sh               # 目标机执行：校验包完整性（哈希/类别/版本）
└── README.md               # 目标机操作说明（含无网运行示例）
```

设计取舍：

* 默认用 ``pip download --only-binary=:all:`` 只取 wheel（目标机无需编译工具链）；
* ``--platform/--python-version`` 可交叉下载（例如在 x86 工作站为车载 ARM64 设备准备包）；
* ``rdinspect`` 自身用 ``--no-deps`` 单独下载，其余依赖由 pip 解析，避免把工作站的 torch 带进去；
* 不联网的目标机上只需 ``pip install --no-index --find-links wheels road-inspect onnxruntime pillow numpy``
  （发行版名是 `road-inspect`，命令名是 `rdinspect`）；
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
#: 边缘推理真正需要的最小依赖（不含 fastapi/uvicorn：那是工作站 serve 才用的）
#: pyyaml 是 rdinspect.config 的模块级依赖，即便只跑 infer 也会被导入
EDGE_REQUIREMENTS = ("onnxruntime", "pillow", "numpy", "pyyaml")
EDGE_INSTALL_ORDER = ("numpy", "pillow", "onnxruntime", "pyyaml", "road-inspect")
DEFAULT_EDGE_CONFIG = PROJECT_ROOT / "configs" / "edge.yaml"


def _run(command: list[str], *, dry_run: bool) -> None:
    printable = " ".join(command)
    print(f"  $ {printable}", flush=True)
    if dry_run:
        return
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0:
        # pip 的报错要完整暴露出来，否则只看到 CalledProcessError 无从排查
        print(completed.stdout[-2000:], file=sys.stderr)
        print(completed.stderr[-2000:], file=sys.stderr)
        raise SystemExit(f"命令失败（退出码 {completed.returncode}）：{printable}")


def build(package: Path, out: Path, *, python_version: str | None = None, platform: str | None = None,
          dry_run: bool = False, skip_wheels: bool = False, pip: str = sys.executable) -> dict:
    if not (package / "manifest.json").exists():
        raise SystemExit(f"{package} 不是导出包（缺少 manifest.json）")
    manifest = json.loads((package / "manifest.json").read_text(encoding="utf-8"))
    bundle_model = out / "model" / (f"{manifest.get('name')}-{manifest.get('version')}"
                                    f"-{manifest.get('imgsz')}")
    wheels = out / "wheels"

    print(f"[1/5] 目录准备：{out}")
    if not dry_run:
        wheels.mkdir(parents=True, exist_ok=True)
        bundle_model.parent.mkdir(parents=True, exist_ok=True)

    if skip_wheels:
        print("[2/5] 跳过 wheels 下载（--skip-wheels：只生成骨架，无网机器上也能准备目录结构）")
    print("[2/5] 下载边缘依赖 wheels（onnxruntime/pillow/numpy，含依赖）" if not skip_wheels else "")
    cross: list[str] = []
    if platform:
        cross += ["--platform", platform]
    if python_version:
        cross += ["--python-version", python_version]
    if cross:
        cross += ["--only-binary=:all:"]
    if not skip_wheels:
        _run([pip, "-m", "pip", "download", *EDGE_REQUIREMENTS, "-d", str(wheels),
              "-i", "https://pypi.tuna.tsinghua.edu.cn/simple", *cross], dry_run=dry_run)
        print("[3/5] 构建 rdinspect 自身 wheel（--no-deps：避免把训练依赖带进边缘）")
        # 注意：`pip download <本地目录>` 不会为本地项目构建 wheel，必须用 `pip wheel`
        _run([pip, "-m", "pip", "wheel", "--no-deps", "-w", str(wheels), str(PROJECT_ROOT)],
             dry_run=dry_run)
        produced = sorted(item.name for item in wheels.glob("road_inspect-*.whl"))
        if not produced and not dry_run:
            raise SystemExit("打包失败：wheels/ 里没有 road_inspect wheel（离线安装会缺 rdinspect）")
        if produced:
            print(f"      → {produced[-1]}")

    print(f"[4/5] 拷贝导出包 → {bundle_model}")
    if not dry_run:
        if bundle_model.exists():
            shutil.rmtree(bundle_model)
        shutil.copytree(package, bundle_model)
        (out / "configs").mkdir(parents=True, exist_ok=True)
        if DEFAULT_EDGE_CONFIG.exists():
            template = DEFAULT_EDGE_CONFIG.read_text(encoding="utf-8")
            template = template.replace("package_dir: ./exports/yolo11s-road-2026.09.12-a",
                                        f"package_dir: ./model/{bundle_model.name}")
            (out / "configs" / "edge.yaml").write_text(template, encoding="utf-8")

    print("[5/5] 生成冒烟样图 + install.sh / verify.sh / README.md")
    if not dry_run:
        samples = out / "samples"
        samples.mkdir(parents=True, exist_ok=True)
        _write_smoke_image(samples / "smoke.jpg")
        (out / "install.sh").write_text(_install_script(bundle_model.name), encoding="utf-8")
        (out / "verify.sh").write_text(_verify_script(bundle_model.name), encoding="utf-8")
        (out / "README.md").write_text(_readme(manifest, bundle_model.name), encoding="utf-8")
        for script in ("install.sh", "verify.sh"):
            (out / script).chmod(0o755)
    sizes = {}
    if not dry_run:
        sizes = {"wheels": sum(item.stat().st_size for item in wheels.glob("*")),
                 "model": sum(item.stat().st_size for item in bundle_model.rglob("*") if item.is_file())}
    return {"out": str(out), "model_dir": str(bundle_model), "manifest": manifest, "sizes": sizes}


def _write_smoke_image(path: Path) -> None:
    """生成一张最小样图：离线自检（install.sh）需要真实输入，不能依赖工作站的测试数据。"""
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (320, 240), (78, 78, 82))
    draw = ImageDraw.Draw(image)
    draw.line([(20, 120), (300, 128)], fill=(30, 30, 32), width=4)          # 一条横向裂缝
    draw.ellipse([60, 40, 110, 85], fill=(45, 43, 45))                       # 一个坑洞
    draw.rectangle([200, 160, 260, 200], fill=(196, 188, 160))               # 一处垃圾
    image.save(path, quality=88)


def _install_script(model_dir: str) -> str:
    return f"""#!/usr/bin/env bash
# 边缘端一键安装（无网环境）。目标机需要 python3.11+ 与 venv 模块。
set -euo pipefail
HERE="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
PYTHON="${{PYTHON:-python3}}"

echo "[1/4] 创建虚拟环境 $HERE/.venv"
"$PYTHON" -m venv "$HERE/.venv"

echo "[2/4] 离线安装依赖（不访问网络）"
"$HERE/.venv/bin/pip" install --no-index --find-links "$HERE/wheels" \\
    onnxruntime pillow numpy rdinspect

echo "[3/4] 自检：导出包校验 + 冒烟推理"
"$HERE/.venv/bin/rdinspect" infer --package "$HERE/model/{model_dir}" --input "$HERE/model/{model_dir}" \\
    --limit 1 --out "$HERE/.smoke" --no-resume || {{
        echo "自检失败：请检查 wheels 与导出包是否完整" >&2; exit 1; }}

echo "[4/4] 完成。示例："
echo "  $HERE/.venv/bin/rdinspect infer --package $HERE/model/{model_dir} \\\\"
echo "      --input /mnt/sd/photos --out /mnt/sd/out --edge-config $HERE/configs/edge.yaml --resume"
"""


def _verify_script(model_dir: str) -> str:
    return f"""#!/usr/bin/env bash
# 校验导出包完整性：manifest 与模型文件哈希、类别数、schema 版本（不匹配则非 0 退出）
set -euo pipefail
HERE="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
"$HERE/.venv/bin/python" - "$HERE/model/{model_dir}" <<'PY'
import sys
from rdinspect.edge.package import PackageError, validate_package

try:
    info = validate_package(sys.argv[1])
except PackageError as exc:
    print(f"校验失败：{{exc}}", file=sys.stderr)
    raise SystemExit(1)
print(f"校验通过：{{info.model_label}} · schema {{info.schema_version}} · {{len(info.labels)}} 类 · imgsz {{info.imgsz}}")
print("检查项：" + "；".join(info.checks))
PY
"""


def verify_bundle(out: Path, model_dir: Path, *, skip_wheels: bool = False) -> dict:  # noqa: D401
    """在临时 venv 里用 --no-index 离线安装并跑一张图（真正的"目标机模拟"）。"""
    import tempfile

    if skip_wheels:
        return {"verified": False, "reason": "--skip-wheels 时没有 wheels 可验证"}
    with tempfile.TemporaryDirectory() as tmp:
        venv = Path(tmp) / "venv"
        subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True)
        pip = str(venv / "bin" / "pip")
        subprocess.run([pip, "install", "-q", "--no-index", "--no-deps",
                        "--find-links", str(out / "wheels"), *EDGE_INSTALL_ORDER], check=True)
        smoke_out = Path(tmp) / "out"
        completed = subprocess.run(
            [str(venv / "bin" / "rdinspect"), "infer", "--package", str(out / "model" / model_dir.name),
             "--input", str(out / "samples"), "--out", str(smoke_out), "--limit", "1",
             "--json"],
            capture_output=True, text=True)
        ok = completed.returncode == 0 and (smoke_out / "results.jsonl").exists()
        return {"verified": ok, "returncode": completed.returncode,
                "stderr": (completed.stderr or "").strip().splitlines()[-1][:160] if completed.stderr else "",
                "wheels": sorted(item.name for item in (out / "wheels").glob("*.whl"))}


def _readme(manifest: dict, model_dir: str) -> str:
    labels = ", ".join(manifest.get("labels") or [])
    return f"""# 边缘推理离线包 · {manifest.get('name')}:{manifest.get('version')}

| 项 | 值 |
|---|---|
| schema_version | {manifest.get('schema_version')} |
| 类别（顺序即类别下标） | {labels} |
| 输入 | {manifest.get('imgsz')}×{manifest.get('imgsz')}，RGB，居中 letterbox（见 `preprocess.json`） |
| 数据集清单哈希 | {(manifest.get('dataset') or {}).get('manifest_hash')} |
| 模型 sha256 | `{manifest.get('model_sha256')}` |
| 一致性验收 | {(manifest.get('parity') or {}).get('passed')}（ONNX↔.pt 逐框误差 ≤ {(manifest.get('parity') or {}).get('tolerance')}） |

## 目标机安装（无网）

```bash
tar -xzf edge-bundle.tar.gz && cd edge-bundle
./install.sh          # 建 venv + 离线安装 wheels（--no-index）+ 冒烟自检
./verify.sh           # 只校验导出包（哈希/类别/内嵌 names/schema）
```

边缘端最小依赖：`numpy + pillow + onnxruntime + pyyaml`（+ `road-inspect` 自身）。
**不需要** fastapi/uvicorn/torch/ultralytics；视频或流输入额外需要 OpenCV 或 ffmpeg。

## 运行

```bash
.venv/bin/rdinspect infer --package ./model/{model_dir} \\
    --input /mnt/sd/photos --out /mnt/sd/out --resume --threads 4
# 视频/流（需 OpenCV 或 ffmpeg）：
.venv/bin/rdinspect infer --package ./model/{model_dir} --input patrol.mp4 --fps 2 --out ./out
.venv/bin/rdinspect infer --package ./model/{model_dir} --input rtsp://192.168.1.10/stream --out ./out
# 性能基准：
.venv/bin/rdinspect infer --package ./model/{model_dir} --input /mnt/sd/photos --benchmark 100 --out ./bench
```

输出：`out/results.jsonl`（每图一条）、`out/results.csv`（每检测一条）、`out/.infer-state.json`（断点续跑）。
断点续跑以 JSONL + 状态文件为准，中断后重跑不会重复计数。
"""


def main() -> int:
    parser = argparse.ArgumentParser(description="构建边缘离线安装包")
    parser.add_argument("--package", required=True, help="导出包目录")
    parser.add_argument("--out", required=True, help="产物目录")
    parser.add_argument("--python-version", default=None, help="目标 Python 版本（交叉下载时用，如 3.12）")
    parser.add_argument("--platform", default=None,
                        help="目标平台（如 manylinux_2_28_x86_64 / manylinux_2_28_aarch64）")
    parser.add_argument("--plan", action="store_true", help="只打印计划，不下载/不落盘")
    parser.add_argument("--skip-wheels", action="store_true",
                        help="跳过 pip download（无网环境/只想验证目录结构与脚本时用）")
    parser.add_argument("--verify", action="store_true",
                        help="构建后在临时 venv 里离线安装并冒烟推理一次（模拟目标机）")
    args = parser.parse_args()
    result = build(Path(args.package).resolve(), Path(args.out).resolve(),
                   python_version=args.python_version, platform=args.platform, dry_run=args.plan,
                   skip_wheels=args.skip_wheels)
    if args.verify and not args.plan:
        verification = verify_bundle(Path(args.out).resolve(), Path(result["model_dir"]),
                                    skip_wheels=args.skip_wheels)
        result["verify"] = verification
        print("目标机模拟：", "通过 ✅" if verification.get("verified") else f"未通过 ❌ {verification}")
        if not verification.get("verified"):
            return 1
    print(json.dumps({key: value for key, value in result.items() if key != "manifest"},
                     ensure_ascii=False, indent=2))
    if result["sizes"]:
        total = sum(result["sizes"].values())
        print(f"体积：wheels {result['sizes']['wheels'] / 1e6:.1f}MB + "
              f"model {result['sizes']['model'] / 1e6:.1f}MB = {total / 1e6:.1f}MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
