"""感知哈希（pHash）与汉明距离：近似重复检测（连拍/相邻视频帧）。

实现：32×32 灰度 → 二维 DCT(II) → 取左上 8×8 低频 → 与中位数比较得到 64bit。
不依赖 scipy（DCT 矩阵用 numpy 直接构造），与 docs/03-data-model.md 的去重约定一致。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


def _dct_matrix(n: int) -> np.ndarray:
    """正交 DCT-II 矩阵（行 = 频率 k，列 = 样本 x）。"""
    k = np.arange(n).reshape(-1, 1)
    x = np.arange(n).reshape(1, -1)
    matrix = np.cos(np.pi * (2 * x + 1) * k / (2 * n))
    matrix[0, :] *= np.sqrt(1.0 / n)
    matrix[1:, :] *= np.sqrt(2.0 / n)
    return matrix


_DCT32 = _dct_matrix(32)


def phash_from_image(image: Image.Image, hash_size: int = 8, highfreq_factor: int = 4) -> str:
    """计算 64bit pHash（16 位十六进制）。"""
    size = hash_size * highfreq_factor
    gray = ImageOps.exif_transpose(image).convert("L").resize((size, size), Image.LANCZOS)
    pixels = np.asarray(gray, dtype=np.float64)
    dct = _DCT32 @ pixels @ _DCT32.T
    low = dct[:hash_size, :hash_size]
    median = np.median(low)
    bits = (low > median).flatten()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return f"{value:016x}"


def phash_file(path: str | Path) -> str:
    with Image.open(path) as image:
        return phash_from_image(image)


def hamming_hex(a: str, b: str) -> int:
    """两个十六进制 pHash 的汉明距离（位数不同按左对齐补零处理）。"""
    width = max(len(a), len(b))
    return ((int(a, 16) << (4 * (width - len(a)))) ^ (int(b, 16) << (4 * (width - len(b))))).bit_count()


def find_near_duplicate(phash: str, candidates: list[tuple[int, str]], threshold: int = 6) -> int | None:
    """在候选 [(image_id, phash)] 中找最近的近似重复；返回 image_id 或 None。"""
    best: tuple[int, int] | None = None
    for image_id, other in candidates:
        if not other:
            continue
        distance = hamming_hex(phash, other)
        if distance <= threshold and (best is None or distance < best[1]):
            best = (image_id, distance)
    return best[0] if best else None
