"""第 4 层 · 闭环：宫格拼图与留底视频（Solver_Prework.md §5.3）。

本模块只吃 **numpy 图像数组**，不涉及渲染本身，因此与 pytorch3d 解耦：
渲染器负责产出 ``[T, H, W, 3] uint8`` 的逐帧图像，本模块负责

1. 按步态相位抽帧 → 裁掉无效背景 → 拼成 R×C 宫格图（提交给 VLM）；
2. 多机位横向拼接 → 写出 mp4 / gif（留底给人看，也可作为 VLM 输入形态的备选）。

分辨率纪律：宫格图中单帧只占 1/(R·C) 面积，若整图再被模型下采样，
膝屈曲 98° 与 123° 的差别可能直接丢失。因此 :func:`crop_to_content`
默认开启——把人物填满每格，是零成本且必做的一步。
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "crop_to_content",
    "union_content_box",
    "resize",
    "make_grid",
    "hstack_views",
    "write_video",
    "write_gif",
]


def _require_cv2():
    try:
        import cv2  # noqa: F401
    except ImportError as e:  # pragma: no cover - 环境相关
        raise ImportError(
            "宫格拼图与视频输出需要 opencv-python（目标 Linux 环境已装 4.13.0）。"
        ) from e
    import cv2
    return cv2


# --------------------------------------------------------------------------- #
def union_content_box(frames: Sequence[np.ndarray],
                      bg_tol: int = 8, margin: int = 12) -> Tuple[int, int, int, int]:
    """计算一组帧的**共同**内容包围盒 ``(y0, y1, x0, x1)``。

    必须取并集而非逐帧各裁各的：逐帧裁剪会让人物在宫格里忽大忽小，
    VLM 会误判为"身高/步幅发生了变化"，引入虚假信号。
    """
    if not len(frames):
        raise ValueError("frames 为空")
    h, w = frames[0].shape[:2]
    y0, y1, x0, x1 = h, 0, w, 0
    for f in frames:
        gray = f[..., :3].astype(np.int16)
        bg = np.median(gray.reshape(-1, 3), axis=0)
        mask = (np.abs(gray - bg).max(axis=-1) > bg_tol)
        if not mask.any():
            continue
        ys, xs = np.where(mask)
        y0, y1 = min(y0, int(ys.min())), max(y1, int(ys.max()))
        x0, x1 = min(x0, int(xs.min())), max(x1, int(xs.max()))
    if y1 <= y0 or x1 <= x0:
        return 0, h, 0, w
    y0 = max(0, y0 - margin); x0 = max(0, x0 - margin)
    y1 = min(h, y1 + margin + 1); x1 = min(w, x1 + margin + 1)
    return y0, y1, x0, x1


def crop_to_content(frames: Sequence[np.ndarray],
                    bg_tol: int = 8, margin: int = 12) -> List[np.ndarray]:
    """按共同包围盒裁剪所有帧，保留统一的人物尺度。"""
    y0, y1, x0, x1 = union_content_box(frames, bg_tol, margin)
    return [f[y0:y1, x0:x1] for f in frames]


def resize(img: np.ndarray, size: int, pad_value: int = 255) -> np.ndarray:
    """等比缩放到 ``size×size`` 画布并居中留白，不拉伸变形。"""
    cv2 = _require_cv2()
    h, w = img.shape[:2]
    scale = size / max(h, w)
    nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    small = cv2.resize(img, (nw, nh), interpolation=interp)
    canvas = np.full((size, size, img.shape[2]), pad_value, dtype=img.dtype)
    oy, ox = (size - nh) // 2, (size - nw) // 2
    canvas[oy:oy + nh, ox:ox + nw] = small
    return canvas


# --------------------------------------------------------------------------- #
def make_grid(frames: Sequence[np.ndarray], rows: int = 2, cols: int = 4,
              cell_px: int = 512, crop: bool = True,
              label: bool = True, border: int = 2) -> np.ndarray:
    """把 ``rows*cols`` 帧拼成宫格图，顺序为**从左到右、从上到下**。

    ``label=True`` 会在每格左上角标注序号——VLM prompt 里声明了时间顺序，
    显式标号可以消除"模型自己猜顺序"这一不可控环节。
    """
    n = rows * cols
    if len(frames) != n:
        raise ValueError(f"需要 {n} 帧（{rows}×{cols}），收到 {len(frames)}")
    cv2 = _require_cv2()
    fs = crop_to_content(frames) if crop else list(frames)
    cells = [resize(f, cell_px) for f in fs]
    if label:
        for i, c in enumerate(cells, 1):
            cv2.putText(c, str(i), (int(cell_px * 0.03), int(cell_px * 0.10)),
                        cv2.FONT_HERSHEY_SIMPLEX, cell_px / 512.0 * 1.2,
                        (0, 0, 0), max(1, cell_px // 256), cv2.LINE_AA)
    if border > 0:
        cells = [cv2.copyMakeBorder(c, border, border, border, border,
                                    cv2.BORDER_CONSTANT, value=(160, 160, 160))
                 for c in cells]
    rows_img = [np.concatenate(cells[r * cols:(r + 1) * cols], axis=1)
                for r in range(rows)]
    return np.concatenate(rows_img, axis=0)


def hstack_views(views: Sequence[np.ndarray]) -> np.ndarray:
    """把多个机位的同一时刻画面横向拼接（高度不一致时按最大高度补白）。"""
    if not len(views):
        raise ValueError("views 为空")
    h = max(v.shape[0] for v in views)
    out = []
    for v in views:
        if v.shape[0] != h:
            pad = np.full((h - v.shape[0], v.shape[1], v.shape[2]), 255, dtype=v.dtype)
            v = np.concatenate([v, pad], axis=0)
        out.append(v)
    return np.concatenate(out, axis=1)


# --------------------------------------------------------------------------- #
def write_video(frames: Sequence[np.ndarray], path: Path | str,
                fps: float = 24.0, rgb: bool = True) -> Path:
    """写出 mp4。``frames`` 为 ``[T, H, W, 3] uint8``（默认 RGB 顺序）。"""
    cv2 = _require_cv2()
    if not len(frames):
        raise ValueError("frames 为空")
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"无法打开视频写入器：{p}")
    try:
        for f in frames:
            img = np.ascontiguousarray(f[..., :3])
            writer.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if rgb else img)
    finally:
        writer.release()
    return p


def write_gif(frames: Sequence[np.ndarray], path: Path | str,
              fps: float = 24.0, max_width: int = 1280) -> Path:
    """写出 gif。

    GIF 的帧间延迟以**厘秒**存储，24fps = 4.2 厘秒；大量播放器会把小于 2 厘秒的
    延迟强制改成 10 厘秒（=10fps）。因此 **gif 结构上无法忠实表达高帧率**，
    做绝对速度判断时请用 mp4。
    """
    try:
        import imageio.v2 as imageio
    except ImportError as e:
        raise ImportError("写 gif 需要 imageio；若环境缺失可跳过 gif，仅保留 mp4。") from e
    cv2 = _require_cv2()
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    out = []
    for f in frames:
        img = np.ascontiguousarray(f[..., :3])
        if img.shape[1] > max_width:
            s = max_width / img.shape[1]
            img = cv2.resize(img, (max_width, int(round(img.shape[0] * s))),
                             interpolation=cv2.INTER_AREA)
        out.append(img)
    # imageio 跨版本对 duration 的单位（秒/毫秒）不一致，优先用语义明确的 fps
    try:
        imageio.mimsave(str(p), out, format="GIF", fps=float(fps), loop=0)
    except TypeError:
        imageio.mimsave(str(p), out, format="GIF", duration=1000.0 / float(fps), loop=0)
    return p
