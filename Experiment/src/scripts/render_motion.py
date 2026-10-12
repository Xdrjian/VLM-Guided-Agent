#!/usr/bin/env python
"""多机位渲染 CLI。

    # 渲染参考动作（A0 探针素材）
    python scripts/render_motion.py --motion ../pipeline/up_bone/A_ref_straight \
        --out ../pipeline/renders/A_ref --tag ref

    # 渲染求解结果，跳过视频只出宫格（快）
    python scripts/render_motion.py --motion ../pipeline/solved/heavy \
        --out ../pipeline/renders/heavy --tag heavy --no-video

需要 pytorch3d，只能在 Linux 机上跑。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from motionedit.prompts import GridSpec                       # noqa: E402
from motionedit.renderer import MotionRenderer, RenderConfig  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="多机位相位抽帧渲染")
    ap.add_argument("--motion", type=Path, required=True, help="pkl 文件或同名目录")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tag", default="motion")
    ap.add_argument("--rows", type=int, default=2)
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--cell-px", type=int, default=512)
    ap.add_argument("--grid-render-px", type=int, default=1200)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--video-batch", type=int, default=6)
    ap.add_argument("--video-stride", type=int, default=1)
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--gif", action="store_true")
    ap.add_argument("--no-crop", action="store_true",
                    help="关闭宫格裁剪（调试构图时用；正式跑请保持开启）")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cfg = RenderConfig(
        device=args.device,
        grid_render_px=args.grid_render_px,
        batch_size=args.batch,
        video_batch_size=args.video_batch,
        video_stride=args.video_stride,
        write_video=not args.no_video,
        write_gif=args.gif,
    )
    grid = GridSpec(rows=args.rows, cols=args.cols, cell_px=args.cell_px,
                    crop_to_subject=not args.no_crop)

    print(f"设备 {cfg.device} | 宫格 {grid.rows}×{grid.cols} @ {grid.cell_px}px "
          f"| 机位 {len(cfg.cameras)}")
    out = MotionRenderer(cfg, repo_root=REPO)(args.motion, args.out, grid, args.tag)

    print("\n宫格图:")
    for p in out.grids:
        print(f"  {p}")
    if out.video:
        print(f"视频:   {out.video}")
    if out.gif:
        print(f"GIF:    {out.gif}")
    meta_p = Path(args.out) / f"{args.tag}_render_meta.json"
    meta_p.write_text(json.dumps(out.meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"元信息: {meta_p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
