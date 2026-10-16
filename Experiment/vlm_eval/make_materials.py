#!/usr/bin/env python
"""一键生成 VLM / prompt 测试所需的全部视频素材。

与 ``Experiment/pipeline`` 完全隔离：所有产物落在本目录的 ``materials/`` 下，
不污染正式实验的证据包。

**文件命名纪律**：视频一律命名为 ``video.mp4``，语义信息只体现在**目录名**上。
目录名便于脚本定位，而上传给模型的文件名不含任何暗示
（实测 Gemini 的 ``files.upload`` 不回传 display_name，但不值得赌）。

产出结构::

    materials/
      a_ref/         video.mp4  motion.pkl  meta.json   A_ref 动捕真值，未经求解器
      knee_low/      ...  knee_flex_scale = 1.00（经求解器，渲染风格与其余一致）
      knee_mid/      ...  knee_flex_scale = 1.30
      knee_high/     ...  knee_flex_scale = 1.60
      super_heavy/   ...  七参数全部推到较极端但合法的取值
      chaotic/       ...  人为注入关节噪声，故意不自然
      index.json     每份素材的参数、帧数、时长、实测特征

用法（Linux，需 CUDA + pytorch3d）::

    cd Experiment/vlm_eval
    python make_materials.py
    python make_materials.py --only chaotic --force   # 只重做某一份
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]

HERE = Path(__file__).resolve().parent
SRC = HERE.parents[0] / "src"
REPO = HERE.parents[1]
sys.path.insert(0, str(SRC))

import torch  # noqa: E402

from motionedit.features import extract_features                 # noqa: E402
from motionedit.motion_io import load_motion, save_motion         # noqa: E402
from motionedit.params import GaitParams                          # noqa: E402
from motionedit.pipeline import MotionSolveFn                     # noqa: E402
from motionedit.prompts import VideoSpec                          # noqa: E402
from motionedit.renderer import MotionRenderer, RenderConfig      # noqa: E402
from motionedit.rotation import quaternion_multiply, rotvec_to_quaternion  # noqa: E402
from motionedit.skeleton import CORE_BONES                        # noqa: E402
from motionedit.solver import SolverConfig                        # noqa: E402
from motionedit.tray import update_tray_in_sequence               # noqa: E402

DEFAULT_SKEL = REPO / "humoto/human_model/human_model_up_bone_zup.json"
DEFAULT_MOTION = REPO / "Experiment/pipeline/up_bone/A_ref_straight"
TRAY = "draw_organizer_tray"


@dataclass
class Material:
    name: str
    purpose: str
    params: Optional[GaitParams] = None   # None = 不经求解器
    corrupt: bool = False


#: ``knee_low`` 用恒等参数但**仍走求解器**，这样它与 mid/high 的唯一差异就是
#: 膝屈曲缩放本身，而不是"过没过求解器"。a_ref 则刻意不走，用作自然度正参考点。
MATERIALS: List[Material] = [
    Material("a_ref", "动捕真值，未经任何处理；自然度正参考点"),
    Material("knee_low", "单调性测试低档：knee_flex_scale=1.00",
             GaitParams()),
    Material("knee_mid", "单调性测试中档：knee_flex_scale=1.30",
             GaitParams(knee_flex_scale=1.30)),
    Material("knee_high", "单调性测试高档：knee_flex_scale=1.60",
             GaitParams(knee_flex_scale=1.60)),
    Material("super_heavy", "七参数推到较极端但合法；预期有一定不自然",
             GaitParams(knee_flex_scale=1.60, trunk_lean_offset=20.0,
                        cadence_scale=0.60, double_support_offset=0.20,
                        stride_length_scale=0.50, elbow_flex_offset=35.0,
                        tray_chest_distance_scale=0.55)),
    Material("chaotic", "人为注入关节噪声，故意不自然；自然度负参考点",
             corrupt=True),
]


# --------------------------------------------------------------------------- #
def corrupt_motion(seq, skel, seed: int = 0, amp_deg: float = 22.0,
                   period: int = 3):
    """往核心骨骼注入高频旋转噪声，制造人眼可辨的"不自然"。

    不是随便加噪声就行：低频大幅噪声看起来像"另一种走法"，仍然自然；
    要的是**高频抖动 + 关节弹跳**，即每隔几帧换一次方向，幅度足以越过 ROM。
    托盘仍按抓握关系重建，使不自然完全归因于人体运动而非物体穿帮。
    """
    g = torch.Generator().manual_seed(seed)
    pose = seq.pose.clone()
    T = pose.shape[0]
    idx = [skel.index(b) for b in CORE_BONES if b in skel.bone_names]

    # 每 period 帧换一次随机方向，再线性插值到逐帧，得到高频但不撕裂的抖动
    n_key = max(T // period, 2)
    key = torch.randn(n_key, len(idx), 3, generator=g)
    key = key / key.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    key = key * torch.deg2rad(torch.tensor(float(amp_deg)))
    t = torch.linspace(0, n_key - 1, T)
    lo = t.floor().long().clamp(max=n_key - 1)
    hi = (lo + 1).clamp(max=n_key - 1)
    w = (t - lo.to(t.dtype)).view(T, 1, 1)
    rv = key[lo] * (1 - w) + key[hi] * w                      # [T, K, 3]

    dq = rotvec_to_quaternion(rv)
    pose[:, idx, :4] = quaternion_multiply(pose[:, idx, :4], dq)
    out = seq.clone()
    out.pose = pose
    return out


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description="生成 VLM 测试素材")
    ap.add_argument("--out", type=Path, default=HERE / "materials")
    ap.add_argument("--motion", type=Path, default=DEFAULT_MOTION)
    ap.add_argument("--skeleton", type=Path, default=DEFAULT_SKEL)
    ap.add_argument("--only", nargs="*", default=None, help="只生成指定素材")
    ap.add_argument("--force", action="store_true", help="已存在也重做")
    ap.add_argument("--adam-iters", type=int, default=600)
    ap.add_argument("--lbfgs-iters", type=int, default=60)
    ap.add_argument("--video-stride", type=int, default=1)
    ap.add_argument("--render-batch", type=int, default=8)
    ap.add_argument("--video-batch", type=int, default=6)
    ap.add_argument("--chaos-amp-deg", type=float, default=28.0,
                    help="噪声幅度。本地实测：10°→ROM 0.1（仍自然），"
                         "22°→ROM 9.5，28°→明显越界，35°→ROM 132.7")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    wanted = set(args.only) if args.only else {m.name for m in MATERIALS}

    print(f"设备 {args.device} | 输出 {out_root}")
    solve_fn = MotionSolveFn(
        args.motion, args.skeleton, tray_name=TRAY,
        solver_cfg=SolverConfig(adam_iters=args.adam_iters,
                                lbfgs_iters=args.lbfgs_iters,
                                device=args.device,
                                log_every=max(args.adam_iters // 3, 1)))
    renderer = MotionRenderer(
        RenderConfig(device=args.device, video_stride=args.video_stride,
                     batch_size=args.render_batch,
                     video_batch_size=args.video_batch, write_video=True),
        repo_root=REPO, skeleton=solve_fn.skel)
    spec = VideoSpec()

    index: Dict[str, dict] = {}
    idx_path = out_root / "index.json"
    if idx_path.exists():
        index = json.loads(idx_path.read_text(encoding="utf-8"))

    for m in MATERIALS:
        d = out_root / m.name
        video = d / "video.mp4"
        if m.name not in wanted:
            continue
        if video.exists() and not args.force:
            print(f"[skip] {m.name}（已存在，加 --force 可重做）")
            continue

        print(f"\n{'=' * 70}\n{m.name}：{m.purpose}\n{'=' * 70}")
        d.mkdir(parents=True, exist_ok=True)
        work = d / "_render"
        shutil.rmtree(work, ignore_errors=True)

        if m.corrupt:
            seq = corrupt_motion(solve_fn.seq, solve_fn.skel,
                                 amp_deg=args.chaos_amp_deg)
            if solve_fn.grip is not None:
                seq = update_tray_in_sequence(seq, solve_fn.skel, solve_fn.grip)
            motion_path = save_motion(seq, d / "motion.pkl")
            params = None
        elif m.params is None:
            motion_path = d / "motion.pkl"
            shutil.copy2(_pkl_of(args.motion), motion_path)
            params = None
        else:
            res = solve_fn(m.params, d)
            motion_path = res.motion_path
            params = m.params.to_dict()
            if res.warnings:
                for w in res.warnings:
                    print(f"  ⚠ {w}")

        renderer(motion_path, work, spec, tag="m")
        src = work / "m_video.mp4"
        if not src.exists():
            raise FileNotFoundError(f"渲染未产出视频：{src}")
        shutil.move(str(src), str(video))
        grids = d / "grids"
        shutil.rmtree(grids, ignore_errors=True)
        (work / "images").rename(grids) if (work / "images").exists() else None
        shutil.rmtree(work, ignore_errors=True)

        seq2 = load_motion(motion_path, bone_order=solve_fn.skel.bone_names)
        feats, _ = extract_features(seq2, solve_fn.skel, tray_name=TRAY)
        index[m.name] = {
            "purpose": m.purpose,
            "params": params,
            "corrupted": m.corrupt,
            "num_frames": feats.num_frames,
            "duration_s": feats.duration_s,
            "knee_flex_left_max": feats.knee_flex_left_max,
            "trunk_lean_mean": feats.trunk_lean_mean,
            "cadence_steps_per_s": feats.cadence_steps_per_s,
            "stride_length_m": feats.stride_length_m,
            "video": str(video.relative_to(out_root)),
            "size_mb": round(video.stat().st_size / 1e6, 2),
        }
        print(f"  -> {video}  ({index[m.name]['size_mb']} MB, "
              f"{feats.num_frames} 帧 / {feats.duration_s:.2f}s, "
              f"左膝峰值 {feats.knee_flex_left_max:.1f}°)")
        idx_path.write_text(json.dumps(index, ensure_ascii=False, indent=2),
                            encoding="utf-8")

    print(f"\n{'=' * 70}\n素材清单\n{'=' * 70}")
    print(f"{'名称':<14}{'帧数':>6}{'时长s':>8}{'左膝峰值':>10}{'步频':>8}{'步长m':>8}")
    for k, v in index.items():
        print(f"{k:<14}{v['num_frames']:6d}{v['duration_s']:8.2f}"
              f"{v['knee_flex_left_max']:10.1f}{v['cadence_steps_per_s']:8.2f}"
              f"{v['stride_length_m']:8.3f}")
    print(f"\n清单已写入 {idx_path}")
    print("注意：视频一律命名为 video.mp4，语义只在目录名上。")
    return 0


def _pkl_of(p: Path) -> Path:
    p = Path(p)
    return p / f"{p.name}.pkl" if p.is_dir() else p


if __name__ == "__main__":
    raise SystemExit(main())
