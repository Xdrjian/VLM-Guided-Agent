"""检视自动拟合出的胶囊体，并核对参考动作上的间隙标定。

拟合错了后面全错，所以这一步要人眼过一遍：半径是否符合解剖学常识
（大腿 ~8-10cm、小臂 ~4-5cm、躯干最粗），长度是否接近骨长。

用法::

    python scripts/inspect_capsules.py
    python scripts/inspect_capsules.py --pairs 25
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from motionedit.collision import build_collision_penalty, fit_capsules  # noqa: E402
from motionedit.motion_io import load_motion                            # noqa: E402
from motionedit.skeleton import Skeleton                                # noqa: E402

DEFAULT_SKEL = REPO / "humoto/human_model/human_model_up_bone_zup.json"
DEFAULT_MOTION = REPO / "Experiment/pipeline/up_bone/A_ref_straight"


def banner(t: str) -> None:
    print(f"\n{'=' * 72}\n{t}\n{'=' * 72}")


def main() -> int:
    ap = argparse.ArgumentParser(description="检视胶囊体拟合与间隙标定")
    ap.add_argument("--skeleton", type=Path, default=DEFAULT_SKEL)
    ap.add_argument("--motion", type=Path, default=DEFAULT_MOTION)
    ap.add_argument("--radius-quantile", type=float, default=0.75)
    ap.add_argument("--exclude-hops", type=int, default=1)
    ap.add_argument("--slack", type=float, default=0.02)
    ap.add_argument("--pairs", type=int, default=20, help="列出最紧的前 N 对")
    args = ap.parse_args()

    skel = Skeleton.from_json(args.skeleton)
    seq = load_motion(args.motion, bone_order=skel.bone_names)

    caps = fit_capsules(args.skeleton, skel, radius_quantile=args.radius_quantile)
    banner(f"胶囊体拟合（{caps.count} 个，radius_quantile={args.radius_quantile}）")
    print(f"  {'骨骼':<30}{'长度cm':>9}{'半径cm':>9}")
    for row in caps.summary():
        print(f"  {row['name']:<30}{row['length_cm']:9.1f}{row['radius_cm']:9.1f}")

    world = skel.forward_kinematics(seq.pose)
    pen = build_collision_penalty(caps, skel, world,
                                  exclude_hops=args.exclude_hops, slack_m=args.slack)

    banner(f"参考动作上的间隙标定（{pen.num_pairs} 对，排除树距 ≤ {args.exclude_hops}）")
    clear = pen.clearance(world)
    mins = clear.min(dim=0).values
    order = torch.argsort(mins)
    print(f"  最紧的 {args.pairs} 对：")
    print(f"  {'胶囊对':<56}{'最小间隙cm':>12}")
    for p in order[: args.pairs].tolist():
        name = (f"{caps.names[int(pen.pair_i[p])]} × "
                f"{caps.names[int(pen.pair_j[p])]}")
        print(f"  {name:<56}{float(mins[p]) * 100:12.1f}")

    neg = int((mins < 0).sum())
    print(f"\n  参考动作上已重叠的对：{neg}/{pen.num_pairs}"
          f"（重叠不等于错误——非相邻部位贴合时胶囊会轻微相交，"
          f"自标定正是为了把这些吸收掉）")

    # 恒等性：参考动作自身的惩罚必须为 0，否则标定逻辑有误
    value = float(pen(world))
    print(f"  参考动作上的碰撞惩罚：{value:.6f}  "
          f"{'✓ 为零，标定正确' if value < 1e-9 else '✗ 非零，标定有误'}")

    left = caps.names.index("mixamorig:LeftLeg") if "mixamorig:LeftLeg" in caps.names else None
    right = caps.names.index("mixamorig:RightLeg") if "mixamorig:RightLeg" in caps.names else None
    if left is not None and right is not None:
        hit = ((pen.pair_i == left) & (pen.pair_j == right)) | \
              ((pen.pair_i == right) & (pen.pair_j == left))
        if bool(hit.any()):
            p = int(torch.nonzero(hit)[0])
            print(f"\n  左右小腿这一对：margin={float(pen.margin[p]) * 100:.1f}cm，"
                  f"参考动作最小间隙={float(mins[p]) * 100:.1f}cm")
        else:
            print("\n  ⚠ 左右小腿未进入检测对，请检查 exclude_hops")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
