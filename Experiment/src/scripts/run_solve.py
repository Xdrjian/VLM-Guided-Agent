#!/usr/bin/env python3
"""端到端演示：给定一组参数，求解出新动作并落盘。

这是第 1~3 层的完整串联，也是外层 agent 循环将要调用的核心流程：

    参考动作 → 特征提取 → 目标曲线 → 求解 → 托盘跟随 → 保存

用法
----
    # 用内置的"负重 30kg"参数组合
    python scripts/run_solve.py --preset heavy --out ../pipeline/solved/heavy

    # 自定义参数
    python scripts/run_solve.py --params '{"knee_flex_scale":1.3,"trunk_lean_offset":12}' \\
                                --out ../pipeline/solved/custom

    # 快速冒烟（迭代数压低）
    python scripts/run_solve.py --preset heavy --adam-iters 100 --lbfgs-iters 20
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

from motionedit.features import extract_features              # noqa: E402
from motionedit.motion_io import MotionSequence, load_motion, save_motion  # noqa: E402
from motionedit.params import GaitParams                      # noqa: E402
from motionedit.skeleton import Skeleton                      # noqa: E402
from motionedit.solver import SolverConfig, solve             # noqa: E402
from motionedit.targets import build_targets                  # noqa: E402
from motionedit.tray import update_tray_in_sequence           # noqa: E402

DEFAULT_SKEL = REPO / "humoto/human_model/human_model_up_bone_zup.json"
DEFAULT_MOTION = REPO / "Experiment/pipeline/up_bone/A_ref_straight"
TRAY = "draw_organizer_tray"

#: 内置参数预设。``heavy`` 是一组符合负重步态生物力学直觉的初始猜测，
#: 正式实验中这一组应由 LLM 给出，此处仅作演示与回归基线。
PRESETS = {
    "identity": GaitParams(),
    "heavy": GaitParams(
        knee_flex_scale=1.25, trunk_lean_offset=8.0, cadence_scale=0.90,
        double_support_offset=0.06, stride_length_scale=0.80,
        elbow_flex_offset=10.0, tray_chest_distance_scale=0.85),
    "very_heavy": GaitParams(
        knee_flex_scale=1.40, trunk_lean_offset=15.0, cadence_scale=0.80,
        double_support_offset=0.12, stride_length_scale=0.65,
        elbow_flex_offset=18.0, tray_chest_distance_scale=0.75),
}


def banner(t: str) -> None:
    print(f"\n{'=' * 70}\n{t}\n{'=' * 70}")


def main() -> int:
    ap = argparse.ArgumentParser(description="端到端求解演示")
    ap.add_argument("--preset", choices=sorted(PRESETS), default="heavy")
    ap.add_argument("--params", type=str, default=None, help="JSON 参数，覆盖 preset")
    ap.add_argument("--motion", type=Path, default=DEFAULT_MOTION)
    ap.add_argument("--skeleton", type=Path, default=DEFAULT_SKEL)
    ap.add_argument("--out", type=Path, default=None, help="输出目录（不给则不落盘）")
    ap.add_argument("--adam-iters", type=int, default=600)
    ap.add_argument("--lbfgs-iters", type=int, default=60)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    params = PRESETS[args.preset]
    if args.params:
        params = GaitParams.from_dict(json.loads(args.params))
    params = params.clipped()

    banner("1. 加载")
    skel = Skeleton.from_json(args.skeleton)
    seq = load_motion(args.motion, bone_order=skel.bone_names)
    print(f"  设备 {args.device} | {seq.num_frames} 帧 | {skel.num_bones} 骨骼")
    print(f"  参数 {json.dumps(params.to_dict(), ensure_ascii=False)}")

    banner("2. 参考特征")
    ref_feats, curves = extract_features(seq, skel, tray_name=TRAY)
    print(f"  步频 {ref_feats.cadence_steps_per_s:.2f} 步/秒 | "
          f"左膝峰值 {ref_feats.knee_flex_left_max:.1f}° | "
          f"躯干 {ref_feats.trunk_lean_mean:.1f}°")

    banner("3. 目标曲线")
    target = build_targets(curves, params, ref_feats.tray_chest_distance_mean)
    print(json.dumps(target.summary(), ensure_ascii=False, indent=2))

    banner("4. 求解")
    res = solve(seq, skel, target, cfg=SolverConfig(
        adam_iters=args.adam_iters, lbfgs_iters=args.lbfgs_iters,
        device=args.device, log_every=max(args.adam_iters // 6, 1)))
    print(json.dumps(res.summary(), ensure_ascii=False, indent=2))

    banner("5. 结果对比（参考 → 目标 → 实际）")
    out = MotionSequence(pose=res.pose.cpu(), bone_names=seq.bone_names,
                         objects={k: v.clone() for k, v in seq.objects.items()},
                         fps=seq.fps, meta=seq.meta,
                         object_models=seq.object_models)
    out = update_tray_in_sequence(out, skel, tray_name=TRAY)
    new_feats, _ = extract_features(out, skel, tray_name=TRAY)

    rows = [
        ("左膝屈曲峰值 (°)", ref_feats.knee_flex_left_max,
         float(target.knee_flex_left.max()), new_feats.knee_flex_left_max),
        ("躯干前倾均值 (°)", ref_feats.trunk_lean_mean,
         float(target.trunk_lean.mean()), new_feats.trunk_lean_mean),
        ("左肘屈曲均值 (°)", ref_feats.elbow_flex_left_mean,
         float(target.elbow_flex_left.mean()), new_feats.elbow_flex_left_mean),
        ("步频 (步/秒)", ref_feats.cadence_steps_per_s,
         ref_feats.cadence_steps_per_s * params.cadence_scale,
         new_feats.cadence_steps_per_s),
        ("托盘-胸骨距离 (m)", ref_feats.tray_chest_distance_mean,
         target.tray_chest_distance, new_feats.tray_chest_distance_mean),
    ]
    print(f"  {'指标':<20}{'参考':>10}{'目标':>10}{'实际':>10}{'误差':>10}")
    for name, a, b, cval in rows:
        if a is None or cval is None:
            continue
        print(f"  {name:<20}{a:10.2f}{b:10.2f}{cval:10.2f}{abs(cval - b):10.2f}")

    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        pkl = save_motion(out, args.out / f"{args.out.name}.pkl")
        (args.out / "solve_report.json").write_text(json.dumps({
            "params": params.to_dict(),
            "target": target.summary(),
            "solve": res.summary(),
            "ref_features": ref_feats.to_dict(),
            "new_features": new_feats.to_dict(),
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n  已保存: {pkl}")
        print(f"  已保存: {args.out / 'solve_report.json'}")

    return 0 if res.converged else 1


if __name__ == "__main__":
    raise SystemExit(main())
