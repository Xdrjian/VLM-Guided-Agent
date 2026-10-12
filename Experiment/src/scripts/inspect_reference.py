#!/usr/bin/env python3
"""体检脚本：对一段动作序列打印完整的结构化摘要。

用途
----
1. **迁移验证**：代码从 Windows 开发机迁到 Ubuntu 目标机后，跑一次本脚本，
   把输出与 ``README.md`` 第 4 节的"预期输出"逐行比对。数值一致即说明
   迁移无误（依赖版本、数据路径、浮点行为都正常）。
2. **日常诊断**：对求解器产出的新动作跑一次，快速判断结果是否合理
   （步频是否失真、是否出现 ROM 越界、托盘是否脱手）。

用法
----
    python scripts/inspect_reference.py                    # 默认检查 A_ref_straight
    python scripts/inspect_reference.py --motion <路径>     # 检查指定序列
    python scripts/inspect_reference.py --json out.json    # 同时导出 JSON
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]

ROOT = Path(__file__).resolve().parents[1]      # Experiment/src
REPO = ROOT.parents[1]                          # 仓库根目录
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from motionedit.features import extract_features                    # noqa: E402
from motionedit.motion_io import load_motion                        # noqa: E402
from motionedit.phase import sample_frames_by_phase                 # noqa: E402
from motionedit.rom import ROM_LIMITS, build_rom_penalty, measure_swing_twist  # noqa: E402
from motionedit.skeleton import CORE_BONES, Skeleton                # noqa: E402
from motionedit.tray import solve_tray_pose                         # noqa: E402

DEFAULT_SKEL = REPO / "humoto/human_model/human_model_up_bone_zup.json"
DEFAULT_MOTION = REPO / "Experiment/pipeline/up_bone/A_ref_straight"


def section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def pad(text: str, width: int) -> str:
    """按显示宽度补空格（中日韩字符占 2 列）。"""
    w = sum(2 if ord(c) > 0x2E80 else 1 for c in text)
    return text + " " * max(width - w, 0)


def main() -> int:
    ap = argparse.ArgumentParser(description="动作序列体检")
    ap.add_argument("--motion", type=Path, default=DEFAULT_MOTION, help="pkl 文件或其所在目录")
    ap.add_argument("--skeleton", type=Path, default=DEFAULT_SKEL, help="静止骨架 json")
    ap.add_argument("--tray", type=str, default="draw_organizer_tray", help="托盘物体名")
    ap.add_argument("--json", type=Path, default=None, help="把摘要导出为 JSON")
    args = ap.parse_args()

    section("0. 环境")
    print(f"  Python        : {sys.version.split()[0]}")
    print(f"  torch         : {torch.__version__}")
    print(f"  CUDA 可用     : {torch.cuda.is_available()}"
          + (f"  ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else ""))
    print(f"  骨架文件      : {args.skeleton}")
    print(f"  动作文件      : {args.motion}")

    if not args.skeleton.exists():
        print(f"\n[错误] 骨架文件不存在：{args.skeleton}")
        return 2
    if not args.motion.exists():
        print(f"\n[错误] 动作文件不存在：{args.motion}")
        return 2

    skel = Skeleton.from_json(args.skeleton)
    seq = load_motion(args.motion, bone_order=skel.bone_names)

    section("1. 数据结构")
    print(f"  骨骼数        : {skel.num_bones}")
    print(f"  帧数          : {seq.num_frames}")
    print(f"  帧率 / 时长   : {seq.fps:.1f} fps / {seq.duration:.3f} s")
    print(f"  pose 张量形状 : {tuple(seq.pose.shape)}")
    print(f"  附带物体      : {list(seq.objects)}")
    print(f"  可优化骨骼    : {int(skel.dof_mask(CORE_BONES).sum())} / {skel.num_bones}")

    section("2. 骨架尺度（FK 正确性的第一道校验）")
    joints = skel.joint_positions(seq.pose)
    rest_hip = float(skel.rest[skel.index("mixamorig:Hips"), 2, 3])
    head_h = float(skel.rest[skel.index("mixamorig:HeadTop_End"), 2, 3])
    print(f"  Hips 静止高度 : {rest_hip:.4f} m")
    print(f"  角色总高      : {head_h:.4f} m")
    print(f"  运动中足部最低: {float(joints[:, skel.index('mixamorig:LeftToe_End'), 2].min()):.4f} m")

    section("3. 步态与生物力学特征")
    feats, curves = extract_features(seq, skel, tray_name=args.tray)
    d = feats.to_dict()
    labels = {
        "cadence_steps_per_s": ("步频", "步/秒"),
        "cycle_frames": ("步态周期", "帧"),
        "double_support_ratio": ("双支撑相占比", ""),
        "stride_length_m": ("步长", "m"),
        "knee_flex_left_max": ("左膝屈曲峰值", "°"),
        "knee_flex_right_max": ("右膝屈曲峰值", "°"),
        "trunk_lean_mean": ("躯干前倾均值", "°"),
        "elbow_flex_left_mean": ("左肘屈曲均值", "°"),
        "elbow_flex_right_mean": ("右肘屈曲均值", "°"),
        "com_vertical_range_m": ("重心垂直起伏", "m"),
        "tray_chest_distance_mean": ("托盘-胸骨距离", "m"),
    }
    for k, (name, unit) in labels.items():
        v = d.get(k)
        print(f"  {pad(name, 18)}: {'N/A' if v is None else f'{v:.3f}'} {unit}")
    print(f"  {pad('相位识别成功', 18)}: {feats.phase_valid}")
    print(f"  {pad('相位采样 8 帧', 18)}: {sample_frames_by_phase(curves.gait.phase, 8)}")

    section("4. 关节活动度（swing-twist，越界量应全为 0）")
    pen = build_rom_penalty(skel)
    st = measure_swing_twist(seq.pose, skel)
    viol = pen.per_bone(seq.pose)
    print(f"  {'关节':<22}{'swing 实测':>20}{'上限':>6}{'twist 实测':>20}{'上限':>6}{'越界':>8}")
    for bone, (s, t) in st.items():
        lim = ROM_LIMITS[bone]
        short = bone.replace("mixamorig:", "")
        print(f"  {short:<22}"
              f"{float(s.min()):8.1f} ..{float(s.max()):7.1f}"
              f"{lim.max_swing:8.0f}"
              f"{float(t.min()):9.1f} ..{float(t.max()):7.1f}"
              f"{lim.max_twist:8.0f}"
              f"{viol[bone]:8.2f}")
    total = float(pen(seq.pose))
    print(f"\n  ROM 总惩罚: {total:.4f}   （参考动作应 ≈ 0）")
    if total > 1.0:
        print("  [警告] 参考动作出现 ROM 越界，说明限位表或坐标约定有问题，请勿继续优化。")

    section("5. 托盘位姿一致性")
    lh = joints[:, skel.index("mixamorig:LeftHand")]
    rh = joints[:, skel.index("mixamorig:RightHand")]
    tray = solve_tray_pose(lh, rh)
    q_norm_err = float((tray[:, :4].norm(dim=-1) - 1.0).abs().max())
    print(f"  托盘轨道形状  : {tuple(tray.shape)}")
    print(f"  四元数归一化误差: {q_norm_err:.2e}   （应 < 1e-5）")
    if args.tray in seq.objects:
        orig = seq.objects[args.tray]
        print(f"  与原始托盘位置偏差: 均值 "
              f"{float((tray[:, 4:] - orig[:, 4:]).norm(dim=-1).mean()):.4f} m")

    section("总结")
    ok = feats.phase_valid and total < 1.0 and q_norm_err < 1e-5
    print("  [通过] 各项指标正常，可继续后续流程。" if ok
          else "  [失败] 存在异常项，请对照 README 第 4 节排查。")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(
            {"features": d, "rom_total": total, "rom_violation": viol},
            ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  摘要已导出    : {args.json}")

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
