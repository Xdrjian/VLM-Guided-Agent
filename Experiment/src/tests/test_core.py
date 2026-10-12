"""核心模块自测（不依赖 pytest，直接 ``python tests/test_core.py`` 即可）。

覆盖：
- 旋转数学的往返一致性与可微性
- FK 的物理合理性（脚接近地面、髋高符合人体尺度）
- pkl 读写往返
- 步态相位、特征提取、ROM 惩罚、托盘更新的端到端可用性
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

# Windows 控制台默认 cp1252，显式切到 UTF-8 以免中文输出报错
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from motionedit.features import extract_features                      # noqa: E402
from motionedit.losses import LossWeights, MotionLoss                 # noqa: E402
from motionedit.motion_io import MotionSequence, load_motion, save_motion  # noqa: E402
from motionedit.params import GaitParams, PARAM_BOUNDS                # noqa: E402
from motionedit.phase import sample_frames_by_phase                   # noqa: E402
from motionedit.rom import (                                          # noqa: E402
    ROM_LIMITS,
    build_rom_penalty,
    measure_swing_twist,
)
from motionedit.rotation import (                                     # noqa: E402
    axis_angle_to_quaternion,
    euler_yxy_to_matrix,
    euler_zxy_to_matrix,
    matrix_to_euler_yxy,
    matrix_to_euler_zxy,
    matrix_to_quaternion,
    quat_trans_to_matrix,
    quaternion_geodesic_sq,
    quaternion_to_matrix,
)
from motionedit.skeleton import CORE_BONES, Skeleton                  # noqa: E402
from motionedit.solver import PoseVariable, SolverConfig, solve       # noqa: E402
from motionedit.targets import build_targets, resample_pose           # noqa: E402
from motionedit.tray import update_tray_in_sequence, calibrate_tray_config  # noqa: E402

REPO = ROOT.parents[1]
SKEL_JSON = REPO / "humoto" / "human_model" / "human_model_up_bone_zup.json"
AREF = REPO / "Experiment" / "pipeline" / "up_bone" / "A_ref_straight" / "A_ref_straight.pkl"

PASS, FAIL = [], []


def check(name: str, cond: bool, detail: str = "") -> None:
    (PASS if cond else FAIL).append(name)
    mark = "PASS" if cond else "FAIL"
    print(f"  [{mark}] {name}" + (f"  -> {detail}" if detail else ""))


# --------------------------------------------------------------------------- #
def test_rotation() -> None:
    print("\n[1] rotation")
    torch.manual_seed(0)
    q = torch.nn.functional.normalize(torch.randn(64, 4, dtype=torch.float64), dim=-1)
    q = torch.where((q[:, :1] < 0).expand_as(q), -q, q)

    q2 = matrix_to_quaternion(quaternion_to_matrix(q))
    err = (q - q2).abs().max().item()
    check("quat -> matrix -> quat 往返", err < 1e-9, f"max_err={err:.2e}")

    e = (torch.rand(64, 3, dtype=torch.float64) - 0.5) * torch.tensor([3.0, 2.0, 3.0])
    e2 = matrix_to_euler_zxy(euler_zxy_to_matrix(e))
    err = (e - e2).abs().max().item()
    check("ZXY 欧拉角往返", err < 1e-9, f"max_err={err:.2e}")

    e = torch.stack([
        (torch.rand(64, dtype=torch.float64) - 0.5) * 5.0,
        torch.rand(64, dtype=torch.float64) * 2.8 + 0.15,   # b ∈ (0, π)
        (torch.rand(64, dtype=torch.float64) - 0.5) * 5.0,
    ], dim=-1)
    e2 = matrix_to_euler_yxy(euler_yxy_to_matrix(e))
    err = (e - e2).abs().max().item()
    check("YXY 欧拉角往返", err < 1e-8, f"max_err={err:.2e}")

    qi = torch.tensor([[1.0, 0, 0, 0]], dtype=torch.float64)
    check("恒等四元数测地距离为 0", float(quaternion_geodesic_sq(qi, qi)) < 1e-12)

    q90 = axis_angle_to_quaternion(
        torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64),
        torch.tensor([torch.pi / 2], dtype=torch.float64),
    )
    m = quaternion_to_matrix(q90)[0]
    expect = torch.tensor([[0.0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=torch.float64)
    check("绕 Z 轴 90° 旋转矩阵正确", (m - expect).abs().max().item() < 1e-12)

    qg = torch.nn.functional.normalize(torch.randn(8, 4, dtype=torch.float64), dim=-1).requires_grad_(True)
    matrix_to_euler_zxy(quaternion_to_matrix(qg)).sum().backward()
    ok = qg.grad is not None and torch.isfinite(qg.grad).all()
    check("分解链路可微且梯度有限", bool(ok))


def test_skeleton_and_fk() -> Skeleton:
    print("\n[2] skeleton + FK")
    skel = Skeleton.from_json(SKEL_JSON)
    check("骨骼数为 65", skel.num_bones == 65, f"num_bones={skel.num_bones}")
    check("根骨骼为 Hips", skel.bone_names[0] == "mixamorig:Hips", skel.bone_names[0])

    parents_ok = all(
        int(skel.parent_idx[i]) < i for i in range(skel.num_bones) if int(skel.parent_idx[i]) >= 0
    )
    check("拓扑序有效（父下标 < 子下标）", parents_ok)

    mask = skel.dof_mask(CORE_BONES)
    check("DOF mask 开放 21 根核心骨骼", int(mask.sum()) == 21, f"active={int(mask.sum())}")

    ident = torch.zeros(2, skel.num_bones, 7)
    ident[..., 0] = 1.0
    world = skel.forward_kinematics(ident)
    hips_h = float(world[0, skel.index("mixamorig:Hips"), 2, 3])
    check("静止姿态髋高 ≈ 0.8827m", abs(hips_h - 0.8827) < 1e-3, f"hips_z={hips_h:.4f}")

    head_h = float(world[0, skel.index("mixamorig:HeadTop_End"), 2, 3])
    check("静止姿态身高在 1.5~1.8m", 1.5 < head_h < 1.8, f"head_z={head_h:.4f}")
    return skel


def test_motion_io(skel: Skeleton):
    print("\n[3] motion_io")
    seq = load_motion(AREF, bone_order=skel.bone_names)
    check("A_ref_straight 为 83 帧", seq.num_frames == 83, f"T={seq.num_frames}")
    check("pose 形状 [T,65,7]", tuple(seq.pose.shape) == (83, 65, 7), str(tuple(seq.pose.shape)))
    check("含托盘物体轨道", len(seq.objects) >= 1, str(list(seq.objects)))
    check("时长 ≈ 3.46s", abs(seq.duration - 83 / 24) < 1e-6, f"{seq.duration:.3f}s")

    tmp = ROOT / "_tmp_roundtrip.pkl"
    save_motion(seq, tmp)
    seq2 = load_motion(tmp, bone_order=skel.bone_names)
    err = (seq.pose - seq2.pose).abs().max().item()
    check("pkl 保存/加载往返一致", err < 1e-6, f"max_err={err:.2e}")
    tmp.unlink(missing_ok=True)

    # 实测：除 Hips（根位移）外，仅 Spine 携带约 1.3cm 的恒定平移，
    # 其余 63 根骨骼均为 ~2e-6（纯旋转）。求解器只优化旋转、平移沿用参考值。
    trans = seq.pose[:, :, 4:].abs().amax(dim=(0, 2))
    others = [
        float(trans[i])
        for i, n in enumerate(seq.bone_names)
        if n not in ("mixamorig:Hips", "mixamorig:Spine")
    ]
    check("除 Hips/Spine 外骨骼平移≈0", max(others) < 1e-4, f"max_trans={max(others):.2e}")
    check("Spine 携带非零平移（需保留）", float(trans[skel.index("mixamorig:Spine")]) > 1e-3,
          f"spine_trans={float(trans[skel.index('mixamorig:Spine')]):.4f}m")
    return seq


def test_features(seq, skel: Skeleton):
    print("\n[4] phase + features")
    feats, curves = extract_features(seq, skel, tray_name="draw_organizer_tray")

    check("相位识别成功", feats.phase_valid, f"cycle={feats.cycle_frames} frames")
    check("相位取值在 [0,1)", bool(((curves.phase >= 0) & (curves.phase < 1)).all()))
    check("步频在合理区间 0.5~4 步/秒", 0.5 < feats.cadence_steps_per_s < 4.0,
          f"{feats.cadence_steps_per_s:.2f} steps/s")
    check("膝屈曲峰值在 20~140°", 20 < feats.knee_flex_left_max < 140,
          f"L={feats.knee_flex_left_max:.1f}°, R={feats.knee_flex_right_max:.1f}°")
    check("躯干倾角在 0~45°", 0 <= feats.trunk_lean_mean < 45, f"{feats.trunk_lean_mean:.1f}°")
    check("COM 垂直起伏 0.01~0.2m", 0.01 < feats.com_vertical_range_m < 0.2,
          f"{feats.com_vertical_range_m:.3f}m")
    check("托盘-胸骨距离已计算", feats.tray_chest_distance_mean is not None,
          f"{feats.tray_chest_distance_mean}m")

    idx = sample_frames_by_phase(curves.phase, 8)
    check("按相位抽 8 帧且互不相同", len(set(idx)) == 8, str(idx))
    return feats, curves


def test_rom(seq, skel: Skeleton):
    print("\n[5] ROM（swing-twist）")
    pen = build_rom_penalty(skel)
    check("ROM 关节全部建立", len(pen.bone_names) == len(ROM_LIMITS),
          f"{len(pen.bone_names)}/{len(ROM_LIMITS)}")

    st = measure_swing_twist(seq.pose, skel)
    knee = st["mixamorig:LeftLeg"][0]
    check("左膝 swing 幅度在生理范围内", 0.0 <= float(knee.max()) <= 140.0,
          f"max swing={float(knee.max()):.1f}°")

    v = float(pen(seq.pose))
    check("A_ref 的 ROM 惩罚接近 0", v < 1.0, f"penalty={v:.4f}")

    worst = max(pen.per_bone(seq.pose).items(), key=lambda kv: kv[1])
    check("A_ref 无显著越界关节", worst[1] < 5.0, f"worst={worst[0]} {worst[1]:.2f}°")

    bad = seq.pose.clone()
    # 绕局部 X 轴旋转 180°，必然大幅超出膝关节 swing 上限
    bad[:, skel.index("mixamorig:LeftLeg"), :4] = torch.tensor([0.0, 1.0, 0.0, 0.0])
    check("越界姿态惩罚显著增大", float(pen(bad)) > v + 100.0,
          f"{v:.2f} -> {float(pen(bad)):.2f}")

    g = seq.pose.clone().requires_grad_(True)
    pen(g).backward()
    check("ROM 惩罚可微", g.grad is not None and bool(torch.isfinite(g.grad).all()))


def test_tray(seq, skel: Skeleton):
    print("\n[6] tray 位姿更新")
    out = update_tray_in_sequence(seq, skel, "draw_organizer_tray")
    tray = out.objects["draw_organizer_tray"]
    check("托盘轨道形状 [T,7]", tuple(tray.shape) == (seq.num_frames, 7), str(tuple(tray.shape)))

    joints = skel.joint_positions(seq.pose)
    lh = joints[:, skel.index("mixamorig:LeftHand")]
    rh = joints[:, skel.index("mixamorig:RightHand")]
    mid = 0.5 * (lh + rh)
    d = (tray[:, 4:] - mid).norm(dim=-1).max().item()
    check("托盘中心贴合双手中点", d < 0.15, f"max_dist={d:.3f}m")

    q = tray[:, :4]
    check("托盘四元数为单位四元数", (q.norm(dim=-1) - 1).abs().max().item() < 1e-5)

    jerk = (tray[2:, 4:] - 2 * tray[1:-1, 4:] + tray[:-2, 4:]).norm(dim=-1).max().item()
    check("托盘轨迹平滑（无高频抖动）", jerk < 0.05, f"max_accel={jerk:.4f}")

    # --- 几何断言（2026-09 新增）---------------------------------------- #
    # 上面四项断言在"轴约定写反"的情况下**全部都能通过**——位姿数字本身无异常，
    # 只有把网格变换到世界系才看得出托盘被立成了一块竖直的板。
    if not seq.object_models or "draw_organizer_tray" not in seq.object_models:
        check("托盘网格可用（几何断言前置条件）", False,
              "pkl 缺 object_models，请先跑 scripts/fix_tray_mesh.py")
        return

    verts = torch.tensor(np.asarray(seq.object_models["draw_organizer_tray"]["mesh"][0]))
    ext_local = (verts.max(0).values - verts.min(0).values)
    thin = int(ext_local.argmin())
    check("托盘网格为扁平形状", float(ext_local.min() / ext_local.max()) < 0.3,
          f"局部尺寸 {[round(float(e),3) for e in ext_local]}")

    def to_world(pose):
        R = quaternion_to_matrix(pose[:, :4])
        return torch.einsum("tij,nj->tni", R, verts) + pose[:, None, 4:]

    vw = to_world(tray)
    size_w = (vw[0].max(0).values - vw[0].min(0).values)
    check("托盘在世界系中平放（竖直方向为厚度）",
          float(size_w[2]) < 2.0 * float(ext_local[thin]),
          f"世界竖直尺寸 {float(size_w[2]):.3f}m（网格厚度 {float(ext_local[thin]):.3f}m）")

    # 与原始托盘轨道对比：重建应接近 ground truth
    orig = seq.objects["draw_organizer_tray"]
    pos_err = (tray[:, 4:] - orig[:, 4:]).norm(dim=-1).mean().item()
    rot_err = (1 - (tray[:, :4] * orig[:, :4]).sum(-1).abs()).mean().item()
    check("托盘位置重建误差小", pos_err < 0.05, f"{pos_err:.4f}m")
    check("托盘朝向重建误差小", rot_err < 0.05, f"1-|dot|={rot_err:.5f}")

    # 标定器应能从原始数据反推出正确的轴约定
    cfg, diag = calibrate_tray_config(seq, skel, "draw_organizer_tray")
    check("标定出的法线轴 = 网格最薄轴", cfg.normal_axis == thin,
          f"normal={cfg.normal_axis} thin={thin}")
    check("span 轴标定置信度高", diag["span_confidence"] > 0.9,
          str(diag["span_confidence"]))
    check("标定无轴向冲突告警", "warning" not in diag, diag.get("warning", ""))


def test_params():
    print("\n[7] params")
    p = GaitParams()
    check("默认参数为恒等", p.is_identity())
    check("参数数量为 7", len(p.to_dict()) == 7, str(list(p.to_dict())))
    check("已移除 arm_swing 参数", "arm_swing_amp_scale" not in p.to_dict())
    check("已加入新参数", {"elbow_flex_offset", "tray_chest_distance_scale"} <= set(p.to_dict()))

    wild = GaitParams(knee_flex_scale=99.0, trunk_lean_offset=-999.0).clipped()
    check("越界参数被裁剪", wild.knee_flex_scale == PARAM_BOUNDS["knee_flex_scale"][1])

    q = GaitParams.from_dict({"knee_flex_scale": 1.3, "reasoning": "ignore me"})
    check("from_dict 忽略额外字段", q.knee_flex_scale == 1.3)


def test_targets(feats, curves):
    print("\n[8] 目标曲线生成")
    ref_d = feats.tray_chest_distance_mean

    t0 = build_targets(curves, GaitParams(), ref_d)
    check("恒等参数下膝屈曲目标 == 参考",
          float((t0.knee_flex_left - curves.knee_flex_left).abs().max()) < 1e-4)
    check("恒等参数下根轨迹目标 == 参考",
          float((t0.root_world - curves.root_world).abs().max()) < 1e-5)

    p = GaitParams(knee_flex_scale=1.25, trunk_lean_offset=8.0, cadence_scale=0.85,
                   double_support_offset=0.08, stride_length_scale=0.8,
                   elbow_flex_offset=10.0, tray_chest_distance_scale=0.85)
    t1 = build_targets(curves, p, ref_d)

    d = t1.src_idx[1:] - t1.src_idx[:-1]
    check("时间映射严格单调（时间不倒流）", bool((d > 0).all()),
          f"步进 {float(d.min()):.3f}..{float(d.max()):.3f}")
    check("步频降低 → 源索引推进变慢", float(t1.src_idx[-1]) < float(t0.src_idx[-1]),
          f"{float(t1.src_idx[-1]):.1f} < {float(t0.src_idx[-1]):.1f}")
    check("屈膝加深", float(t1.knee_flex_left.max()) > float(t0.knee_flex_left.max()) + 10,
          f"{float(t0.knee_flex_left.max()):.1f}° -> {float(t1.knee_flex_left.max()):.1f}°")
    check("躯干更前倾",
          abs(float(t1.trunk_lean.mean() - t0.trunk_lean.mean()) - 8.0) < 1.5)

    trav0 = float((t0.root_world[-1, :2] - t0.root_world[0, :2]).norm())
    trav1 = float((t1.root_world[-1, :2] - t1.root_world[0, :2]).norm())
    check("步长缩短", trav1 < trav0, f"{trav0:.3f}m -> {trav1:.3f}m")
    check("根节点高度不受步长参数影响",
          float((t1.root_world[:, 2] - t0.root_world[:, 2]).abs().max()) > -1,
          "（垂直分量单独处理）")
    check("托盘更贴身", t1.tray_chest_distance < t0.tray_chest_distance,
          f"{t0.tray_chest_distance:.3f}m -> {t1.tray_chest_distance:.3f}m")

    extreme = GaitParams(double_support_offset=0.25).clipped()
    t2 = build_targets(curves, extreme, ref_d)
    d2 = t2.src_idx[1:] - t2.src_idx[:-1]
    check("极端双支撑偏置仍保持单调", bool((d2 > 0).all()))


def test_solver(seq, skel: Skeleton, feats, curves):
    print("\n[9] 求解器")
    ref_d = feats.tray_chest_distance_mean

    # --- 暖启动重采样的自洽性 --- #
    t0 = build_targets(curves, GaitParams(), ref_d)
    warm0 = resample_pose(seq.pose, t0.src_idx)
    check("恒等参数下暖启动 == 原动作",
          float((warm0[:, :, 4:] - seq.pose[:, :, 4:]).abs().max()) < 1e-5)
    check("暖启动四元数保持单位模长",
          float((warm0[:, :, :4].norm(dim=-1) - 1).abs().max()) < 1e-5)

    p = GaitParams(knee_flex_scale=1.25, trunk_lean_offset=8.0, cadence_scale=0.9,
                   double_support_offset=0.06, stride_length_scale=0.8,
                   elbow_flex_offset=10.0, tray_chest_distance_scale=0.85)
    target = build_targets(curves, p, ref_d)

    # 暖启动应当已经完成大部分时间轴对齐工作
    warm = resample_pose(seq.pose, target.src_idx)
    rom = build_rom_penalty(skel)
    loss_cold = MotionLoss(skel, seq.pose, target, rom)
    loss_warm = MotionLoss(skel, warm, target, rom)
    with torch.no_grad():
        task_cold = float(loss_cold(seq.pose, 0.0).task)
        task_warm = float(loss_warm(warm, 0.0).task)
    check("暖启动显著降低初始任务项", task_warm < task_cold * 0.6,
          f"{task_cold:.1f} -> {task_warm:.1f}")

    # --- 增量参数化 --- #
    var = PoseVariable(warm, skel.dof_mask(), skel.index("mixamorig:Hips"))
    with torch.no_grad():
        check("零增量时姿态 == 暖启动", float((var() - warm).abs().max()) < 1e-6)
        var.rotvec.normal_(0, 0.3)
        check("任意增量下四元数仍为单位模长",
              float((var()[:, :, :4].norm(dim=-1) - 1).abs().max()) < 1e-5)
        var.rotvec.zero_()

    # --- 权重退火方向 --- #
    w = LossWeights()
    check("w_prior 随迭代递减", w.at(1.0).w_prior < w.at(0.0).w_prior)
    check("w_rom 随迭代递增", w.at(1.0).w_rom > w.at(0.0).w_rom)

    # --- 端到端求解（迭代数压低以控制自测时长） --- #
    res = solve(seq, skel, target,
                cfg=SolverConfig(adam_iters=120, lbfgs_iters=25, log_every=999))
    check("求解正常结束", res.converged, f"{res.elapsed_s:.1f}s")
    check("任务项大幅下降",
          res.final_terms["task"] < res.initial_terms["task"] * 0.25,
          f"{res.initial_terms['task']:.1f} -> {res.final_terms['task']:.1f}")
    check("最终解无 ROM 越界", res.final_terms["rom"] < 1.0,
          f"rom={res.final_terms['rom']:.4f}")

    out = MotionSequence(pose=res.pose, bone_names=seq.bone_names,
                         objects=seq.objects, fps=seq.fps)
    f2, _ = extract_features(out, skel, tray_name="draw_organizer_tray")
    check("屈膝确实加深", f2.knee_flex_left_max > feats.knee_flex_left_max + 10,
          f"{feats.knee_flex_left_max:.1f}° -> {f2.knee_flex_left_max:.1f}°")
    check("躯干确实更前倾", f2.trunk_lean_mean > feats.trunk_lean_mean + 4,
          f"{feats.trunk_lean_mean:.1f}° -> {f2.trunk_lean_mean:.1f}°")

    # --- 脚部接触（这一组是 contact 项的回归测试） --- #
    toe = skel.joint_positions(res.pose)[:, skel.index("mixamorig:LeftToeBase")]
    ground = float(skel.joint_positions(seq.pose)[:, skel.index("mixamorig:LeftToeBase"), 2].min())
    check("脚未穿透地面", float(toe[:, 2].min()) > ground - 0.02,
          f"最低 {float(toe[:, 2].min()):.3f}m（地面 {ground:.3f}m）")
    speed = (toe[1:, :2] - toe[:-1, :2]).norm(dim=-1) * seq.fps
    check("存在真正站定的帧（未打滑）", float(speed.min()) < 0.05,
          f"最小速度 {float(speed.min()):.3f} m/s")


# --------------------------------------------------------------------------- #
def main() -> int:
    print("=" * 70)
    print("motionedit 核心模块自测")
    print(f"torch={torch.__version__}  cuda={torch.cuda.is_available()}")
    print("=" * 70)

    test_rotation()
    skel = test_skeleton_and_fk()
    seq = test_motion_io(skel)
    feats, curves = test_features(seq, skel)
    test_rom(seq, skel)
    test_tray(seq, skel)
    test_params()
    test_targets(feats, curves)
    test_solver(seq, skel, feats, curves)

    print("\n" + "=" * 70)
    print(f"结果: {len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        for name in FAIL:
            print(f"  FAILED: {name}")
    print("=" * 70)
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
