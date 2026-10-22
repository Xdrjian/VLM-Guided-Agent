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
from motionedit.targets import (build_targets, resample, resample_objects,      # noqa: E402
                                resample_pose)
from motionedit.tray import calibrate_grip, update_tray_in_sequence  # noqa: E402

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

    # 屈髋必须有符号：支撑后期大腿摆到身后，无符号定义会把这段折叠成正值
    check("屈髋曲线跨零（存在伸髋相）", float(curves.hip_flex_left.min()) < -2.0,
          f"min={float(curves.hip_flex_left.min()):.1f}°, "
          f"max={float(curves.hip_flex_left.max()):.1f}°")
    check("屈髋峰值在 20~70°", 20 < feats.hip_flex_left_max < 70,
          f"L={feats.hip_flex_left_max:.1f}°, R={feats.hip_flex_right_max:.1f}°")
    # 摆动相腿往前摆，屈髋必大于支撑相——用于锁死正方向约定
    sw_mean = float(curves.hip_flex_left[~curves.gait.left_contact].mean())
    check("屈髋正方向正确（摆动相 > 支撑相）",
          sw_mean > feats.hip_flex_left_stance_mean + 5.0,
          f"摆动 {sw_mean:.1f}° vs 支撑 {feats.hip_flex_left_stance_mean:.1f}°")
    # 这条就是旧审计失效的根源：膝屈曲峰值只出现在摆动相
    check("膝屈曲峰值远高于支撑相均值",
          feats.knee_flex_left_max > feats.knee_flex_left_stance_mean + 30,
          f"峰值 {feats.knee_flex_left_max:.1f}° vs "
          f"支撑均值 {feats.knee_flex_left_stance_mean:.1f}°")
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
    print("\n[6] 托盘刚性抓握")
    TRAY = "draw_organizer_tray"
    grip = calibrate_grip(seq, skel, TRAY)
    check("抓握残差极小（双手扣在同一刚体上）", grip.residual_m < 0.005,
          f"{grip.residual_m * 1000:.2f} mm")
    check("参考动作托盘近乎水平", grip.tilt_max_deg < 15.0,
          f"倾角峰值 {grip.tilt_max_deg:.2f}°")

    out = update_tray_in_sequence(seq, skel, grip)
    tray = out.objects[TRAY]
    check("托盘轨道形状 [T,7]", tuple(tray.shape) == (seq.num_frames, 7), str(tuple(tray.shape)))

    q = tray[:, :4]
    check("托盘四元数为单位四元数", (q.norm(dim=-1) - 1).abs().max().item() < 1e-5)

    # 与原始轨道对比：刚性重建应远小于托盘半厚（启发式方案曾达 24.5 mm）
    orig = seq.objects[TRAY]
    pos_err = (tray[:, 4:] - orig[:, 4:]).norm(dim=-1).mean().item()
    rot_err = (1 - (tray[:, :4] * orig[:, :4]).sum(-1).abs()).mean().item()
    check("托盘位置重建误差 < 5 mm", pos_err < 0.005, f"{pos_err * 1000:.2f} mm")
    check("托盘朝向重建误差极小", rot_err < 1e-4, f"1-|dot|={rot_err:.6f}")

    # --- 几何断言：位姿数字本身正常也可能把托盘立成一块竖板 --- #
    if not seq.object_models or TRAY not in seq.object_models:
        check("托盘网格可用（几何断言前置条件）", False,
              "pkl 缺 object_models，请先跑 scripts/fix_tray_mesh.py")
        return

    verts = torch.tensor(np.asarray(seq.object_models[TRAY]["mesh"][0]))
    ext_local = (verts.max(0).values - verts.min(0).values)
    thin = int(ext_local.argmin())
    check("托盘网格为扁平形状", float(ext_local.min() / ext_local.max()) < 0.3,
          f"局部尺寸 {[round(float(e), 3) for e in ext_local]}")
    check("标定出的法线轴 = 网格最薄轴", grip.normal_axis == thin,
          f"normal={grip.normal_axis} thin={thin}")

    R = quaternion_to_matrix(tray[:, :4])
    vw = torch.einsum("tij,nj->tni", R, verts) + tray[:, None, 4:]
    size_w = (vw[0].max(0).values - vw[0].min(0).values)
    check("托盘在世界系中平放（竖直方向为厚度）",
          float(size_w[2]) < 2.0 * float(ext_local[thin]),
          f"世界竖直尺寸 {float(size_w[2]):.3f}m（网格厚度 {float(ext_local[thin]):.3f}m）")

    # 双手落在托盘两侧外沿，而不是盘面正中央——这是穿模与否的关键
    joints = skel.joint_positions(seq.pose)
    lh = joints[:, skel.index("mixamorig:LeftHand")]
    rh = joints[:, skel.index("mixamorig:RightHand")]
    sep = (rh - lh).norm(dim=-1)
    span = float(ext_local.max())
    check("双手扣在托盘外沿", float(sep.mean()) > span,
          f"手距 {float(sep.mean()):.3f}m vs 托盘跨度 {span:.3f}m")
    check("双手间距恒定（刚体约束）", float(sep.std()) < 0.002,
          f"标准差 {float(sep.std()) * 1000:.2f} mm")


def test_params():
    print("\n[7] params")
    p = GaitParams()
    check("默认参数为恒等", p.is_identity())
    check("参数数量为 8", len(p.to_dict()) == 8, str(list(p.to_dict())))
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
    check("步频降低 → 每帧源索引推进变慢", float(d.mean()) < 1.0,
          f"均步进 {float(d.mean()):.3f}")
    check("步频降低 → 输出变长（而非截断动作）", t1.num_frames > t0.num_frames,
          f"{t0.num_frames} 帧 -> {t1.num_frames} 帧")
    check("参考动作被完整走完", float(t1.src_idx[-1]) >= float(t0.src_idx[-1]) - 1.0,
          f"源索引末端 {float(t1.src_idx[-1]):.1f} / {float(t0.src_idx[-1]):.1f}")
    check("尾部无定格帧", not any("越界" in s for s in t1.notes))
    check("屈膝加深", float(t1.knee_flex_left.max()) > float(t0.knee_flex_left.max()) + 10,
          f"{float(t0.knee_flex_left.max()):.1f}° -> {float(t1.knee_flex_left.max()):.1f}°")
    check("躯干更前倾",
          abs(float(t1.trunk_lean.mean() - t0.trunk_lean.mean()) - 8.0) < 1.5)

    trav0 = float((t0.root_world[-1, :2] - t0.root_world[0, :2]).norm())
    trav1 = float((t1.root_world[-1, :2] - t1.root_world[0, :2]).norm())
    check("步长缩短", trav1 < trav0, f"{trav0:.3f}m -> {trav1:.3f}m")
    check("根节点高度不受步长参数影响",
          float((t1.root_world[:, 2]
                 - resample(curves.root_world, t1.src_idx)[:, 2]).abs().max()) < 1e-5,
          "（垂直分量单独处理）")
    check("托盘更贴身",
          t1.tray_chest_distance_mean < t0.tray_chest_distance_mean,
          f"{t0.tray_chest_distance_mean:.3f}m -> {t1.tray_chest_distance_mean:.3f}m")
    check("托盘距离目标为逐帧曲线（非标量均值）",
          t0.tray_chest_distance is not None
          and t0.tray_chest_distance.shape == (t0.num_frames,)
          and float(t0.tray_chest_distance.std()) > 1e-4,
          f"标准差 {float(t0.tray_chest_distance.std()):.4f}m")

    # 快慢两侧必须同标准：都把参考动作走完，只是花的时间不同
    ref_last = float(curves.knee_flex_left.shape[0] - 1)
    for cs in (0.5, 0.7, 1.0, 1.3, 1.8):
        tc = build_targets(curves, GaitParams(cadence_scale=cs).clipped(), ref_d)
        cov = float(tc.src_idx[-1]) / ref_last
        check(f"步频 {cs:.1f} 内容覆盖完整且无定格",
              cov > 0.985 and not any("越界" in s for s in tc.notes),
              f"{tc.num_frames} 帧，覆盖 {cov * 100:.1f}%")

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
    cold = resample_pose(seq.pose, torch.arange(
        target.num_frames, dtype=seq.pose.dtype).clamp(max=seq.num_frames - 1.0))
    rom = build_rom_penalty(skel)
    loss_cold = MotionLoss(skel, cold, target, rom)
    loss_warm = MotionLoss(skel, warm, target, rom)
    with torch.no_grad():
        task_cold = float(loss_cold(cold, 0.0).task)
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
    check("求解无硬性失败", not res.failed, f"{res.elapsed_s:.1f}s")
    # converged 必须名副其实：这点迭代量远不足以到驻点，它就该报 False。
    # 旧实现里该字段恒为 True（只在抛异常或 ROM 越界时才翻转），误导过一次排查。
    check("迭代量不足时 converged 如实报 False", not res.converged,
          str(res.notes[-2:]))
    check("记录 Adam 实际用了多少次", res.adam_used == 120, str(res.adam_used))

    # 停止判据：把阈值放宽到"每次检查必须再降 50%"，正常情况下立刻触发
    early = solve(seq, skel, target, cfg=SolverConfig(
        adam_iters=2000, lbfgs_iters=5, log_every=9999, anneal_frac=0.1,
        converge_tol=0.5, converge_check_every=20, converge_patience=2))
    check("任务项走平后提前停止", early.adam_used < 2000, str(early.adam_used))
    check("提前停止写进 notes",
          any("提前停止" in n for n in early.notes), str(early.notes))

    check("任务项大幅下降",
          res.final_terms["task"] < res.initial_terms["task"] * 0.25,
          f"{res.initial_terms['task']:.1f} -> {res.final_terms['task']:.1f}")
    check("最终解无 ROM 越界", res.final_terms["rom"] < 1.0,
          f"rom={res.final_terms['rom']:.4f}")

    out = MotionSequence(pose=res.pose, bone_names=seq.bone_names,
                         objects=resample_objects(seq.objects, target.src_idx),
                         fps=seq.fps, object_models=seq.object_models)
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

    # --- 恒等参数自洽性：目标就是 A_ref 本身，解也应当是 A_ref --- #
    # 这一条此前只在"目标层"验过、没在"求解器层"验过，于是 tray 任务项用标量均值
    # 去压真实波动的 bug 活了很久：恒等参数下它仍产生 3.92 的伪压力，
    # 求解器用肩关节 24.6° 的旋转去抹平，把双手从 0.579m 挤到 0.31m。
    grip_ref = calibrate_grip(seq, skel, "draw_organizer_tray")
    t_id = build_targets(curves, GaitParams(), ref_d)
    res_id = solve(seq, skel, t_id, grip=grip_ref,
                   cfg=SolverConfig(adam_iters=120, lbfgs_iters=20, log_every=999))
    check("恒等参数下任务项本就接近 0", res_id.initial_terms["task"] < 1.0,
          f"初始 task={res_id.initial_terms['task']:.3f}")

    def hand_sep(pose):
        j = skel.joint_positions(pose)
        return (j[:, skel.index("mixamorig:RightHand")]
                - j[:, skel.index("mixamorig:LeftHand")]).norm(dim=-1)

    sep0, sep1 = hand_sep(seq.pose), hand_sep(res_id.pose.cpu())
    check("恒等参数下双手间距不塌缩",
          abs(float(sep1.mean() - sep0.mean())) < 0.005,
          f"{float(sep0.mean()):.4f}m -> {float(sep1.mean()):.4f}m")
    check("恒等参数下抓握不脱手", res_id.final_terms["grip"] < 1.0,
          f"grip={res_id.final_terms['grip']:.4f}")
    check("恒等参数下托盘不倾倒", res_id.final_terms["level"] < 1.0,
          f"level={res_id.final_terms['level']:.4f}")

    q_a, q_b = seq.pose[:, :, :4], res_id.pose.cpu()[:, :, :4]
    dev_deg = 2 * torch.rad2deg(torch.acos((q_a * q_b).sum(-1).abs().clamp(max=1.0)))
    # 用逐骨骼**均值**而非全局最大值：足尖在未接触帧处于损失零空间，会游走
    # 十几度而不影响视觉；当初那个肩部塔缩 bug 的均值是 24.58°，本判据能抓住
    check("恒等参数下各骨骼均未显著偏离 A_ref",
          float(dev_deg.mean(0).max()) < 3.0,
          f"最大逐骨骼均值 {float(dev_deg.mean(0).max()):.2f}°")
    check("恒等参数下总损失近乎为零", res_id.final_terms["total"] < 1.0,
          f"total={res_id.final_terms['total']:.4f}")

    # --- 非恒等参数下托盘仍须守住 --- #
    check("heavy 参数下抓握仍不脱手", res.final_terms["grip"] < 5.0,
          f"grip={res.final_terms['grip']:.4f}")
    check("heavy 参数下托盘仍大致水平", res.final_terms["level"] < 25.0,
          f"level={res.final_terms['level']:.4f} deg²")

    # --- 设备一致性（CPU 自测无法发现跨设备 bug，故显式检查搬迁契约） --- #
    check("Skeleton 可搬迁", hasattr(skel, "to") and skel.to("cpu") is skel)
    check("TargetSpec 可搬迁", hasattr(target, "to") and target.to("cpu") is target)
    check("ROMPenalty 可搬迁", hasattr(rom, "to") and rom.to("cpu") is rom)
    check("GripCalibration 可搬迁",
          hasattr(grip_ref, "to") and grip_ref.to("cpu") is grip_ref)
    if torch.cuda.is_available():
        cuda_cfg = SolverConfig(adam_iters=20, lbfgs_iters=5,
                                device="cuda", log_every=999)
        # 两条分支都要覆盖：自动标定（solve 内部）与显式传入（run_solve.py 走这条）
        res_g = solve(seq, skel, target, cfg=cuda_cfg)
        check("CUDA 下可求解 · 自动标定抓握",
              res_g.pose.is_cuda and skel.device.type == "cpu")
        res_g2 = solve(seq, skel, target, cfg=cuda_cfg, grip=grip_ref)
        check("CUDA 下可求解 · 显式传入 CPU 抓握标定",
              res_g2.pose.is_cuda and grip_ref.offsets.device.type == "cpu")
        check("CUDA 与 CPU 的托盘约束数值一致",
              abs(res_g.final_terms["grip"] - res_g2.final_terms["grip"]) < 1e-3,
              f"{res_g.final_terms['grip']:.5f} vs {res_g2.final_terms['grip']:.5f}")
    else:
        check("CUDA 不可用，跨设备路径未覆盖（需在 Linux 机复测）", True, "skipped")


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
