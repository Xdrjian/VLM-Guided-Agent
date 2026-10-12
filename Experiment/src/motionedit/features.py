"""生物力学步态特征提取器。

本模块是 Solver_Prework.md 中三处共用的同一套提取器：
1. §4.4 任务项目标曲线的基准（参数作用在这些特征通道上）；
2. §4.4/§4.5 提供给 LLM 的"求解器实际执行效果"反馈；
3. §4.4 "求解器执行失效"告警的判据来源。

设计取舍
--------
- 膝 / 肘屈曲用 **三点夹角**（世界坐标，几何直接、与骨架朝向约定无关），
  比 ISB 欧拉角更稳，适合作为"报告给 LLM 的可读特征"；
- ISB 欧拉角仅用于 ROM 软惩罚（见 :mod:`motionedit.rom`），二者用途不同、不冲突；
- 躯干前倾用躯干向量与世界竖直轴夹角，正值表示前倾。
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Optional

import torch
from torch import Tensor

from .motion_io import MotionSequence
from .phase import ContactConfig, GaitPhase, detect_gait_phase
from .skeleton import Skeleton

__all__ = ["GaitFeatures", "extract_features", "joint_angle_3pt", "trunk_lean_angle"]

_UP = 2  # Z-up


# --------------------------------------------------------------------------- #
def joint_angle_3pt(a: Tensor, b: Tensor, c: Tensor) -> Tensor:
    """三点夹角的 **屈曲量**（度）：``0°`` 表示完全伸直。

    ``a-b-c`` 为近端 / 关节 / 远端三点，``[T,3]``。
    返回 ``180° - ∠abc``，即临床意义上的屈曲角度。
    """
    v1 = a - b
    v2 = c - b
    cos = (v1 * v2).sum(-1) / (v1.norm(dim=-1) * v2.norm(dim=-1)).clamp_min(1e-8)
    return 180.0 - torch.rad2deg(torch.acos(cos.clamp(-1 + 1e-7, 1 - 1e-7)))


def trunk_lean_angle(hips: Tensor, chest: Tensor) -> Tensor:
    """躯干相对世界竖直轴的倾角（度），正值表示偏离竖直。"""
    v = chest - hips
    cos = v[:, _UP] / v.norm(dim=-1).clamp_min(1e-8)
    return torch.rad2deg(torch.acos(cos.clamp(-1 + 1e-7, 1 - 1e-7)))


# --------------------------------------------------------------------------- #
@dataclass
class GaitFeatures:
    """一段动作的标量特征摘要（可直接序列化成 JSON 喂给 LLM）。"""

    num_frames: int
    duration_s: float

    cadence_steps_per_s: float
    cycle_frames: float
    stride_length_m: float
    double_support_ratio: float

    knee_flex_left_mean: float
    knee_flex_left_max: float
    knee_flex_right_mean: float
    knee_flex_right_max: float

    trunk_lean_mean: float
    trunk_lean_max: float

    elbow_flex_left_mean: float
    elbow_flex_right_mean: float

    com_vertical_range_m: float
    tray_chest_distance_mean: Optional[float] = None
    phase_valid: bool = True

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass
class FeatureCurves:
    """逐帧曲线（任务项构造目标曲线时使用，不直接给 LLM）。"""

    phase: Tensor                # [T]
    knee_flex_left: Tensor       # [T] 度
    knee_flex_right: Tensor
    trunk_lean: Tensor
    elbow_flex_left: Tensor
    elbow_flex_right: Tensor
    root_world: Tensor           # [T,3] 根节点世界位置
    com_height: Tensor           # [T]
    gait: GaitPhase


# --------------------------------------------------------------------------- #
def extract_features(
    seq: MotionSequence,
    skel: Skeleton,
    contact_cfg: ContactConfig = ContactConfig(),
    tray_name: Optional[str] = None,
) -> tuple[GaitFeatures, FeatureCurves]:
    """提取标量特征与逐帧曲线。

    Returns
    -------
    ``(GaitFeatures, FeatureCurves)``
    """
    joints = skel.joint_positions(seq.pose)          # [T, B, 3]

    def J(name: str) -> Tensor:
        return joints[:, skel.index(name)]

    hips = J("mixamorig:Hips")
    chest = J("mixamorig:Spine2")

    knee_l = joint_angle_3pt(J("mixamorig:LeftUpLeg"), J("mixamorig:LeftLeg"), J("mixamorig:LeftFoot"))
    knee_r = joint_angle_3pt(J("mixamorig:RightUpLeg"), J("mixamorig:RightLeg"), J("mixamorig:RightFoot"))
    elbow_l = joint_angle_3pt(J("mixamorig:LeftArm"), J("mixamorig:LeftForeArm"), J("mixamorig:LeftHand"))
    elbow_r = joint_angle_3pt(J("mixamorig:RightArm"), J("mixamorig:RightForeArm"), J("mixamorig:RightHand"))
    trunk = trunk_lean_angle(hips, chest)

    gait = detect_gait_phase(
        J("mixamorig:LeftToeBase"), J("mixamorig:RightToeBase"), seq.fps, contact_cfg, up_axis=_UP
    )

    # 步长：一个完整周期内根节点水平位移；周期不可用时退化为全程平均
    horiz = torch.stack([hips[:, 0], hips[:, 1]], dim=-1)
    step = (horiz[1:] - horiz[:-1]).norm(dim=-1)
    dist_per_frame = float(step.mean()) if step.numel() else 0.0
    stride_length = dist_per_frame * gait.cycle_frames

    cadence = 2.0 * seq.fps / gait.cycle_frames if gait.cycle_frames > 0 else 0.0

    tray_dist = None
    if tray_name and tray_name in seq.objects:
        tray_xy = seq.objects[tray_name][:, 4:6]
        tray_dist = float((tray_xy - chest[:, :2]).norm(dim=-1).mean())

    feats = GaitFeatures(
        num_frames=seq.num_frames,
        duration_s=round(seq.duration, 4),
        cadence_steps_per_s=round(cadence, 4),
        cycle_frames=round(gait.cycle_frames, 3),
        stride_length_m=round(stride_length, 4),
        double_support_ratio=round(gait.double_support_ratio, 4),
        knee_flex_left_mean=round(float(knee_l.mean()), 3),
        knee_flex_left_max=round(float(knee_l.max()), 3),
        knee_flex_right_mean=round(float(knee_r.mean()), 3),
        knee_flex_right_max=round(float(knee_r.max()), 3),
        trunk_lean_mean=round(float(trunk.mean()), 3),
        trunk_lean_max=round(float(trunk.max()), 3),
        elbow_flex_left_mean=round(float(elbow_l.mean()), 3),
        elbow_flex_right_mean=round(float(elbow_r.mean()), 3),
        com_vertical_range_m=round(float(hips[:, _UP].max() - hips[:, _UP].min()), 4),
        tray_chest_distance_mean=round(tray_dist, 4) if tray_dist is not None else None,
        phase_valid=gait.valid,
    )

    curves = FeatureCurves(
        phase=gait.phase,
        knee_flex_left=knee_l,
        knee_flex_right=knee_r,
        trunk_lean=trunk,
        elbow_flex_left=elbow_l,
        elbow_flex_right=elbow_r,
        root_world=hips,
        com_height=hips[:, _UP],
        gait=gait,
    )
    return feats, curves
