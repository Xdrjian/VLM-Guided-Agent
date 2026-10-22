"""托盘位姿重建（Solver_Prework.md §5.4.1）。

实证结论（2026-10，推翻了本模块此前的全部启发式）
--------------------------------------------------
托盘在 HUMOTO 中是独立对象轨道（``pkl['objects']``），但那条轨道在数值上
**完全是双手的派生物**——对 ``A_ref_straight`` 实测::

    左手→右手 相对位姿   位置抖动 2.08 mm   朝向抖动 0.05°
    托盘在左手局部系中   位置抖动 0.03 mm   朝向抖动 0.02°

这是一次**刚性抓握**：两只手扣在同一个刚体上，各自与托盘的相对位姿恒定
（双手在跨度方向位于 ±1.14 倍半跨处，即扣在托盘两侧外沿）。

因此托盘位姿不需要猜。标定出各手的恒定偏移后，用**双手对称最小二乘**拟合::

    方法                              位置误差
    启发式（跨度轴+法线轴+两个偏置）   24.50 mm   ← 与托盘半厚 24.9 mm 同量级，会穿模
    只绑左手                           0.03 mm   ← 精确但无视右手，右手脱手无人察觉
    双手对称最小二乘                   0.92 mm   ← 本模块采用

选双手而非绑单手的理由不是精度（两者都远小于托盘半厚），而是**可证伪性**：
双手各自预测的托盘位姿之差就是抓握残差，参考动作中为 1.84 mm；一旦求解器让
某只手脱离抓握，这个量立刻变大，既是诊断指标，也正是闭链损失项惩罚的对象。

随之删除的启发式超参：``palm_offset`` / ``lift_offset`` / ``span_axis`` /
``normal_axis`` / ``max_tilt_deg`` / ``smooth_window``。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import torch
from torch import Tensor

from .motion_io import MotionSequence
from .rotation import matrix_to_quaternion, quat_trans_to_matrix
from .skeleton import Skeleton

__all__ = ["GripCalibration", "DEFAULT_GRIP_BONES", "calibrate_grip",
           "tray_frames_from_hands", "tray_pose_from_hands",
           "tray_normal_from_frames", "update_tray_in_sequence"]

_UP = 2          # 世界竖直轴（Z-up）

#: 抓握托盘的骨骼。标定与使用必须用同一组、同一顺序。
DEFAULT_GRIP_BONES: Tuple[str, ...] = (
    "mixamorig:LeftHand", "mixamorig:RightHand")


@dataclass
class GripCalibration:
    """双手对托盘的刚性抓握标定。

    Attributes
    ----------
    tray_name    : 被抓握的物体名
    bones        : 抓握骨骼，与 ``offsets`` 一一对应
    offsets      : ``[K,4,4]`` 托盘在第 k 根骨骼局部系中的恒定位姿
    normal_axis  : 托盘局部的法线轴（厚度方向）索引
    normal_sign  : 使法线朝上的符号
    residual_m   : 标定残差（各手预测的位置分歧），参考动作约 0.002 m
    tilt_max_deg : 参考动作中托盘倾角的峰值，水平损失据此取阈值
    """

    tray_name: str
    bones: Tuple[str, ...]
    offsets: Tensor
    normal_axis: int
    normal_sign: float
    residual_m: float
    tilt_max_deg: float

    def to(self, device: torch.device | str) -> "GripCalibration":
        device = torch.device(device)
        if device == self.offsets.device:
            return self
        return GripCalibration(
            tray_name=self.tray_name, bones=self.bones,
            offsets=self.offsets.to(device), normal_axis=self.normal_axis,
            normal_sign=self.normal_sign, residual_m=self.residual_m,
            tilt_max_deg=self.tilt_max_deg)

    def summary(self) -> Dict[str, object]:
        return {
            "tray_name": self.tray_name,
            "bones": list(self.bones),
            "normal_axis": self.normal_axis,
            "normal_sign": self.normal_sign,
            "residual_mm": round(self.residual_m * 1000, 3),
            "tilt_max_deg": round(self.tilt_max_deg, 3),
        }


# --------------------------------------------------------------------------- #
def _world_up(n: int, like: Tensor) -> Tensor:
    up = torch.zeros(n, 3, dtype=like.dtype, device=like.device)
    up[:, _UP] = 1.0
    return up


def _orthonormalize(rot: Tensor) -> Tensor:
    """把一批近似旋转矩阵投影回 SO(3)（最近的正交阵，排除反射）。"""
    u, _, vh = torch.linalg.svd(rot.double())
    out = u @ vh
    neg = torch.linalg.det(out) < 0
    if bool(neg.any()):
        u = u.clone()
        u[neg, :, -1] = -u[neg, :, -1]
        out = u @ vh
    return out.to(rot.dtype)


# --------------------------------------------------------------------------- #
def calibrate_grip(
    seq: MotionSequence,
    skel: Skeleton,
    tray_name: Optional[str] = None,
    bones: Sequence[str] = DEFAULT_GRIP_BONES,
    residual_limit_m: float = 0.02,
) -> GripCalibration:
    """从**带真实托盘轨道的参考动作**标定刚性抓握。

    只能在参考动作上调用：求解输出的托盘轨道本身就是派生量，拿它标定会自我循环。

    Raises
    ------
    ValueError
        缺少该物体轨道，或抓握残差大到不足以视为刚性连接——后者说明这段动作里
        托盘并非被双手固定握持，此时**不应**用本模块重建其轨道，而应保留原轨道。
    """
    name = tray_name or (next(iter(seq.objects)) if seq.objects else None)
    if name is None or name not in seq.objects:
        raise ValueError(f"序列中没有物体轨道 '{name}'，无法标定抓握")

    world = skel.forward_kinematics(seq.pose).double()          # [T,B,4,4]
    tray = quat_trans_to_matrix(seq.objects[name]).double()     # [T,4,4]

    offsets = []
    for b in bones:
        o = (torch.linalg.inv(world[:, skel.index(b)]) @ tray).mean(0)
        o[:3, :3] = _orthonormalize(o[:3, :3].unsqueeze(0))[0]
        o[3] = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=o.dtype, device=o.device)
        offsets.append(o)
    off = torch.stack(offsets)                                   # [K,4,4]

    preds = torch.stack([world[:, skel.index(b)] @ off[k]
                         for k, b in enumerate(bones)])          # [K,T,4,4]
    pos = preds[:, :, :3, 3]
    residual = float((pos - pos.mean(0, keepdim=True)).norm(dim=-1).sum(0).mean())
    if residual > residual_limit_m:
        raise ValueError(
            f"抓握残差 {residual * 1000:.1f} mm 过大，'{name}' 不像被双手刚性握持。"
            f"若该动作中物体确实自由运动，请保留其原始轨道而非重建。")

    # 法线轴 = 托盘局部中最贴近世界竖直的那根轴
    rot = tray[:, :3, :3]
    up = _world_up(rot.shape[0], rot)
    scores = [float((rot[:, :, i] * up).sum(-1).mean()) for i in range(3)]
    normal_axis = int(max(range(3), key=lambda i: abs(scores[i])))
    normal_sign = 1.0 if scores[normal_axis] >= 0 else -1.0

    n = rot[:, :, normal_axis] * normal_sign
    sin_t = torch.linalg.cross(n, up, dim=-1).norm(dim=-1)
    tilt_max = float(torch.rad2deg(torch.asin(sin_t.clamp(max=1.0 - 1e-7))).max())

    return GripCalibration(
        tray_name=name, bones=tuple(bones), offsets=off.to(seq.pose.dtype),
        normal_axis=normal_axis, normal_sign=normal_sign,
        residual_m=residual, tilt_max_deg=tilt_max)


# --------------------------------------------------------------------------- #
def tray_frames_from_hands(
    world: Tensor, skel: Skeleton, calib: GripCalibration
) -> Tensor:
    """各抓握骨骼**各自**预测的托盘 4×4 位姿，``[K,T,4,4]``，保持梯度。

    这里刻意不做平均：两手预测之差就是抓握残差，闭链损失项需要它。
    """
    off = calib.offsets.to(dtype=world.dtype, device=world.device)
    return torch.stack([world[:, skel.index(b)] @ off[k]
                        for k, b in enumerate(calib.bones)])


def tray_normal_from_frames(frames: Tensor, calib: GripCalibration) -> Tensor:
    """由各手预测的托盘位姿取平均法线（朝上），``[T,3]``，保持梯度。"""
    n = frames[:, :, :3, calib.normal_axis].mean(0) * calib.normal_sign
    return torch.nn.functional.normalize(n, dim=-1)


def tray_pose_from_hands(
    pose: Tensor, skel: Skeleton, calib: GripCalibration
) -> Tensor:
    """由姿态解算托盘轨道 ``[T,7]``（双手对称最小二乘）。"""
    world = skel.forward_kinematics(pose)
    frames = tray_frames_from_hands(world, skel, calib)
    trans = frames[:, :, :3, 3].mean(0)
    rot = _orthonormalize(frames[:, :, :3, :3].mean(0))
    return torch.cat([matrix_to_quaternion(rot), trans], dim=-1)


def update_tray_in_sequence(
    seq: MotionSequence, skel: Skeleton, calib: GripCalibration
) -> MotionSequence:
    """用抓握标定重建序列中的托盘轨道，返回新的 :class:`MotionSequence`。"""
    if calib.tray_name not in seq.objects:
        return seq
    out = seq.clone()
    out.objects[calib.tray_name] = tray_pose_from_hands(seq.pose, skel, calib)
    return out
