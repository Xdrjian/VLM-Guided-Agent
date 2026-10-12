"""关节活动度（ROM）约束：基于 **swing-twist 分解** 的实现。

为什么不用"固定欧拉分量 + ISB 轴序"
------------------------------------
实测发现（见 ``scripts/diagnose_rom.py``）：在 HUMOTO 的 **up_bone** 约定下，
每根骨骼的局部坐标轴统一指向世界上方，与骨骼自身走向、与解剖学屈伸轴都无关。
因此"膝屈曲 = ZXY 分解的 X 分量"这类假设不成立——A_ref 的膝关节几何屈曲实测
最大 98.8°（完全正常），但固定分量读数却给出 166.9° / -152.3° 的虚假越界，
若据此施加惩罚，会在参考动作本身上产生上万量级的伪惩罚，把优化器推向错误方向。

采用的方案
----------
对每个关节，先从 **静止骨架几何** 解析出该骨骼在自身局部坐标系下的长轴 ``u``
（骨骼指向其子骨骼的方向），再把局部旋转四元数做 swing-twist 分解::

    q = q_swing * q_twist       (q_twist 绕 u，q_swing 垂直于 u)

- ``twist``：绕骨骼长轴的旋转 → 解剖学上的 **内/外旋**
- ``swing``：骨骼长轴的偏摆幅度 → 解剖学上的 **屈伸 + 内收外展** 的合成量

两者都由数据本身定义，不依赖任何坐标轴约定，因此对 up_bone / mixamo_bone 均成立。
ROM 约束相应表述为"swing 幅度上限"与"twist 幅度上限"。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor

from .rotation import normalize_quaternion
from .skeleton import Skeleton

__all__ = [
    "SwingTwistLimit",
    "ROM_LIMITS",
    "swing_twist_decompose",
    "measure_swing_twist",
    "ROMPenalty",
    "build_rom_penalty",
]

_EPS = 1e-8


@dataclass(frozen=True)
class SwingTwistLimit:
    """单个关节的 swing / twist 上限（度）。"""

    max_swing: float
    max_twist: float


#: 各关节的 swing/twist 上限，数值由 Solver_Prework.md 附录 A 的解剖学 ROM 归并而来：
#: ``max_swing`` 取该关节屈伸/外展中的较大者并留出余量，``max_twist`` 取内外旋上限。
ROM_LIMITS: Dict[str, SwingTwistLimit] = {
    # 下肢：髋为球窝关节（swing 大），膝/踝以屈伸为主
    "mixamorig:LeftUpLeg":  SwingTwistLimit(max_swing=125.0, max_twist=50.0),
    "mixamorig:RightUpLeg": SwingTwistLimit(max_swing=125.0, max_twist=50.0),
    "mixamorig:LeftLeg":    SwingTwistLimit(max_swing=140.0, max_twist=25.0),
    "mixamorig:RightLeg":   SwingTwistLimit(max_swing=140.0, max_twist=25.0),
    "mixamorig:LeftFoot":   SwingTwistLimit(max_swing=55.0, max_twist=30.0),
    "mixamorig:RightFoot":  SwingTwistLimit(max_swing=55.0, max_twist=30.0),
    # 躯干：单节脊柱承担的角度有限
    "mixamorig:Spine":      SwingTwistLimit(max_swing=60.0, max_twist=35.0),
    "mixamorig:Spine1":     SwingTwistLimit(max_swing=30.0, max_twist=35.0),
    "mixamorig:Spine2":     SwingTwistLimit(max_swing=20.0, max_twist=20.0),
    # 上肢：肩活动范围极大，肘以屈伸为主
    "mixamorig:LeftArm":    SwingTwistLimit(max_swing=180.0, max_twist=95.0),
    "mixamorig:RightArm":   SwingTwistLimit(max_swing=180.0, max_twist=95.0),
    "mixamorig:LeftForeArm":  SwingTwistLimit(max_swing=150.0, max_twist=95.0),
    "mixamorig:RightForeArm": SwingTwistLimit(max_swing=150.0, max_twist=95.0),
    # 注：在该绑定中，前臂旋前/旋后（pronation/supination）被分配到 Hand 骨骼的
    # twist 通道上（实测 A_ref 稳定在 ±35~60°，对应端托姿势的旋前），
    # 因此 Hand 的 twist 上限按前臂旋前旋后范围（±90°）设定，而非腕关节本身的 ±30°。
    "mixamorig:LeftHand":   SwingTwistLimit(max_swing=75.0, max_twist=90.0),
    "mixamorig:RightHand":  SwingTwistLimit(max_swing=75.0, max_twist=90.0),
    "mixamorig:Neck":       SwingTwistLimit(max_swing=55.0, max_twist=75.0),
}


# --------------------------------------------------------------------------- #
def swing_twist_decompose(q: Tensor, axis: Tensor) -> Tuple[Tensor, Tensor]:
    """把四元数分解为绕 ``axis`` 的 twist 与垂直于它的 swing。

    Parameters
    ----------
    q    : ``[..., 4]`` 单位四元数 ``[w,x,y,z]``
    axis : ``[3]`` 或 ``[..., 3]`` 单位轴

    Returns
    -------
    ``(swing_deg, twist_deg)``：``swing_deg`` 为非负幅值，``twist_deg`` 带符号
    （绕 ``axis`` 的右手方向为正），单位均为度。
    """
    q = normalize_quaternion(q)
    axis = axis / axis.norm(dim=-1, keepdim=True).clamp_min(_EPS)
    if axis.dim() == 1:
        axis = axis.expand(q.shape[:-1] + (3,))

    w, v = q[..., :1], q[..., 1:]

    # twist = 归一化的 (w, proj_axis(v))
    proj = (v * axis).sum(-1, keepdim=True) * axis
    twist = torch.cat([w, proj], dim=-1)
    twist_norm = twist.norm(dim=-1, keepdim=True)

    # 退化情形（旋转恰好 180° 且完全垂直于 axis）时回退为单位四元数
    ident = torch.zeros_like(twist)
    ident[..., 0] = 1.0
    twist = torch.where(
        (twist_norm < 1e-6).expand_as(twist),
        ident,
        twist / twist_norm.clamp_min(_EPS),
    )

    # swing = q * twist^{-1}
    tw, tv = twist[..., :1], -twist[..., 1:]
    sw = w * tw - (v * tv).sum(-1, keepdim=True)
    sv = w * tv + tw * v + torch.cross(v, tv, dim=-1)
    swing = torch.cat([sw, sv], dim=-1)

    swing_deg = 2.0 * torch.rad2deg(torch.acos(swing[..., 0].abs().clamp(max=1.0 - 1e-7)))

    sign = torch.sign((twist[..., 1:] * axis).sum(-1))
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    twist_deg = sign * 2.0 * torch.rad2deg(torch.acos(twist[..., 0].abs().clamp(max=1.0 - 1e-7)))
    return swing_deg, twist_deg


# --------------------------------------------------------------------------- #
def _bone_axis_local(skel: Skeleton, idx: int) -> Optional[Tensor]:
    """骨骼在 **自身局部坐标系** 下的长轴方向（指向子骨骼）。"""
    children = (skel.parent_idx == idx).nonzero().flatten()
    if children.numel() == 0:
        return None

    rest = skel.rest
    # 多个子骨骼时取平均方向（如 Spine2 同时连 Neck / 双肩）
    tgt = rest[children, :3, 3].mean(dim=0)
    vec_world = tgt - rest[idx, :3, 3]
    if vec_world.norm() < 1e-6:
        return None

    vec_local = rest[idx, :3, :3].transpose(0, 1) @ vec_world
    return vec_local / vec_local.norm().clamp_min(_EPS)


def measure_swing_twist(
    pose: Tensor, skel: Skeleton, bones: Optional[Tuple[str, ...]] = None
) -> Dict[str, Tuple[Tensor, Tensor]]:
    """逐关节测量 swing / twist 角度序列，便于诊断与特征报告。"""
    names = tuple(bones) if bones else tuple(ROM_LIMITS)
    out: Dict[str, Tuple[Tensor, Tensor]] = {}
    for bone in names:
        if bone not in skel.bone_names:
            continue
        idx = skel.index(bone)
        axis = _bone_axis_local(skel, idx)
        if axis is None:
            continue
        out[bone] = swing_twist_decompose(pose[:, idx, :4], axis)
    return out


# --------------------------------------------------------------------------- #
@dataclass
class ROMPenalty:
    """预编译的 ROM 软惩罚算子（relu 形式，可微）。"""

    bone_idx: Tensor        # [K]
    axes: Tensor            # [K, 3] 局部长轴
    max_swing: Tensor       # [K]
    max_twist: Tensor       # [K]
    bone_names: Tuple[str, ...]

    def per_bone(self, pose: Tensor) -> Dict[str, float]:
        """返回每根骨骼的最大越界量（度），用于诊断与告警。"""
        out: Dict[str, float] = {}
        for k, name in enumerate(self.bone_names):
            s, t = swing_twist_decompose(pose[:, int(self.bone_idx[k]), :4], self.axes[k])
            viol = torch.relu(s - self.max_swing[k]).max()
            viol = torch.maximum(viol, torch.relu(t.abs() - self.max_twist[k]).max())
            out[name] = round(float(viol), 3)
        return out

    def __call__(self, pose: Tensor) -> Tensor:
        """``pose``: ``[T, B, 7]`` → 标量惩罚（单位：度²，已对帧数取平均）。"""
        total = pose.new_zeros(())
        for k in range(len(self.bone_names)):
            s, t = swing_twist_decompose(pose[:, int(self.bone_idx[k]), :4], self.axes[k])
            total = total + torch.relu(s - self.max_swing[k]).pow(2).sum()
            total = total + torch.relu(t.abs() - self.max_twist[k]).pow(2).sum()
        return total / max(pose.shape[0], 1)


def build_rom_penalty(
    skel: Skeleton, limits: Optional[Dict[str, SwingTwistLimit]] = None
) -> ROMPenalty:
    """构造 ROM 惩罚算子。"""
    limits = limits if limits is not None else ROM_LIMITS

    idx, axes, sw, tw, names = [], [], [], [], []
    for bone, lim in limits.items():
        if bone not in skel.bone_names:
            continue
        i = skel.index(bone)
        axis = _bone_axis_local(skel, i)
        if axis is None:
            continue
        idx.append(i)
        axes.append(axis)
        sw.append(lim.max_swing)
        tw.append(lim.max_twist)
        names.append(bone)

    dev, dt = skel.device, skel.rest.dtype
    return ROMPenalty(
        bone_idx=torch.tensor(idx, dtype=torch.long, device=dev),
        axes=torch.stack(axes).to(dtype=dt, device=dev),
        max_swing=torch.tensor(sw, dtype=dt, device=dev),
        max_twist=torch.tensor(tw, dtype=dt, device=dev),
        bone_names=tuple(names),
    )
