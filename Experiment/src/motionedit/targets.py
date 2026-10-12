"""第 2 层：把 7 个宏观参数翻译成逐帧目标曲线（Solver_Prework.md §4.4）。

设计要点
--------
求解器不能直接优化"膝屈曲缩放 1.3 倍"这种抽象描述，它需要**每一帧的具体数值目标**。
本模块负责这次翻译，分三步：

1. **时间重映射**（``cadence_scale`` / ``double_support_offset``）
   先算出"目标动作的第 t 帧，应当对应参考动作的第几帧"，得到一条时间映射曲线。
   步频缩放是整体快慢；双支撑相偏置是周期内部的非均匀快慢（在双脚同时着地的
   相位附近放慢、摆动相加快），二者复合成一条单调递增的映射。

2. **姿态目标**（``knee_flex_scale`` / ``trunk_lean_offset`` / ``elbow_flex_offset``）
   把参考动作的角度曲线按时间映射重采样，再施加缩放/偏置。

3. **根节点目标**（``stride_length_scale``）
   对重采样后的根节点轨迹，把**世界坐标水平分量**的逐帧位移增量做缩放，
   垂直分量保持不变（否则会连带改变身高起伏）。

为什么目标是"角度"而不是"四元数"
--------------------------------
与 §1.5 的 ROM 结论同源：本数据的骨骼局部坐标轴与解剖学轴无对应关系，
把目标角度反推回四元数需要假设一个轴向，而该假设在本数据上不成立。
因此任务项统一在**几何角度域**上比较（三点夹角，与坐标约定无关）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import torch
from torch import Tensor

from .features import FeatureCurves
from .params import GaitParams

__all__ = ["TargetSpec", "build_targets", "resample", "resample_pose", "build_time_map"]

_TWO_PI = 2.0 * torch.pi


# --------------------------------------------------------------------------- #
def resample(curve: Tensor, src_idx: Tensor) -> Tensor:
    """在浮点索引 ``src_idx`` 处对 ``curve`` 做线性插值重采样。

    Parameters
    ----------
    curve   : ``[T]`` 或 ``[T, D]``
    src_idx : ``[T']`` 浮点源索引（可非整数）

    Returns
    -------
    ``[T']`` 或 ``[T', D]``
    """
    n = curve.shape[0]
    idx = src_idx.clamp(0.0, n - 1.0)
    lo = idx.floor().long()
    hi = (lo + 1).clamp(max=n - 1)
    w = (idx - lo.to(idx.dtype))
    if curve.dim() == 2:
        w = w.unsqueeze(-1)
    return curve[lo] * (1.0 - w) + curve[hi] * w


def resample_pose(pose: Tensor, src_idx: Tensor) -> Tensor:
    """按时间映射重采样整段姿态，用作求解器的**暖启动初值与先验锚点**。

    为什么需要它
    ------------
    步频/双支撑参数改变的是"时间的疏密"。若不先做重采样，求解器就得在**原时间轴上
    逐帧重新摆姿势**去模拟时间平移——实测这会让膝关节付出 71° 的局部旋转才能追上
    仅 25° 的屈曲变化差距，既浪费优化预算，又与姿势先验项直接对抗（先验要它别动，
    任务项要它大动）。先按映射把参考动作整体"拉伸/压缩"到新时间轴上，剩下的残差
    才是真正需要优化器解决的姿态问题。

    四元数用 **nlerp**（归一化线性插值）而非 slerp：相邻帧夹角很小（24fps 下通常
    < 5°），两者差异可忽略，而 nlerp 无奇异点、无分支、更快。插值前会做半球对齐，
    否则 ``q`` 与 ``-q`` 混插会得到错误的中间朝向。
    """
    n = pose.shape[0]
    idx = src_idx.clamp(0.0, n - 1.0)
    lo = idx.floor().long()
    hi = (lo + 1).clamp(max=n - 1)
    w = (idx - lo.to(idx.dtype)).unsqueeze(-1).unsqueeze(-1)   # [T',1,1]

    q_lo, q_hi = pose[lo, :, :4], pose[hi, :, :4]
    # 半球对齐：点积为负时翻转 q_hi，保证沿短弧插值
    sign = torch.where((q_lo * q_hi).sum(-1, keepdim=True) < 0, -1.0, 1.0)
    q = q_lo * (1.0 - w) + (q_hi * sign) * w
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    t = pose[lo, :, 4:] * (1.0 - w) + pose[hi, :, 4:] * w
    return torch.cat([q, t], dim=-1)


def build_time_map(    num_frames: int,
    cycle_frames: float,
    cadence_scale: float,
    double_support_offset: float,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> tuple[Tensor, List[str]]:
    """构造时间映射曲线 ``src[t]``（目标第 t 帧 → 参考第 src[t] 帧）。

    - ``cadence_scale`` > 1：步频加快，同样帧数内走完更多参考帧（源索引推进更快）
    - ``double_support_offset`` > 0：双支撑相变长，在相位 0 / 0.5 附近源索引推进变慢

    双支撑扰动采用 :math:`g(\\phi) = \\phi - \\frac{a}{4\\pi}\\sin(4\\pi\\phi)`，
    其中 :math:`a = 3 \\cdot \\text{offset}`。该形式满足：
    (a) 处处可微；(b) 当 :math:`|a| < 1` 时严格单调（保证时间不倒流）；
    (c) 每周期积分为 0（不改变周期总长，只改变周期内部的疏密）。

    Returns
    -------
    ``(src_idx [T], notes)``
    """
    notes: List[str] = []
    t = torch.arange(num_frames, device=device, dtype=dtype)

    # 第一步：整体快慢
    base = t * float(cadence_scale)

    # 第二步：周期内部的非均匀扰动
    a = 3.0 * float(double_support_offset)
    if abs(a) >= 0.95:
        a = 0.95 * (1.0 if a > 0 else -1.0)
        notes.append("double_support_offset 过大，已收紧以保证时间映射单调")
    if cycle_frames > 1e-6 and abs(a) > 1e-9:
        phi = base / float(cycle_frames)
        warped = float(cycle_frames) * (
            phi - a * torch.sin(2.0 * _TWO_PI * phi) / (2.0 * _TWO_PI))
        # 扰动只应改变周期内部的疏密，不应改变整段的总跨度。
        # 序列一般不是整数个周期，末端会残留一点正弦余量，这里按比例压回去
        # （正比例缩放不破坏单调性）。
        span, warped_span = float(base.max()), float(warped.max())
        if warped_span > 1e-6:
            warped = warped * (span / warped_span)
        base = warped.clamp_min(0.0)

    if float(base.max()) > num_frames - 1 + 1e-6:
        notes.append(
            f"步频放大导致源索引越界（需要 {float(base.max()):.1f} 帧、"
            f"仅有 {num_frames - 1} 帧），尾部已钳位，该段目标会静止"
        )
    return base.clamp(0.0, num_frames - 1.0), notes


# --------------------------------------------------------------------------- #
@dataclass
class TargetSpec:
    """一组参数对应的完整逐帧目标（任务项 loss 的比较对象）。"""

    #: ``[T]`` 时间映射：目标第 t 帧对应参考的第几帧
    src_idx: Tensor

    #: ``[T]`` 目标角度（度）
    knee_flex_left: Tensor
    knee_flex_right: Tensor
    trunk_lean: Tensor
    elbow_flex_left: Tensor
    elbow_flex_right: Tensor

    #: ``[T, 3]`` 目标根节点世界位置
    root_world: Tensor

    #: 目标托盘-胸骨水平距离（米），标量
    tray_chest_distance: float

    #: ``[T]`` 重采样后的参考相位，供渲染抽帧复用
    phase: Tensor

    notes: List[str] = field(default_factory=list)

    @property
    def num_frames(self) -> int:
        return int(self.src_idx.shape[0])

    def to(self, device: torch.device | str) -> "TargetSpec":
        """返回搬到指定设备的副本（``tray_chest_distance`` 是标量，原样保留）。"""
        device = torch.device(device)
        if device == self.src_idx.device:
            return self
        return TargetSpec(
            src_idx=self.src_idx.to(device),
            knee_flex_left=self.knee_flex_left.to(device),
            knee_flex_right=self.knee_flex_right.to(device),
            trunk_lean=self.trunk_lean.to(device),
            elbow_flex_left=self.elbow_flex_left.to(device),
            elbow_flex_right=self.elbow_flex_right.to(device),
            root_world=self.root_world.to(device),
            tray_chest_distance=self.tray_chest_distance,
            phase=self.phase.to(device),
            notes=list(self.notes),
        )

    def summary(self) -> dict:
        """可打印/可序列化的摘要，便于日志与调试。"""
        def rng(x: Tensor) -> str:
            return f"{float(x.min()):.1f}..{float(x.max()):.1f}"
        return {
            "num_frames": self.num_frames,
            "src_idx_range": rng(self.src_idx),
            "knee_flex_left": rng(self.knee_flex_left),
            "knee_flex_right": rng(self.knee_flex_right),
            "trunk_lean": rng(self.trunk_lean),
            "elbow_flex_left": rng(self.elbow_flex_left),
            "root_travel_m": round(
                float((self.root_world[-1, :2] - self.root_world[0, :2]).norm()), 4),
            "tray_chest_distance": round(self.tray_chest_distance, 4),
            "notes": self.notes,
        }


def build_targets(
    curves: FeatureCurves,
    params: GaitParams,
    tray_chest_distance_ref: float | None = None,
    num_frames: int | None = None,
) -> TargetSpec:
    """由参考动作的特征曲线 + 一组参数，生成逐帧目标。

    Parameters
    ----------
    curves                  : :func:`motionedit.features.extract_features` 的第二个返回值
    params                  : LLM 给出的一组参数（调用前应先 ``.clipped()``）
    tray_chest_distance_ref : 参考动作的托盘-胸骨距离（米）；``None`` 表示无托盘
    num_frames              : 输出帧数，默认与参考等长

    Notes
    -----
    当 ``params`` 为恒等参数时，本函数的输出应当逐点等于参考动作本身
    （误差仅来自浮点），这是一条重要的自洽性检查。
    """
    p = params
    ref_n = int(curves.knee_flex_left.shape[0])
    n = int(num_frames) if num_frames else ref_n
    dev, dt = curves.knee_flex_left.device, curves.knee_flex_left.dtype

    src, notes = build_time_map(
        n, curves.gait.cycle_frames, p.cadence_scale, p.double_support_offset,
        device=dev, dtype=dt,
    )

    # --- 姿态目标：重采样后施加缩放/偏置 -------------------------------- #
    knee_l = resample(curves.knee_flex_left, src) * p.knee_flex_scale
    knee_r = resample(curves.knee_flex_right, src) * p.knee_flex_scale
    trunk = resample(curves.trunk_lean, src) + p.trunk_lean_offset
    elbow_l = resample(curves.elbow_flex_left, src) + p.elbow_flex_offset
    elbow_r = resample(curves.elbow_flex_right, src) + p.elbow_flex_offset

    # 屈曲角为非负量（0° = 完全伸直），偏置可能把它压到负数，需钳回
    knee_l, knee_r = knee_l.clamp_min(0.0), knee_r.clamp_min(0.0)
    elbow_l, elbow_r = elbow_l.clamp_min(0.0), elbow_r.clamp_min(0.0)

    # --- 根节点目标：只缩放世界水平位移增量 ------------------------------ #
    root = resample(curves.root_world, src)                 # [n, 3]
    delta = root[1:] - root[:-1]                            # [n-1, 3]
    scaled = delta.clone()
    scaled[:, :2] = delta[:, :2] * p.stride_length_scale     # 世界 XY = 水平面
    root_target = torch.cat(
        [root[:1], root[:1] + torch.cumsum(scaled, dim=0)], dim=0)

    tray_d = (tray_chest_distance_ref or 0.0) * p.tray_chest_distance_scale

    return TargetSpec(
        src_idx=src,
        knee_flex_left=knee_l,
        knee_flex_right=knee_r,
        trunk_lean=trunk,
        elbow_flex_left=elbow_l,
        elbow_flex_right=elbow_r,
        root_world=root_target,
        tray_chest_distance=float(tray_d),
        phase=resample(curves.phase, src),
        notes=notes,
    )
