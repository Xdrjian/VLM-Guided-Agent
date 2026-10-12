"""托盘位姿更新（Solver_Prework.md §5.4.1）。

背景
----
托盘在 HUMOTO 中是 **独立对象轨道**（``pkl['objects']``），与人体骨架没有父子绑定。
因此求解器改变步态后，不能沿用 A_ref 的托盘轨道，否则手与托盘会脱节。

策略：双手驱动 + 朝向约束 + 时间平滑
------------------------------------
1. 位置 = 双手中点 + 沿托盘局部前向/法线的常量偏置；
2. **span 轴**（默认局部 Z）对齐 ``RightHand - LeftHand``；
3. **normal 轴**（默认局部 Y）尽量贴近世界竖直，并限制最大倾角，防止"翻盘"；
4. 对最终位姿做时间平滑，抑制手部抖动引起的高频晃动。

轴约定不能硬编码（2026-09 实证）
--------------------------------
文档早期版本写的是"局部 X 对齐双手连线、局部 Z 为法线"，但对 A_ref 的实测结果是::

    局部 X: 与手连线 |cos|=0.224   与竖直 |cos|=0.035    ← 托盘前后向（0.369 m）
    局部 Y: 与手连线 |cos|=0.029   与竖直 |cos|=0.998    ← 法线，最薄轴（0.050 m）
    局部 Z: 与手连线 |cos|=0.974   与竖直 |cos|=0.037    ← 与双手连线对齐（0.493 m）

按错误约定重建会把托盘**立成一块竖直的板**（世界竖直方向占 0.50 m 而非 0.05 m）。
危险之处在于：位姿四元数本身没有任何异常，只有把网格变换到世界系才看得出来——
在 ``object_models`` 缺失的那段时间里，这个 bug 完全不可见。

因此 :class:`TrayConfig` 把轴约定显式暴露为字段，并提供
:func:`calibrate_tray_config` 从参考动作反推，避免换数据集时重蹈覆辙。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import Tensor

from .motion_io import MotionSequence
from .rotation import matrix_to_quaternion, quaternion_to_matrix
from .skeleton import Skeleton

__all__ = ["TrayConfig", "solve_tray_pose", "update_tray_in_sequence",
           "calibrate_tray_config"]

_UP = 2          # 世界竖直轴（Z-up）


@dataclass
class TrayConfig:
    """托盘更新超参数。

    默认值来自对 ``A_ref_straight`` 的标定（:func:`calibrate_tray_config`），
    换数据集/换托盘模型时**必须重新标定**，不要沿用。
    """

    #: 沿托盘局部 forward 轴的偏置（米）。实测 A_ref ≈ 0.114
    palm_offset: float = 0.114
    #: 沿托盘局部 normal 轴的偏置（米）。实测 A_ref ≈ -0.016（托盘原点略低于手心）
    lift_offset: float = -0.016
    #: 托盘法线相对世界竖直的最大允许倾角
    max_tilt_deg: float = 20.0
    #: 移动平均窗口（帧），需为奇数；<=1 表示不平滑
    smooth_window: int = 5
    #: 与双手连线对齐的局部轴索引（0=X, 1=Y, 2=Z）
    span_axis: int = 2
    #: 托盘法线（厚度方向）的局部轴索引
    normal_axis: int = 1

    @property
    def forward_axis(self) -> int:
        """剩下的那个轴，即托盘的前后方向。"""
        return 3 - self.span_axis - self.normal_axis

    def validate(self) -> None:
        a = {self.span_axis, self.normal_axis, self.forward_axis}
        if a != {0, 1, 2}:
            raise ValueError(
                f"轴索引非法：span={self.span_axis}, normal={self.normal_axis}")


# --------------------------------------------------------------------------- #
def _moving_average(x: Tensor, window: int) -> Tensor:
    """沿时间维的移动平均，边界用边缘值填充，保持长度不变。"""
    if window <= 1:
        return x
    if window % 2 == 0:
        window += 1
    pad = window // 2

    xt = x.transpose(0, -1).unsqueeze(0)                     # [1, C, T]
    xt = torch.nn.functional.pad(xt, (pad, pad), mode="replicate")
    kernel = torch.ones(1, 1, window, dtype=x.dtype, device=x.device) / window
    kernel = kernel.expand(xt.shape[1], 1, window)
    out = torch.nn.functional.conv1d(xt, kernel, groups=xt.shape[1])
    return out.squeeze(0).transpose(0, -1)


def _align_hemisphere(q: Tensor) -> Tensor:
    """让相邻帧四元数落在同一半球，避免 ``q``/``-q`` 跳变污染平均值。"""
    out = q.clone()
    for t in range(1, out.shape[0]):
        if float((out[t] * out[t - 1]).sum()) < 0.0:
            out[t] = -out[t]
    return out


def _smooth_pose(pose: Tensor, window: int) -> Tensor:
    """对 ``[T,7]`` 位姿平滑：平移直接平均，旋转先对齐半球、平均后重新归一化。"""
    if window <= 1:
        return pose
    quat = _moving_average(_align_hemisphere(pose[:, :4]), window)
    quat = torch.nn.functional.normalize(quat, dim=-1)
    trans = _moving_average(pose[:, 4:], window)
    return torch.cat([quat, trans], dim=-1)


def _clamp_tilt(normal: Tensor, max_tilt_deg: float) -> Tensor:
    """把法线与世界竖直的夹角限制在 ``max_tilt_deg`` 内。"""
    up = torch.zeros_like(normal)
    up[:, _UP] = 1.0

    cos_max = torch.cos(torch.deg2rad(torch.tensor(max_tilt_deg, dtype=normal.dtype, device=normal.device)))
    cos = (normal * up).sum(-1, keepdim=True).clamp(-1 + 1e-7, 1 - 1e-7)

    # 超限时朝 up 方向做球面插值收回
    over = (cos < cos_max).squeeze(-1)
    if over.any():
        blended = normal.clone()
        blended[over] = torch.nn.functional.normalize(
            normal[over] * 0.3 + up[over] * 0.7, dim=-1
        )
        normal = blended
    return torch.nn.functional.normalize(normal, dim=-1)


# --------------------------------------------------------------------------- #
def _perm_sign(a: int, b: int, c: int) -> float:
    """排列 ``(a,b,c)`` 相对 ``(0,1,2)`` 的奇偶性：偶 +1，奇 -1。

    用于保证构造出的基是**右手系**：对右手坐标系，
    ``e_a = sign * (e_b × e_c)``，其中 sign 由排列奇偶性决定。
    """
    return 1.0 if (a, b, c) in {(0, 1, 2), (1, 2, 0), (2, 0, 1)} else -1.0


def solve_tray_pose(
    left_hand: Tensor,
    right_hand: Tensor,
    cfg: TrayConfig = TrayConfig(),
) -> Tensor:
    """由双手世界坐标解算托盘位姿。

    Parameters
    ----------
    left_hand, right_hand : ``[T, 3]`` 世界坐标。

    Returns
    -------
    ``[T, 7]``，格式为 ``[qw,qx,qy,qz,tx,ty,tz]``，可直接写回
    ``MotionSequence.objects``。
    """
    cfg.validate()
    fa, na, sa = cfg.forward_axis, cfg.normal_axis, cfg.span_axis

    # span：左手 → 右手
    span = torch.nn.functional.normalize(right_hand - left_hand, dim=-1)

    # normal：尽量竖直，且与 span 正交
    up = torch.zeros_like(span)
    up[:, _UP] = 1.0
    normal = torch.nn.functional.normalize(
        up - (up * span).sum(-1, keepdim=True) * span, dim=-1)
    normal = _clamp_tilt(normal, cfg.max_tilt_deg)
    normal = torch.nn.functional.normalize(
        normal - (normal * span).sum(-1, keepdim=True) * span, dim=-1)

    # forward：由右手系补全（符号取决于三轴的排列奇偶性）
    forward = _perm_sign(fa, na, sa) * torch.cross(normal, span, dim=-1)
    forward = torch.nn.functional.normalize(forward, dim=-1)

    # 按配置把三个轴放进旋转矩阵的对应**列**
    rot = torch.zeros(span.shape[0], 3, 3, dtype=span.dtype, device=span.device)
    rot[:, :, fa] = forward
    rot[:, :, na] = normal
    rot[:, :, sa] = span
    quat = matrix_to_quaternion(rot)

    # 偏置沿托盘**局部**轴施加，而非世界轴——托盘倾斜时前者才是正确语义
    center = (0.5 * (left_hand + right_hand)
              + cfg.palm_offset * forward
              + cfg.lift_offset * normal)

    pose = torch.cat([quat, center], dim=-1)
    return _smooth_pose(pose, cfg.smooth_window)


# --------------------------------------------------------------------------- #
def calibrate_tray_config(
    seq: MotionSequence,
    skel: Skeleton,
    tray_name: Optional[str] = None,
    base: Optional[TrayConfig] = None,
) -> Tuple[TrayConfig, dict]:
    """从**参考动作的原始托盘轨道**反推轴约定与偏置。

    换数据集或换托盘模型时先跑这个，把结果写回 :class:`TrayConfig` 的默认值，
    不要凭文档描述猜测轴约定——文档描述已被证伪过一次。

    Returns
    -------
    (标定后的配置, 诊断信息)
    """
    name = tray_name or (next(iter(seq.objects)) if seq.objects else None)
    if name is None:
        raise ValueError("序列中没有物体轨道，无法标定")

    joints = skel.joint_positions(seq.pose)
    lh = joints[:, skel.index("mixamorig:LeftHand")]
    rh = joints[:, skel.index("mixamorig:RightHand")]
    mid = 0.5 * (lh + rh)
    hand_dir = torch.nn.functional.normalize(rh - lh, dim=-1)

    tray = seq.objects[name]
    rot = quaternion_to_matrix(tray[:, :4])                  # [T,3,3]
    up = torch.zeros_like(hand_dir)
    up[:, _UP] = 1.0

    # 每个局部轴与"手连线"/"世界竖直"的平均 |cos|
    span_score, normal_score = [], []
    for i in range(3):
        axis = rot[:, :, i]
        span_score.append(float((axis * hand_dir).sum(-1).abs().mean()))
        normal_score.append(float((axis * up).sum(-1).abs().mean()))

    span_axis = int(max(range(3), key=lambda i: span_score[i]))
    normal_axis = int(max((i for i in range(3) if i != span_axis),
                          key=lambda i: normal_score[i]))

    cfg = base or TrayConfig()
    cfg = TrayConfig(max_tilt_deg=cfg.max_tilt_deg, smooth_window=cfg.smooth_window,
                     span_axis=span_axis, normal_axis=normal_axis)

    # 托盘原点相对手中点的偏移，投影到托盘局部系
    delta_local = torch.einsum("tji,tj->ti", rot, tray[:, 4:] - mid)
    cfg.palm_offset = float(delta_local[:, cfg.forward_axis].mean())
    cfg.lift_offset = float(delta_local[:, cfg.normal_axis].mean())

    diag = {
        "tray_name": name,
        "span_scores": [round(s, 3) for s in span_score],
        "normal_scores": [round(s, 3) for s in normal_score],
        "span_axis": span_axis,
        "normal_axis": normal_axis,
        "forward_axis": cfg.forward_axis,
        "palm_offset": round(cfg.palm_offset, 4),
        "lift_offset": round(cfg.lift_offset, 4),
        "span_confidence": round(span_score[span_axis], 3),
        "normal_confidence": round(normal_score[normal_axis], 3),
    }
    if seq.object_models and name in seq.object_models:
        verts = seq.object_models[name]["mesh"][0]
        import numpy as np
        ext = np.asarray(verts).max(0) - np.asarray(verts).min(0)
        diag["mesh_extent"] = [round(float(e), 4) for e in ext]
        diag["thinnest_axis"] = int(ext.argmin())
        if diag["thinnest_axis"] != normal_axis:
            diag["warning"] = (
                f"最薄轴({diag['thinnest_axis']})与推断的法线轴({normal_axis})不一致，"
                f"请人工确认托盘是否为扁平形状")
    return cfg, diag


def update_tray_in_sequence(
    seq: MotionSequence,
    skel: Skeleton,
    tray_name: str | None = None,
    cfg: TrayConfig = TrayConfig(),
) -> MotionSequence:
    """就地更新序列中的托盘轨道，返回新的 :class:`MotionSequence`。

    ``tray_name`` 为 ``None`` 时自动取 ``seq.objects`` 中的第一个物体。
    """
    if not seq.objects:
        return seq
    name = tray_name or next(iter(seq.objects))

    joints = skel.joint_positions(seq.pose)
    lh = joints[:, skel.index("mixamorig:LeftHand")]
    rh = joints[:, skel.index("mixamorig:RightHand")]

    out = seq.clone()
    out.objects[name] = solve_tray_pose(lh, rh, cfg)
    return out
