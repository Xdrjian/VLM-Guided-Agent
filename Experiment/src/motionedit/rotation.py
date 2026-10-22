"""旋转表示与 ISB 关节角分解。

约定
----
- 四元数统一为 **标量在前** 的 ``[w, x, y, z]``，与 HUMOTO pkl 的存储顺序一致。
- 所有函数支持任意前置批量维度 ``(..., 4)`` / ``(..., 3, 3)``，且全程可微。
- ISB 欧拉角均为 **内旋（intrinsic）** 分解，角度单位为弧度。

ISB 轴序（见 Solver_Prework.md §1.5）
-----------------------------------
- 膝 / 髋 / 脊柱 : ``ZXY``  → ``R = Rz(a) @ Rx(b) @ Ry(c)``
- 肩             : ``YXY``  → ``R = Ry(a) @ Rx(b) @ Ry(c)``
"""

from __future__ import annotations

import torch
from torch import Tensor

__all__ = [
    "normalize_quaternion",
    "quaternion_to_matrix",
    "matrix_to_quaternion",
    "quat_trans_to_matrix",
    "matrix_to_quat_trans",
    "matrix_to_euler_zxy",
    "euler_zxy_to_matrix",
    "matrix_to_euler_yxy",
    "euler_yxy_to_matrix",
    "quaternion_geodesic_sq",
    "axis_angle_to_quaternion",
    "quaternion_multiply",
    "rotvec_to_quaternion",
]

_EPS = 1e-8


# --------------------------------------------------------------------------- #
# 四元数基础运算
# --------------------------------------------------------------------------- #
def normalize_quaternion(q: Tensor) -> Tensor:
    """把四元数归一化到单位长度，形状 ``(..., 4)``。"""
    return q / q.norm(dim=-1, keepdim=True).clamp_min(_EPS)


def quaternion_to_matrix(q: Tensor) -> Tensor:
    """``[w,x,y,z]`` → ``3x3`` 旋转矩阵，形状 ``(..., 4)`` → ``(..., 3, 3)``。"""
    q = normalize_quaternion(q)
    w, x, y, z = q.unbind(-1)

    xx, yy, zz = x * x, y * y, z * z
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    m = torch.stack(
        [
            1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy),
            2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx),
            2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy),
        ],
        dim=-1,
    )
    return m.reshape(q.shape[:-1] + (3, 3))


def matrix_to_quaternion(m: Tensor) -> Tensor:
    """``3x3`` 旋转矩阵 → ``[w,x,y,z]``，使用数值稳定的分支选择实现。"""
    m00, m01, m02 = m[..., 0, 0], m[..., 0, 1], m[..., 0, 2]
    m10, m11, m12 = m[..., 1, 0], m[..., 1, 1], m[..., 1, 2]
    m20, m21, m22 = m[..., 2, 0], m[..., 2, 1], m[..., 2, 2]

    # 四个候选分支各自数值稳定的区域不同，取 trace 最大的分支
    t0 = 1.0 + m00 + m11 + m22
    t1 = 1.0 + m00 - m11 - m22
    t2 = 1.0 - m00 + m11 - m22
    t3 = 1.0 - m00 - m11 + m22

    def _branch(t: Tensor, a: Tensor, b: Tensor, c: Tensor, order: int) -> Tensor:
        s = torch.sqrt(t.clamp_min(_EPS)) * 2.0
        u, v, w_ = a / s, b / s, c / s
        q_scalar = 0.25 * s
        if order == 0:
            return torch.stack([q_scalar, u, v, w_], dim=-1)
        if order == 1:
            return torch.stack([u, q_scalar, v, w_], dim=-1)
        if order == 2:
            return torch.stack([u, v, q_scalar, w_], dim=-1)
        return torch.stack([u, v, w_, q_scalar], dim=-1)

    q0 = _branch(t0, m21 - m12, m02 - m20, m10 - m01, 0)
    q1 = _branch(t1, m21 - m12, m01 + m10, m02 + m20, 1)
    q2 = _branch(t2, m02 - m20, m01 + m10, m12 + m21, 2)
    q3 = _branch(t3, m10 - m01, m02 + m20, m12 + m21, 3)

    stacked_t = torch.stack([t0, t1, t2, t3], dim=-1)
    stacked_q = torch.stack([q0, q1, q2, q3], dim=-2)          # (..., 4, 4)
    idx = stacked_t.argmax(dim=-1, keepdim=True)               # (..., 1)
    idx = idx.unsqueeze(-1).expand(idx.shape + (4,))           # (..., 1, 4)
    q = torch.gather(stacked_q, -2, idx).squeeze(-2)

    # 规范化半球，保证 w >= 0，避免 q 与 -q 的二义性
    q = torch.where((q[..., :1] < 0).expand_as(q), -q, q)
    return normalize_quaternion(q)


def axis_angle_to_quaternion(axis: Tensor, angle: Tensor) -> Tensor:
    """绕单位轴 ``axis`` 旋转 ``angle`` 弧度 → ``[w,x,y,z]``。

    ``axis`` 形状 ``(..., 3)``，``angle`` 形状 ``(...)``。
    """
    axis = axis / axis.norm(dim=-1, keepdim=True).clamp_min(_EPS)
    half = angle * 0.5
    return torch.cat([torch.cos(half).unsqueeze(-1), axis * torch.sin(half).unsqueeze(-1)], dim=-1)


# --------------------------------------------------------------------------- #
# SE(3)
# --------------------------------------------------------------------------- #
def quat_trans_to_matrix(qt: Tensor) -> Tensor:
    """HUMOTO 的 7 维 ``[w,x,y,z,tx,ty,tz]`` → ``4x4`` 齐次矩阵。

    形状 ``(..., 7)`` → ``(..., 4, 4)``。
    """
    if qt.shape[-1] != 7:
        raise ValueError(f"expected last dim 7, got {qt.shape[-1]}")
    rot = quaternion_to_matrix(qt[..., :4])
    trans = qt[..., 4:]

    batch = qt.shape[:-1]
    out = torch.zeros(batch + (4, 4), dtype=qt.dtype, device=qt.device)
    out[..., :3, :3] = rot
    out[..., :3, 3] = trans
    out[..., 3, 3] = 1.0
    return out


def matrix_to_quat_trans(m: Tensor) -> Tensor:
    """``4x4`` 齐次矩阵 → 7 维 ``[w,x,y,z,tx,ty,tz]``。"""
    return torch.cat([matrix_to_quaternion(m[..., :3, :3]), m[..., :3, 3]], dim=-1)


# --------------------------------------------------------------------------- #
# ISB 欧拉角分解：ZXY（膝 / 髋 / 脊柱）
# --------------------------------------------------------------------------- #
def matrix_to_euler_zxy(m: Tensor) -> Tensor:
    """内旋 ``R = Rz(a) @ Rx(b) @ Ry(c)`` 的分解。

    返回 ``(..., 3)``，依次为 ``[a(Z), b(X), c(Y)]``，单位弧度。
    奇异点位于 ``b = ±90°``，正常人体关节活动范围内不会触及。
    """
    b = torch.asin(m[..., 2, 1].clamp(-1.0 + 1e-7, 1.0 - 1e-7))
    a = torch.atan2(-m[..., 0, 1], m[..., 1, 1])
    c = torch.atan2(-m[..., 2, 0], m[..., 2, 2])
    return torch.stack([a, b, c], dim=-1)


def euler_zxy_to_matrix(e: Tensor) -> Tensor:
    """``matrix_to_euler_zxy`` 的逆变换，``(..., 3)`` → ``(..., 3, 3)``。"""
    a, b, c = e.unbind(-1)
    ca, sa = torch.cos(a), torch.sin(a)
    cb, sb = torch.cos(b), torch.sin(b)
    cc, sc = torch.cos(c), torch.sin(c)

    m = torch.stack(
        [
            ca * cc - sa * sb * sc, -sa * cb, ca * sc + sa * sb * cc,
            sa * cc + ca * sb * sc, ca * cb, sa * sc - ca * sb * cc,
            -cb * sc, sb, cb * cc,
        ],
        dim=-1,
    )
    return m.reshape(e.shape[:-1] + (3, 3))


# --------------------------------------------------------------------------- #
# ISB 欧拉角分解：YXY（肩，避免大活动范围下卡尔丹角奇异）
# --------------------------------------------------------------------------- #
def matrix_to_euler_yxy(m: Tensor) -> Tensor:
    """内旋 ``R = Ry(a) @ Rx(b) @ Ry(c)`` 的分解。

    返回 ``(..., 3)``，依次为 ``[a(Y), b(X), c(Y)]``，其中 ``b ∈ [0, π]``。
    """
    b = torch.acos(m[..., 1, 1].clamp(-1.0 + 1e-7, 1.0 - 1e-7))
    a = torch.atan2(m[..., 0, 1], m[..., 2, 1])
    c = torch.atan2(m[..., 1, 0], -m[..., 1, 2])
    return torch.stack([a, b, c], dim=-1)


def euler_yxy_to_matrix(e: Tensor) -> Tensor:
    """``matrix_to_euler_yxy`` 的逆变换。"""
    a, b, c = e.unbind(-1)
    ca, sa = torch.cos(a), torch.sin(a)
    cb, sb = torch.cos(b), torch.sin(b)
    cc, sc = torch.cos(c), torch.sin(c)

    m = torch.stack(
        [
            ca * cc - sa * cb * sc, sa * sb, ca * sc + sa * cb * cc,
            sb * sc, cb, -sb * cc,
            -sa * cc - ca * cb * sc, ca * sb, -sa * sc + ca * cb * cc,
        ],
        dim=-1,
    )
    return m.reshape(e.shape[:-1] + (3, 3))


# --------------------------------------------------------------------------- #
# 距离度量
# --------------------------------------------------------------------------- #
def quaternion_geodesic_sq(q1: Tensor, q2: Tensor) -> Tensor:
    """平方弦距离 ``1 - <q1, q2>^2``，对 ``q`` / ``-q`` 不敏感。

    这是 Solver_Prework.md §4.1 中姿态先验项与任务项共用的距离形式，
    相比 ``arccos`` 形式在 ``<q1,q2> → ±1`` 时梯度稳定。
    """
    dot = (normalize_quaternion(q1) * normalize_quaternion(q2)).sum(-1)
    return 1.0 - dot.pow(2)


# --------------------------------------------------------------------------- #
# 组合与增量参数化
# --------------------------------------------------------------------------- #
def quaternion_multiply(q1: Tensor, q2: Tensor) -> Tensor:
    """四元数乘法 ``q1 ⊗ q2``（标量在前），等价于旋转矩阵的 ``R1 @ R2``。"""
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    return torch.stack([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], dim=-1)


def rotvec_to_quaternion(v: Tensor) -> Tensor:
    """旋转向量（轴×角，弧度）→ 四元数，``v = 0`` 处梯度有限。

    这是求解器的**增量参数化**基础：优化变量取 3 维无约束向量 ``v``，
    经本函数映射成单位四元数后再与参考姿态相乘。好处是
    (a) 输出自动是单位四元数，无需在优化中反复归一化或加约束；
    (b) ``v = 0`` 恰对应"不改变参考姿态"，天然是良好初值。

    在 ``|v| → 0`` 处用泰勒展开代替除法，避免 ``0/0`` 产生 NaN 梯度。
    """
    theta = v.norm(dim=-1, keepdim=True)
    half = 0.5 * theta
    # sinc(half) = sin(half)/half，小角度用展开式保证数值与梯度均稳定
    small = theta < 1e-4
    sinc = torch.where(small, 0.5 - half.pow(2) / 12.0, torch.sin(half) / theta.clamp_min(_EPS))
    return torch.cat([torch.cos(half), v * sinc], dim=-1)
