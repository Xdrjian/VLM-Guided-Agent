"""LLM 可调的生物力学中介参数定义（Solver_Prework.md §4.4）。

v0.8 起，原 ``arm_swing_amp_scale`` / ``arm_swing_freq_sync`` 已被替换为
``elbow_flex_offset`` / ``tray_chest_distance_scale``：参考动作为"双手端托盘"，
手臂几乎不摆动（实测肩关节偏离仅 17~31°，且无周期性），摆动类参数信息量过低。
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, fields
from typing import Dict, Tuple

__all__ = ["GaitParams", "DEFAULT_PARAMS", "PARAM_BOUNDS", "PARAM_NAMES"]


@dataclass
class GaitParams:
    """一轮 LLM 决策输出的完整参数组合（始终是绝对值，不是增量）。"""

    #: 膝屈曲角乘性缩放，>1 表示屈膝更深
    knee_flex_scale: float = 1.0
    #: 躯干前倾加性偏置，单位 **度**，正值表示更前倾
    trunk_lean_offset: float = 0.0
    #: 步频乘性缩放，>1 表示相同时长内步数更多
    cadence_scale: float = 1.0
    #: 双支撑相占比加性偏置（归一化相位比例，非度数），正值表示双脚同时着地更久
    double_support_offset: float = 0.0
    #: 步长乘性缩放，作用于根节点 **世界坐标水平面** 位移增量
    stride_length_scale: float = 1.0
    #: 肘屈曲加性偏置，单位 **度**，正值表示肘部更收紧（托盘更贴身）
    elbow_flex_offset: float = 0.0
    #: 托盘到胸骨参考点的水平距离乘性缩放，<1 表示托盘更贴近胸前
    tray_chest_distance_scale: float = 1.0

    # ------------------------------------------------------------------ #
    def to_dict(self) -> Dict[str, float]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, float]) -> "GaitParams":
        """从 LLM 返回的 JSON 构造，忽略 ``reasoning`` 等附加字段。"""
        known = {f.name for f in fields(cls)}
        return cls(**{k: float(v) for k, v in d.items() if k in known})

    def clipped(self) -> "GaitParams":
        """把所有参数裁剪到 :data:`PARAM_BOUNDS` 内，防止 LLM 给出越界值。"""
        out = {}
        for name, value in self.to_dict().items():
            lo, hi = PARAM_BOUNDS[name]
            out[name] = min(max(value, lo), hi)
        return GaitParams(**out)

    def is_identity(self, tol: float = 1e-6) -> bool:
        """是否为恒等参数（此时任务项目标曲线退化为 A_ref 本身）。"""
        d, ref = self.to_dict(), DEFAULT_PARAMS.to_dict()
        return all(abs(d[k] - ref[k]) <= tol for k in d)

    def diff(self, other: "GaitParams") -> Dict[str, float]:
        """相对另一组参数的逐项差值，用于判断是否发生"质变跳变"。"""
        a, b = self.to_dict(), other.to_dict()
        return {k: a[k] - b[k] for k in a}


#: 恒等参数：求解器在此参数下的理论最优解即 A_ref 本身
DEFAULT_PARAMS = GaitParams()

#: 各参数的合法取值区间（软边界，用于裁剪 LLM 输出）
PARAM_BOUNDS: Dict[str, Tuple[float, float]] = {
    "knee_flex_scale": (0.5, 2.0),
    "trunk_lean_offset": (-15.0, 40.0),
    "cadence_scale": (0.5, 1.8),
    "double_support_offset": (-0.10, 0.25),
    "stride_length_scale": (0.4, 1.6),
    "elbow_flex_offset": (-20.0, 45.0),
    "tray_chest_distance_scale": (0.5, 1.5),
}

PARAM_NAMES = tuple(f.name for f in fields(GaitParams))
