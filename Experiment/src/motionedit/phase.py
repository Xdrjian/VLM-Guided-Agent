"""步态相位识别：足部接触 → 触地事件 → 归一化相位 φ(t) ∈ [0,1)。

Solver_Prework.md §4.4 步骤1 与 §5.3 抽帧共用本模块。

实现要点
--------
- 接触判据同时使用 **高度阈值** 与 **水平速度阈值**，比单纯高度阈值鲁棒；
- 对接触布尔序列做形态学开闭运算，滤掉 1~3 帧的抖动（原整段数据中存在
  ``(180,181)`` 这类噪声区间）；
- 相位以 **同侧连续两次触地** 为一个完整步态周期，周期内线性插值。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import torch
from torch import Tensor

__all__ = ["ContactConfig", "GaitPhase", "detect_contacts", "detect_gait_phase"]


@dataclass
class ContactConfig:
    """接触检测超参数（针对 24fps、米制单位的 Mixamo 骨架调校）。"""

    height_margin: float = 0.035      # 高于该序列最低点多少米以内算"贴地"
    speed_threshold: float = 0.55     # 水平速度上限（米/秒）
    min_contact_frames: int = 3       # 短于该长度的接触段视为噪声并剔除
    min_gap_frames: int = 2           # 短于该长度的腾空段视为噪声并填补


@dataclass
class GaitPhase:
    """步态相位识别结果。"""

    phase: Tensor                     # [T] 归一化相位 ∈ [0,1)
    left_contact: Tensor              # [T] bool
    right_contact: Tensor             # [T] bool
    heel_strikes: List[int]           # 用于定义周期的触地帧（左脚）
    cycle_frames: float               # 平均步态周期长度（帧）
    double_support_ratio: float       # 双支撑相占整个周期的比例
    valid: bool = True                # 是否成功识别出至少一个完整周期
    notes: List[str] = field(default_factory=list)

    @property
    def cadence_hz(self) -> float:
        """步频（步/秒）。一个步态周期含左右各一步，故乘 2。"""
        if self.cycle_frames <= 0:
            return 0.0
        return 2.0 * self.fps / self.cycle_frames if hasattr(self, "fps") else 0.0


# --------------------------------------------------------------------------- #
def _clean_binary(x: Tensor, min_on: int, min_off: int) -> Tensor:
    """形态学清理：填补过短的 ``False`` 段，再剔除过短的 ``True`` 段。"""
    out = x.clone()

    for target, min_len in ((False, min_off), (True, min_on)):
        runs = _runs(out, value=target)
        for s, e in runs:
            if (e - s + 1) < min_len:
                # 边界段不翻转，避免把序列首尾的真实状态抹掉
                if s == 0 or e == out.numel() - 1:
                    continue
                out[s : e + 1] = not target
    return out


def _runs(x: Tensor, value: bool) -> List[Tuple[int, int]]:
    """返回所有连续等于 ``value`` 的区间 ``[start, end]``（闭区间）。"""
    out: List[Tuple[int, int]] = []
    start: Optional[int] = None
    vals = x.tolist()
    for i, v in enumerate(vals):
        if bool(v) == value and start is None:
            start = i
        elif bool(v) != value and start is not None:
            out.append((start, i - 1))
            start = None
    if start is not None:
        out.append((start, len(vals) - 1))
    return out


def detect_contacts(
    foot_pos: Tensor,
    fps: float,
    cfg: ContactConfig = ContactConfig(),
    up_axis: int = 2,
) -> Tensor:
    """单只脚的接触检测。

    Parameters
    ----------
    foot_pos : ``[T, 3]`` 该脚参考点（通常是 ToeBase）的世界坐标。
    up_axis  : 竖直轴下标，Z-up 数据为 ``2``。
    """
    height = foot_pos[:, up_axis]
    ground = height.min()

    horiz = torch.stack([foot_pos[:, i] for i in range(3) if i != up_axis], dim=-1)
    vel = torch.zeros(foot_pos.shape[0], dtype=foot_pos.dtype, device=foot_pos.device)
    vel[1:] = (horiz[1:] - horiz[:-1]).norm(dim=-1) * fps
    vel[0] = vel[1]

    contact = (height <= ground + cfg.height_margin) & (vel <= cfg.speed_threshold)
    return _clean_binary(contact, cfg.min_contact_frames, cfg.min_gap_frames)


def detect_gait_phase(
    left_foot: Tensor,
    right_foot: Tensor,
    fps: float,
    cfg: ContactConfig = ContactConfig(),
    up_axis: int = 2,
) -> GaitPhase:
    """从双脚世界坐标轨迹识别归一化步态相位。

    Parameters
    ----------
    left_foot, right_foot : ``[T, 3]`` 世界坐标（建议用 ``LeftToeBase`` / ``RightToeBase``）。

    Returns
    -------
    :class:`GaitPhase`，其中 ``phase`` 在无法识别完整周期时退化为整段线性 0→1。
    """
    T = left_foot.shape[0]
    notes: List[str] = []

    lc = detect_contacts(left_foot, fps, cfg, up_axis)
    rc = detect_contacts(right_foot, fps, cfg, up_axis)

    # 左脚接触段的起始帧即触地（heel strike）事件
    strikes = [s for s, _ in _runs(lc, True)]
    if strikes and strikes[0] == 0:
        # 序列起始就处于接触，该段不是完整周期的起点，跳过
        strikes = strikes[1:]
        notes.append("dropped leading partial contact at frame 0")

    phase = torch.zeros(T, dtype=left_foot.dtype, device=left_foot.device)
    valid = len(strikes) >= 2

    if valid:
        cycles = [strikes[i + 1] - strikes[i] for i in range(len(strikes) - 1)]
        cycle_frames = float(sum(cycles)) / len(cycles)

        for i in range(len(strikes) - 1):
            s, e = strikes[i], strikes[i + 1]
            phase[s:e] = torch.linspace(
                0.0, 1.0, e - s + 1, dtype=phase.dtype, device=phase.device
            )[:-1]
        # 首尾不完整周期按平均周期长度外推，保证相位全程有定义
        first = strikes[0]
        if first > 0:
            back = torch.arange(first, 0, -1, dtype=phase.dtype, device=phase.device)
            phase[:first] = (1.0 - back / cycle_frames) % 1.0
        last = strikes[-1]
        if last < T:
            fwd = torch.arange(0, T - last, dtype=phase.dtype, device=phase.device)
            phase[last:] = (fwd / cycle_frames) % 1.0
    else:
        cycle_frames = float(T)
        phase = torch.linspace(0.0, 1.0, T + 1, dtype=phase.dtype, device=phase.device)[:-1]
        notes.append("insufficient heel strikes; fell back to linear phase")

    both = (lc & rc)
    ds_ratio = float(both.sum()) / max(int((lc | rc).sum()), 1)

    result = GaitPhase(
        phase=phase,
        left_contact=lc,
        right_contact=rc,
        heel_strikes=strikes,
        cycle_frames=cycle_frames,
        double_support_ratio=ds_ratio,
        valid=valid,
        notes=notes,
    )
    result.fps = fps          # type: ignore[attr-defined]
    return result


def sample_frames_by_phase(phase: Tensor, n: int = 8) -> List[int]:
    """按相位均匀抽 ``n`` 帧（Solver_Prework.md §5.3 的抽帧策略）。

    在第一个完整周期内，对目标相位 ``k/n`` 取最近邻帧。
    """
    targets = [k / n for k in range(n)]
    out: List[int] = []
    for t in targets:
        idx = int(torch.argmin((phase - t).abs()))
        out.append(idx)
    return out
