"""第 4 层 · 闭环：把求解器包装成外层循环的 ``SolveFn`` 注入点。

职责
----
把 LLM 给出的一组参数变成一条落盘的动作，并**如实报告它到底被执行成了什么样**。
后半句才是关键：workflow.md §4.4 要求把"指令 vs 实际"的偏差回传给 LLM，
否则 LLM 只能靠 VLM 的语义评价盲猜方向。

求解器执行失效（§4.4）
----------------------
判据分两类，任一成立即标记本轮失效：

1. **指令跟踪偏差**——七个参数各自对应一个可测指标，逐项比对"期望值 vs 实际值"。
   期望值由参考动作的特征按该参数的语义推出（乘性缩放就乘、加性偏置就加）。
2. **物理有效性**——ROM 越界、抓握脱手、托盘倾倒、求解器未收敛。

失效**不立即终止**，而是先告警：单次失效更可能是 LLM 给了一组求解器够不着的
极端参数（如 knee_flex_scale 顶到 ROM 上限），此时正确的反应是让 LLM 退回来，
而不是整个实验停摆。连续 :attr:`ExecCheckConfig.max_consecutive` 次才判定为
求解器真的坏了（由 :class:`MotionSolveFn` 自行计数）。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from .collision import fit_capsules
from .features import GaitFeatures, extract_features
from .motion_io import MotionSequence, load_motion, save_motion
from .outer_loop import SolveOutput
from .params import GaitParams
from .skeleton import Skeleton
from .solver import SolverConfig, solve
from .targets import build_targets, resample_objects
from .tray import GripCalibration, calibrate_grip, update_tray_in_sequence

__all__ = ["ExecCheckConfig", "TrackingRow", "MotionSolveFn"]


@dataclass
class ExecCheckConfig:
    """求解器执行失效的判据。全部为 🟡 经验值，跑几轮有数据后再收紧。"""

    #: 单项指令跟踪的相对误差上限
    rel_tol: float = 0.15
    #: 连续多少轮失效才真正终止流程（单次失效只告警）
    max_consecutive: int = 2
    #: ROM 惩罚上限（度²）。参考动作上实测为 0
    rom_tol: float = 1.0
    #: 抓握闭链残差上限。参考动作约 1.84 mm，恒等求解后为 0
    grip_tol: float = 5.0
    #: 托盘倾角 hinge 上限（度²）
    level_tol: float = 25.0
    #: 肢间穿插上限。**与 ROM 同级的物理有效性判据**：exp07 round 4 的双膝
    #: 穿模在旧审计下全部“合格”，因为没有任何一项在看全局碰撞。
    #: 参考动作上恒为 0，故任何正值都是“比参考更挤”。
    collision_tol: float = 1.0
    #: 欠收敛是否要记一笔。它不计为失效，但各轮收敛程度不一致会让
    #: VLM 看到的差异里混入优化器运气，是个必须可追溯的混淆变量。
    require_converged: bool = True


@dataclass
class TrackingRow:
    """单个指标的"指令 → 实际"对照，直接喂给 LLM。"""

    name: str
    commanded: float
    expected: float
    actual: float
    #: 分母下限。有符号角度指标（如屈髋）可能跨零，此时相对误差毫无意义，
    #: 需改用绝对度数衡量；按度计的指标给 5.0，其余沿用极小值
    floor: float = 1e-3

    @property
    def rel_error(self) -> float:
        # 分母同时看期望值与实际值，避免期望值接近 0 时相对误差爆炸
        denom = max(abs(self.expected), abs(self.actual), self.floor)
        return abs(self.actual - self.expected) / denom

    def to_dict(self) -> Dict[str, float]:
        return {"commanded": round(self.commanded, 4),
                "expected": round(self.expected, 4),
                "actual": round(self.actual, 4),
                "rel_error": round(self.rel_error, 4)}


# --------------------------------------------------------------------------- #
def _expected(ref: GaitFeatures, p: GaitParams) -> Dict[str, Tuple[float, float, float]]:
    """``{指标名: (下达的参数值, 期望达成值, 参考值)}``。

    只收录**真正可测且与某个参数一一对应**的指标；``double_support_offset``
    作用于周期内部的时间疏密，用标量特征衡量不可靠，故不进入跟踪判据。

    膝屈曲同时收录**支撑相均值**与全局峰值：只看峰值会发假绿灯——
    峰值只出现在摆动相，“小腿往后勾”这种畸形解恰好能把它做达标，
    而承载负重感的支撑相纹丝不动。
    """
    out: Dict[str, Tuple[float, float, float]] = {}

    def add(name: str, cmd: float, exp: Optional[float], base: Optional[float]) -> None:
        if exp is not None and base is not None:
            out[name] = (cmd, float(exp), float(base))

    for side in ("left", "right"):
        k_st = getattr(ref, f"knee_flex_{side}_stance_mean")
        k_max = getattr(ref, f"knee_flex_{side}_max")
        add(f"knee_flex_{side}_stance_mean", p.knee_flex_scale,
            k_st * p.knee_flex_scale, k_st)
        add(f"knee_flex_{side}_max", p.knee_flex_scale,
            k_max * p.knee_flex_scale, k_max)
        h_st = getattr(ref, f"hip_flex_{side}_stance_mean")
        add(f"hip_flex_{side}_stance_mean", p.hip_flex_offset,
            h_st + p.hip_flex_offset, h_st)
    add("trunk_lean_mean", p.trunk_lean_offset,
        ref.trunk_lean_mean + p.trunk_lean_offset, ref.trunk_lean_mean)
    add("elbow_flex_left_mean", p.elbow_flex_offset,
        ref.elbow_flex_left_mean + p.elbow_flex_offset, ref.elbow_flex_left_mean)
    add("elbow_flex_right_mean", p.elbow_flex_offset,
        ref.elbow_flex_right_mean + p.elbow_flex_offset, ref.elbow_flex_right_mean)
    add("cadence_steps_per_s", p.cadence_scale,
        ref.cadence_steps_per_s * p.cadence_scale, ref.cadence_steps_per_s)
    add("stride_length_m", p.stride_length_scale,
        ref.stride_length_m * p.stride_length_scale, ref.stride_length_m)
    if ref.tray_chest_distance_mean is not None:
        add("tray_chest_distance_mean", p.tray_chest_distance_scale,
            ref.tray_chest_distance_mean * p.tray_chest_distance_scale,
            ref.tray_chest_distance_mean)
    return out


# --------------------------------------------------------------------------- #
class MotionSolveFn:
    """实现 :class:`outer_loop.SolveFn` 协议。

    参考动作、骨架、抓握标定都只加载/标定一次，跨轮复用。

    Example
    -------
    >>> fn = MotionSolveFn(motion_path, skeleton_path)
    >>> out = fn(GaitParams(knee_flex_scale=1.25), run_dir / "rounds" / "round_001")
    """

    def __init__(self, motion_path: Path | str, skeleton_path: Path | str,
                 tray_name: str = "draw_organizer_tray",
                 solver_cfg: Optional[SolverConfig] = None,
                 check: Optional[ExecCheckConfig] = None) -> None:
        self.skel = Skeleton.from_json(skeleton_path)
        self.seq = load_motion(motion_path, bone_order=self.skel.bone_names)
        self.tray_name = tray_name
        self.solver_cfg = solver_cfg or SolverConfig()
        self.check = check or ExecCheckConfig()

        self.ref_features, self.curves = extract_features(
            self.seq, self.skel, tray_name=tray_name)
        self.grip: Optional[GripCalibration] = None
        if tray_name in self.seq.objects:
            self.grip = calibrate_grip(self.seq, self.skel, tray_name)
        # 胶囊体只与静止骨架有关，拟合一次跨轮复用
        self.capsules = fit_capsules(skeleton_path, self.skel)
        self._consecutive_failures = 0

    # ------------------------------------------------------------------ #
    def __call__(self, params: GaitParams, out_dir: Path) -> SolveOutput:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        p = params.clipped()

        target = build_targets(self.curves, p, self.ref_features.tray_chest_distance_mean)
        res = solve(self.seq, self.skel, target, cfg=self.solver_cfg,
                    grip=self.grip, capsules=self.capsules)

        out = MotionSequence(
            pose=res.pose.detach().cpu(), bone_names=self.seq.bone_names,
            objects=resample_objects(self.seq.objects, target.src_idx.cpu()),
            fps=self.seq.fps, meta=self.seq.meta,
            object_models=self.seq.object_models)
        if self.grip is not None:
            out = update_tray_in_sequence(out, self.skel, self.grip)

        feats, _ = extract_features(out, self.skel, tray_name=self.tray_name)
        motion_path = save_motion(out, out_dir / "motion.pkl")

        rows, violations, notes = self._audit(p, feats, res)
        failed = bool(violations)
        self._consecutive_failures = self._consecutive_failures + 1 if failed else 0
        exec_failure = self._consecutive_failures >= self.check.max_consecutive

        report = {
            "params": p.to_dict(),
            "target": target.summary(),
            "solve": res.summary(),
            "ref_features": self.ref_features.to_dict(),
            "new_features": feats.to_dict(),
            "tracking": {k: v.to_dict() for k, v in rows.items()},
            "violations": violations,
            "notes": notes,
            "tracking_failed": failed,
            "consecutive_failures": self._consecutive_failures,
            "exec_failure": exec_failure,
        }
        (out_dir / "solve_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

        return SolveOutput(
            motion_path=motion_path,
            features=self._llm_features(feats, rows),
            status=("exec_failure" if exec_failure else
                    ("tracking_warning" if failed else
                     ("converged" if res.converged else "not_converged"))),
            # 只回传真正的失效告警；"已启用暖启动"这类说明性 notes 对 LLM 是噪声
            warnings=violations,
            exec_failure=exec_failure,
        )

    # ------------------------------------------------------------------ #
    def _audit(self, p: GaitParams, feats: GaitFeatures, res: Any
               ) -> Tuple[Dict[str, TrackingRow], List[str], List[str]]:
        cfg = self.check
        actual = feats.to_dict()
        rows: Dict[str, TrackingRow] = {}
        violations: List[str] = []

        for name, (cmd, exp, _base) in _expected(self.ref_features, p).items():
            got = actual.get(name)
            if got is None:
                continue
            # 角度类指标可能跨零，相对误差分母须有度数量级的地板
            floor = 5.0 if ("flex" in name or "lean" in name) else 1e-3
            row = TrackingRow(name, cmd, exp, float(got), floor)
            rows[name] = row
            if row.rel_error > cfg.rel_tol:
                violations.append(
                    f"指令跟踪偏差：{name} 期望 {exp:.3f}、实际 {row.actual:.3f}"
                    f"（相对误差 {row.rel_error * 100:.1f}% > {cfg.rel_tol * 100:.0f}%）")

        terms = res.final_terms
        notes = list(res.notes)
        if getattr(res, "failed", False):
            violations.append("求解器硬性失败（损失非有限 / L-BFGS 异常 / ROM 越界）")
        elif cfg.require_converged and not res.converged:
            # 欠收敛不是失效：结果仍可用，其实际后果已由 rel_tol 跟踪误差兑现。
            # 但必须记下来：各轮收敛程度不一致会把优化器差异混进 VLM 的比较里。
            notes.append("求解器未达驻点（仅提示，不计为失效）")
        for key, tol, label in (("rom", cfg.rom_tol, "ROM 越界"),
                                ("grip", cfg.grip_tol, "抓握脱手"),
                                ("level", cfg.level_tol, "托盘倾倒"),
                                ("collision", cfg.collision_tol, "肢体自穿插")):
            val = float(terms.get(key, 0.0))
            if val > tol:
                violations.append(f"{label}：{key}={val:.4f} > {tol}")
        return rows, violations, notes

    @staticmethod
    def _llm_features(feats: GaitFeatures, rows: Dict[str, TrackingRow]
                      ) -> Dict[str, float]:
        """传给 LLM 的特征：实际值 + 每项的跟踪相对误差。

        只给实际值不够——LLM 需要知道"我下达的指令有没有被执行到位"，
        这正是 §4.4 要求把求解器反馈回传的原因。
        """
        out = {k: v for k, v in feats.to_dict().items()
               if isinstance(v, (int, float))}
        for name, row in rows.items():
            out[f"{name}__rel_error"] = round(row.rel_error, 4)
        return out
