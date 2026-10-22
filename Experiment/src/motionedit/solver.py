"""第 3 层之二：轨迹优化求解器主体（Solver_Prework.md §4.7 / §4.8）。

优化变量的参数化
----------------
**不直接优化四元数**。四元数有单位模长约束，直接优化需要每步重新归一化（破坏优化器
内部的动量/曲率估计），或加拉格朗日项（引入额外超参）。

本模块采用**增量旋转向量参数化**：

.. math::
    q_{\\text{cur}}(t, b) = q_{\\text{ref}}(t, b) \\otimes \\exp(v(t, b))

其中 :math:`v \\in \\mathbb{R}^3` 是无约束的优化变量。三个好处：

1. 输出自动是单位四元数，优化器可以完全无约束地跑；
2. :math:`v = 0` 恰对应 A_ref 本身，是天然的良好初值（暖启动）；
3. :math:`\\|v\\|` 直接就是"相对参考转了多少弧度"，便于诊断与限幅。

根节点平移单独用一个 ``[T, 3]`` 的增量变量。**只有 DOF 掩码内的 21 根骨骼参与优化**，
其余 44 根（手指、末端占位骨骼等）保持参考值不变。

两阶段优化
----------
- **阶段一 Adam**：带权重退火，负责从初值走到目标附近。退火要求权重随迭代变化，
  只有一阶方法能自然适应这种非平稳目标。
- **阶段二 L-BFGS**：固定权重（取退火终值）做精修。拟牛顿法利用曲率信息，
  在终局收敛速度远快于 Adam，但要求目标函数平稳，故必须放在退火结束之后。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import torch
from torch import Tensor

from .losses import LossWeights, MotionLoss
from .motion_io import MotionSequence
from .phase import ContactConfig
from .rom import ROMPenalty, build_rom_penalty
from .rotation import quaternion_multiply, rotvec_to_quaternion
from .skeleton import Skeleton
from .targets import TargetSpec, resample_pose
from .tray import GripCalibration, calibrate_grip

__all__ = ["SolverConfig", "SolveResult", "PoseVariable", "solve"]


# --------------------------------------------------------------------------- #
@dataclass
class SolverConfig:
    """求解器超参数。"""

    adam_iters: int = 1500
    adam_lr: float = 0.01
    lbfgs_iters: int = 60
    lbfgs_lr: float = 0.5

    #: 退火在前 ``anneal_frac`` 比例的迭代内完成，其后权重保持终值不变。
    #: 必须与 ``adam_iters`` 解耦：否则提前停止会把退火卡在半路，
    #: 而紧随的 L-BFGS 固定用终值权重，两个阶段优化的就不是同一个目标。
    anneal_frac: float = 0.6

    #: 收敛判据作用在 **task** 上，total 含退火中的 w_prior/w_rom，跨迭代
    #: 不可比（实测 total 在 1426→1956→688 之间乱跳，而 task 单调下降）。
    converge_tol: float = 1e-3
    converge_patience: int = 3
    converge_check_every: int = 50

    #: 单根骨骼相对参考的最大旋转量（度）。软性安全阀，防止优化器把某个关节
    #: "转出去再绕回来"得到数值上等价、视觉上崩坏的解。
    max_rotation_deg: float = 75.0
    #: 根节点相对参考的最大平移量（米）
    max_translation_m: float = 1.5
    #: 梯度范数裁剪阈值，防止早期大梯度把解推飞
    grad_clip: float = 10.0

    log_every: int = 50
    device: str = "cpu"
    seed: int = 0

    #: 是否先按时间映射重采样参考动作作为暖启动初值（强烈建议开启，见
    #: :func:`motionedit.targets.resample_pose` 的说明）
    warm_start: bool = True

    #: 接触判据配置；``None`` 时用 :class:`motionedit.phase.ContactConfig` 默认值。
    #: 应与提取步态相位时使用的配置保持一致，否则"哪些帧算支撑"两处口径会不一致。
    contact_cfg: Optional[ContactConfig] = None

    #: 自动权重定标的目标比例；``None`` 时用 :meth:`MotionLoss.calibrate` 的默认值
    calibrate_ratios: Optional[Dict[str, float]] = None


@dataclass
class SolveResult:
    """求解结果与全过程诊断信息。"""

    pose: Tensor                                  # [T, B, 7] 最优姿态
    history: List[Dict[str, float]] = field(default_factory=list)
    final_terms: Dict[str, float] = field(default_factory=dict)
    initial_terms: Dict[str, float] = field(default_factory=dict)
    weights: Optional[LossWeights] = None
    elapsed_s: float = 0.0
    #: 解是否已到驻点。判据是“L-BFGS 也再推不动任务项”——光看 Adam 跑平
    #: 不算数：实测 Adam 跑满 600 次后，60 次 L-BFGS 还能把 task.knee 从
    #: 407 压到 174，那只是一阶方法在狭长山谷里迈不动，不是真的收敛。
    converged: bool = True
    #: 硬性失败（损失非有限 / L-BFGS 抛异常 / ROM 越界）。与欠收敛是两回事：
    #: 欠收敛只是没做到位，结果仍可用；硬性失败的输出不可用。
    failed: bool = False
    #: Adam 实际跑了多少次（提前停止时小于 adam_iters）
    adam_used: int = 0
    notes: List[str] = field(default_factory=list)

    def summary(self) -> Dict[str, object]:
        i, f = self.initial_terms, self.final_terms
        return {
            "elapsed_s": round(self.elapsed_s, 2),
            "converged": self.converged,
            "failed": self.failed,
            "adam_used": self.adam_used,
            "task": f"{i.get('task', 0):.2f} -> {f.get('task', 0):.2f}",
            "prior": f"{i.get('prior', 0):.4f} -> {f.get('prior', 0):.4f}",
            "rom": f"{i.get('rom', 0):.4f} -> {f.get('rom', 0):.4f}",
            "smooth": f"{i.get('smooth', 0):.2e} -> {f.get('smooth', 0):.2e}",
            "contact": f"{i.get('contact', 0):.2e} -> {f.get('contact', 0):.2e}",
            "grip": f"{i.get('grip', 0):.4f} -> {f.get('grip', 0):.4f}",
            "level": f"{i.get('level', 0):.4f} -> {f.get('level', 0):.4f}",
            "notes": self.notes,
        }


# --------------------------------------------------------------------------- #
class PoseVariable(torch.nn.Module):
    """把"参考姿态 + 增量"封装成可优化模块。"""

    def __init__(self, ref_pose: Tensor, mask: Tensor, root_idx: int) -> None:
        super().__init__()
        self.register_buffer("ref_pose", ref_pose.detach().clone())
        self.register_buffer("mask", mask)
        self.root_idx = root_idx
        self.active = int(mask.sum())

        t = int(ref_pose.shape[0])
        self.rotvec = torch.nn.Parameter(
            torch.zeros(t, self.active, 3, dtype=ref_pose.dtype, device=ref_pose.device))
        self.root_delta = torch.nn.Parameter(
            torch.zeros(t, 3, dtype=ref_pose.dtype, device=ref_pose.device))

    def forward(self) -> Tensor:
        pose = self.ref_pose.clone()
        dq = rotvec_to_quaternion(self.rotvec)
        q_new = quaternion_multiply(self.ref_pose[:, self.mask, :4], dq)

        # 用 index_copy 风格的拼装，避免对 Parameter 做原地写入破坏计算图
        out = pose.clone()
        idx = self.mask.nonzero().flatten()
        out = out.index_copy(
            1, idx, torch.cat([q_new, self.ref_pose[:, self.mask, 4:]], dim=-1))

        root = self.ref_pose[:, self.root_idx, 4:] + self.root_delta
        root_full = torch.cat([out[:, self.root_idx, :4], root], dim=-1).unsqueeze(1)
        out = out.index_copy(
            1, torch.tensor([self.root_idx], device=out.device), root_full)
        return out

    @torch.no_grad()
    def clamp_(self, max_rot_rad: float, max_trans_m: float) -> int:
        """把增量投影回安全球内，返回被裁剪的元素个数。"""
        n = 0
        r = self.rotvec.norm(dim=-1, keepdim=True)
        over = r > max_rot_rad
        if bool(over.any()):
            n += int(over.sum())
            self.rotvec.mul_(torch.where(over, max_rot_rad / r.clamp_min(1e-9),
                                         torch.ones_like(r)))
        d = self.root_delta.norm(dim=-1, keepdim=True)
        over_d = d > max_trans_m
        if bool(over_d.any()):
            n += int(over_d.sum())
            self.root_delta.mul_(torch.where(over_d, max_trans_m / d.clamp_min(1e-9),
                                             torch.ones_like(d)))
        return n

    @torch.no_grad()
    def stats(self) -> Dict[str, float]:
        return {
            "max_rot_deg": float(torch.rad2deg(self.rotvec.norm(dim=-1).max())),
            "mean_rot_deg": float(torch.rad2deg(self.rotvec.norm(dim=-1).mean())),
            "max_root_shift_m": float(self.root_delta.norm(dim=-1).max()),
        }


# --------------------------------------------------------------------------- #
def solve(
    seq: MotionSequence,
    skel: Skeleton,
    target: TargetSpec,
    weights: Optional[LossWeights] = None,
    cfg: Optional[SolverConfig] = None,
    rom: Optional[ROMPenalty] = None,
    callback: Optional[Callable[[int, Dict[str, float]], None]] = None,
    grip: Optional[GripCalibration] = None,
) -> SolveResult:
    """对一段参考动作求解满足 ``target`` 的新动作。

    Parameters
    ----------
    seq      : 参考动作（A_ref）
    skel     : 静止骨架
    target   : :func:`motionedit.targets.build_targets` 的输出
    weights  : 损失权重；``None`` 时自动定标（推荐）
    cfg      : 求解器超参数
    rom      : 预编译的 ROM 惩罚；``None`` 时自动构造
    callback : ``(iter, terms) -> None``，用于外层 agent 的进度上报

    Returns
    -------
    :class:`SolveResult`
    """
    cfg = cfg or SolverConfig()
    torch.manual_seed(cfg.seed)
    t_start = time.time()
    notes: List[str] = list(target.notes)

    dev = torch.device(cfg.device)

    # 抓握标定只能来自**带真实托盘轨道的参考动作**；没有托盘时两项约束自动关闭。
    # 必须在搬迁之前做：seq 由调用方持有、通常在 CPU 上，而 skel 马上要去 cfg.device
    if grip is None and seq.objects:
        try:
            grip = calibrate_grip(seq, skel.to(seq.device))
            notes.append(f"抓握标定：残差 {grip.residual_m * 1000:.2f} mm，"
                         f"参考倾角峰值 {grip.tilt_max_deg:.2f}°")
        except ValueError as e:
            notes.append(f"抓握标定失败，托盘约束已禁用：{e}")
            grip = None

    # 姿态 / 骨架 / 目标曲线 / ROM / 抓握标定必须同在一处设备，否则 CUDA 下会逐个报错
    skel = skel.to(dev)
    target = target.to(dev)
    ref_pose = seq.pose.to(dev)
    rom = (rom or build_rom_penalty(skel)).to(dev)
    if grip is not None:
        grip = grip.to(dev)

    # 暖启动：先把参考动作按时间映射整体拉伸/压缩到新时间轴，
    # 剩下的残差才交给优化器。同时用它作为姿势先验的锚点——否则先验会要求
    # "保持原时间轴上的姿势"，与任务项的时间平移要求直接对抗。
    anchor = resample_pose(ref_pose, target.src_idx) if cfg.warm_start else \
        resample_pose(ref_pose, torch.arange(
            target.num_frames, device=dev, dtype=ref_pose.dtype
        ).clamp(max=ref_pose.shape[0] - 1.0))
    if cfg.warm_start:
        notes.append("已启用时间映射暖启动")

    var = PoseVariable(anchor, skel.dof_mask().to(dev), skel.index("mixamorig:Hips")).to(dev)
    loss_fn = MotionLoss(skel, anchor, target, rom, weights,
                         fps=seq.fps, contact_cfg=cfg.contact_cfg, grip=grip)

    # 初始点自动定标（仅当调用方未显式给权重时）
    if weights is None:
        loss_fn.calibrate(var(), cfg.calibrate_ratios)
        notes.append("权重已按初始点自动定标")

    with torch.no_grad():
        initial = loss_fn(var(), 0.0).to_dict()

    history: List[Dict[str, float]] = []
    max_rot = torch.deg2rad(torch.tensor(cfg.max_rotation_deg)).item()

    def record(it: int, terms, tag: str) -> None:
        row = terms.to_dict()
        row.update({"iter": it, "stage": tag})  # type: ignore[dict-item]
        history.append(row)
        if callback:
            callback(it, row)

    # --- 阶段一：Adam + 权重退火 ---------------------------------------- #
    opt = torch.optim.Adam(var.parameters(), lr=cfg.adam_lr)
    clamped_total = 0
    anneal_span = max(cfg.anneal_frac * (cfg.adam_iters - 1), 1.0)
    prev_task: Optional[float] = None
    stale = 0
    adam_used = cfg.adam_iters
    for it in range(cfg.adam_iters):
        progress = min(it / anneal_span, 1.0)
        opt.zero_grad(set_to_none=True)
        terms = loss_fn(var(), progress)
        terms.total.backward()

        if not torch.isfinite(terms.total):
            notes.append(f"第 {it} 次迭代出现非有限损失，已提前终止")
            return SolveResult(
                pose=var().detach(), history=history, initial_terms=initial,
                final_terms=terms.to_dict(), weights=loss_fn.w,
                elapsed_s=time.time() - t_start, converged=False, failed=True,
                adam_used=it, notes=notes)

        torch.nn.utils.clip_grad_norm_(var.parameters(), cfg.grad_clip)
        opt.step()
        clamped_total += var.clamp_(max_rot, cfg.max_translation_m)

        if it % cfg.log_every == 0 or it == cfg.adam_iters - 1:
            record(it, terms, "adam")

        # 退火未完成时目标函数还在变，谈不上停不停
        if progress >= 1.0 and it % cfg.converge_check_every == 0:
            cur = float(terms.task)
            if prev_task is not None:
                gain = (prev_task - cur) / max(abs(prev_task), 1e-12)
                stale = stale + 1 if gain < cfg.converge_tol else 0
                if stale >= cfg.converge_patience:
                    adam_used = it + 1
                    notes.append(
                        f"Adam 在第 {it} 次提前停止：任务项连续 "
                        f"{cfg.converge_patience} 次检查相对改善 < {cfg.converge_tol:g}")
                    record(it, terms, "adam")
                    break
            prev_task = cur

    if clamped_total:
        notes.append(f"增量安全阀触发 {clamped_total} 次（参数可能过于激进）")

    with torch.no_grad():
        pre_lbfgs = float(loss_fn(var(), 1.0).task)

    # --- 阶段二：L-BFGS 精修（固定权重为退火终值） ----------------------- #
    lbfgs = torch.optim.LBFGS(
        var.parameters(), lr=cfg.lbfgs_lr, max_iter=cfg.lbfgs_iters,
        history_size=20, line_search_fn="strong_wolfe")
    holder: Dict[str, object] = {}

    def closure() -> Tensor:
        lbfgs.zero_grad(set_to_none=True)
        terms = loss_fn(var(), 1.0)
        terms.total.backward()
        holder["terms"] = terms
        return terms.total

    failed = False
    try:
        lbfgs.step(closure)  # type: ignore[arg-type]
    except RuntimeError as e:          # 线搜索失败等
        failed = True
        notes.append(f"L-BFGS 提前结束（{e}）；已保留 Adam 阶段结果")

    var.clamp_(max_rot, cfg.max_translation_m)
    if "terms" in holder:
        record(adam_used, holder["terms"], "lbfgs")  # type: ignore[arg-type]

    pose = var().detach()
    with torch.no_grad():
        final = loss_fn(pose, 1.0).to_dict()

    # 健康检查：ROM 越界是硬性物理错误，必须显式告警
    if final.get("rom", 0.0) > 1.0:
        failed = True
        notes.append(f"最终解存在 ROM 越界（rom={final['rom']:.2f}），"
                     f"建议调低参数激进程度或提高 w_rom")

    # 真正的收敛判据：连二阶方法都再推不动任务项，才算到了驻点
    lbfgs_gain = (pre_lbfgs - final.get("task", pre_lbfgs)) / max(abs(pre_lbfgs), 1e-12)
    converged = (not failed) and lbfgs_gain < cfg.converge_tol
    if not converged and not failed:
        notes.append(
            f"未收敛：L-BFGS 又把任务项降了 {lbfgs_gain * 100:.1f}%"
            f"（{pre_lbfgs:.1f} -> {final.get('task', 0):.1f}），说明解还没到驻点；"
            f"结果仍可用，但跨轮比较会混入优化器差异，建议加大 adam_iters")

    notes.append(f"增量幅度 {var.stats()}")
    return SolveResult(
        pose=pose, history=history, initial_terms=initial, final_terms=final,
        weights=loss_fn.w, elapsed_s=time.time() - t_start,
        converged=converged, failed=failed, adam_used=adam_used, notes=notes)
