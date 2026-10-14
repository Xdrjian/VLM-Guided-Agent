"""第 3 层之一：可微的损失函数组装（Solver_Prework.md §4）。

总损失
------
.. math::
    L = L_{\\text{task}}
      + w_{\\text{prior}}(k)\\, L_{\\text{prior}}
      + w_{\\text{ROM}}(k)\\, L_{\\text{ROM}}
      + w_{\\text{smooth}}\\, L_{\\text{smooth}}
      + w_{\\text{contact}}\\, L_{\\text{contact}}

五项各司其职（缺一不可）：

============  ==========================================  ==============================
项            约束什么                                    不加会怎样
============  ==========================================  ==============================
task          逐帧达到目标角度 / 根节点轨迹                优化器没有前进方向
prior         整体风格不要偏离 A_ref 太远                  欠约束，解出反直觉的等效姿态
ROM           不违反人体关节解剖极限                        膝盖反折等物理不可能姿势
smooth        时间上连续，无高频抖动                        逐帧独立优化导致画面跳变
contact       支撑脚不滑动、不穿地                          **脚底打滑**（实测必现）
============  ==========================================  ==============================

关于 contact 项
---------------
最初的实现没有这一项，实测结果：脚趾高度跑到 ``-0.024 m``（穿进地面）与 ``0.297 m``
（凌空），且全程没有任何一帧速度归零——整段动作在"滑冰"。原因是任务项只约束关节
角度与根节点轨迹，**完全没有提到脚**；当根节点位移被 ``stride_length_scale`` 缩短
后，没有任何机制告诉优化器"支撑脚必须钉在地上"。

脚底打滑是动作编辑中最刺眼的瑕疵之一，VLM 一眼就能看出来，故补为第五项。

权重退火（§4.8）
----------------
- ``w_ROM`` 指数**递增**：早期允许探索、避免一开始就被限位卡死，后期强制回到合法区
- ``w_prior`` 指数**递减**：早期紧贴 A_ref 提供良好初值，后期放开让任务项充分表达
- ``w_smooth`` 恒定
- ``L_task`` 权重恒为 1（作为其余权重的标度基准）

角度域 vs 四元数域
------------------
任务项在**几何角度域**比较（三点夹角），与 §1.5 的 ROM 结论同源：本数据的骨骼
局部坐标轴与解剖学轴无对应关系，把目标角度反推成四元数需要一个在本数据上不成立
的轴向假设。姿势先验项则在**四元数域**比较（平方弦距离），因为它度量的本就是
"整体旋转差异"，无需解剖学解释。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
from torch import Tensor

from .features import joint_angle_3pt, trunk_lean_angle
from .phase import ContactConfig
from .rom import ROMPenalty
from .rotation import quaternion_geodesic_sq
from .skeleton import Skeleton
from .targets import TargetSpec
from .tray import GripCalibration, tray_frames_from_hands, tray_normal_from_frames

__all__ = ["LossWeights", "MotionLoss", "differentiable_curves"]

_UP = 2  # 世界坐标 Z 轴朝上
_SQRT2 = 2.0 ** 0.5


# --------------------------------------------------------------------------- #
def differentiable_curves(pose: Tensor, skel: Skeleton,
                         world: Optional[Tensor] = None) -> Dict[str, Tensor]:
    """从姿态张量直接算出任务项需要的角度/位置曲线，全程保持梯度。

    与 :func:`motionedit.features.extract_features` 计算的是同一批量，但后者会做
    ``float()`` 转换用于报告，梯度会断掉，故优化循环中必须用本函数。

    ``world`` 为已算好的 FK 结果，传入可避免重复计算（FK 是 65 根骨骼的
    Python 循环，是每次前向的主要开销）。
    """
    joints = (skel.forward_kinematics(pose) if world is None else world)[..., :3, 3]

    def J(name: str) -> Tensor:
        return joints[:, skel.index(name)]

    return {        "knee_flex_left": joint_angle_3pt(
            J("mixamorig:LeftUpLeg"), J("mixamorig:LeftLeg"), J("mixamorig:LeftFoot")),
        "knee_flex_right": joint_angle_3pt(
            J("mixamorig:RightUpLeg"), J("mixamorig:RightLeg"), J("mixamorig:RightFoot")),
        "elbow_flex_left": joint_angle_3pt(
            J("mixamorig:LeftArm"), J("mixamorig:LeftForeArm"), J("mixamorig:LeftHand")),
        "elbow_flex_right": joint_angle_3pt(
            J("mixamorig:RightArm"), J("mixamorig:RightForeArm"), J("mixamorig:RightHand")),
        "trunk_lean": trunk_lean_angle(J("mixamorig:Hips"), J("mixamorig:Spine2")),
        "root_world": J("mixamorig:Hips"),
        "chest_world": J("mixamorig:Spine2"),
        "hand_left": J("mixamorig:LeftHand"),
        "hand_right": J("mixamorig:RightHand"),
    }


# --------------------------------------------------------------------------- #
@dataclass
class LossWeights:
    """各项权重及其退火调度。

    默认数值是**量纲对齐后的初值**：角度项以"度"为单位、位置项以"米"为单位，
    二者数值尺度相差约两个量级，故 ``w_root`` 取较大值把米换算到可比尺度。
    实际使用前建议调用 :meth:`MotionLoss.calibrate` 用第一次前向的真实数值自动定标。
    """

    # --- 任务项内部的分项权重（相对，总体标度固定为 1） ---
    w_knee: float = 1.0
    w_trunk: float = 1.0
    w_elbow: float = 0.5
    w_root: float = 300.0          # 位置项（米²）→ 角度项（度²）的量纲换算

    # --- 三个正则项的基准权重 ---
    w_prior: float = 1.0
    w_rom: float = 1.0
    w_smooth: float = 1.0
    w_contact: float = 1.0

    # --- 托盘相关的两项物理有效性约束（workflow.md 的 A 类，不开放给 LLM）---
    #: 闭链抓握：双手各自预测的托盘位姿不得分开。单位已换算到 deg²
    w_grip: float = 1.0
    #: 托盘保持水平：倒过头就不是合法的搬运动作。单边 hinge，deg²
    w_level: float = 1.0
    #: 抓握位置误差的换算率：1° ≙ 5 mm。选 5 mm 是因为参考动作的
    #: 抓握残差实测 1.84 mm，超过 5 mm 就已经是真实脱手而非测量噪声
    grip_deg_per_m: float = 200.0
    #: 托盘允许的倾角余量（度），叠加在参考动作峰值之上。
    #: 只罚"比参考动作做得更差"，不要求做到绝对水平（A_ref 自身就有 3.31°）
    level_margin_deg: float = 0.5

    #: contact 项内部三个分量的相对配比：穿地 / 防滑 / 锚定
    contact_penetration: float = 1.0
    contact_slip: float = 1.0
    contact_drift: float = 0.5

    # --- 退火（k 为当前迭代占总迭代的比例 ∈ [0,1]） ---
    prior_decay: float = 0.1       # w_prior(k) = w_prior · decay^k
    rom_growth: float = 20.0       # w_rom(k)   = w_rom   · growth^k

    def at(self, k: float) -> "LossWeights":
        """返回进度 ``k`` 处退火后的权重（不修改自身）。"""
        k = min(max(k, 0.0), 1.0)
        out = LossWeights(**self.__dict__)
        out.w_prior = self.w_prior * (self.prior_decay ** k)
        out.w_rom = self.w_rom * (self.rom_growth ** k)
        return out


@dataclass
class LossTerms:
    """一次前向的各项数值，用于日志与权重定标。"""

    total: Tensor
    task: Tensor
    prior: Tensor
    rom: Tensor
    smooth: Tensor
    contact: Tensor
    grip: Tensor
    level: Tensor
    task_parts: Dict[str, Tensor] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, float]:
        f = lambda x: float(x.detach())  # noqa: E731
        d = {
            "total": f(self.total), "task": f(self.task),
            "prior": f(self.prior), "rom": f(self.rom), "smooth": f(self.smooth),
            "contact": f(self.contact), "grip": f(self.grip), "level": f(self.level),
        }
        d.update({f"task.{k}": f(v) for k, v in self.task_parts.items()})
        return d


# --------------------------------------------------------------------------- #
class MotionLoss:
    """把参考动作、目标曲线、ROM 约束组装成一个可调用的损失函数。

    Example
    -------
    >>> loss_fn = MotionLoss(skel, ref_pose, target, rom_penalty)
    >>> terms = loss_fn(pose, progress=0.3)
    >>> terms.total.backward()
    """

    def __init__(
        self,
        skel: Skeleton,
        ref_pose: Tensor,
        target: TargetSpec,
        rom: ROMPenalty,
        weights: Optional[LossWeights] = None,
        dof_mask: Optional[Tensor] = None,
        fps: float = 24.0,
        contact_cfg: Optional[ContactConfig] = None,
        grip: Optional[GripCalibration] = None,
    ) -> None:
        self.skel = skel
        self.ref_pose = ref_pose.detach()
        self.target = target
        self.rom = rom
        self.w = weights or LossWeights()
        self.fps = float(fps)
        self.contact_cfg = contact_cfg or ContactConfig()
        self.grip = grip.to(ref_pose.device) if grip is not None else None

        mask = skel.dof_mask() if dof_mask is None else dof_mask
        self.mask = mask.to(ref_pose.device)
        self.num_frames = int(ref_pose.shape[0])

        # --- 接触项的预计算（基于参考/暖启动姿态，优化过程中固定不变） --- #
        self.foot_bones = [
            "mixamorig:LeftToeBase", "mixamorig:RightToeBase",
            "mixamorig:LeftFoot", "mixamorig:RightFoot",
        ]
        cc = self.contact_cfg
        with torch.no_grad():
            self.foot_idx = torch.tensor(
                [skel.index(b) for b in self.foot_bones], device=ref_pose.device)
            ref_joints = skel.joint_positions(self.ref_pose)
            self.ref_foot = ref_joints[:, self.foot_idx]                # [T,4,3]
            # 地面高度取参考动作中足部的最低点（数据未必恰好以 z=0 为地面）
            self.ground_z = float(self.ref_foot[..., _UP].min())
            # 接触判据复用 phase.ContactConfig，保证与步态识别口径一致
            h = self.ref_foot[..., _UP] - self.ground_z                 # [T,4]
            v = torch.zeros_like(h)
            v[1:] = (self.ref_foot[1:, :, :2] - self.ref_foot[:-1, :, :2]).norm(dim=-1)
            v[0] = v[1]
            self.contact_mask = (
                (h < cc.height_margin) & (v * self.fps < cc.speed_threshold)
            ).to(ref_pose.dtype)

            # 落脚点错开：缩短步长要求落脚点整体收拢，若锚定仍用暖启动的原落脚点，
            # contact.drift 会与 task.root 直接对抗。实测 stride_length_scale=0.6 时
            # 求解器选择满足 contact（=0）而把 task.root 留在 50.3，步长参数完全失效。
            root_ref = ref_joints[:, skel.index("mixamorig:Hips")]
            shift = target.root_world.to(ref_pose.device) - root_ref
            shift = shift.clone()
            shift[:, _UP] = 0.0                                      # 只平移水平分量
            self.ref_foot = self.ref_foot + self._hold_per_stance(shift, self.contact_mask)

    @staticmethod
    def _hold_per_stance(shift: Tensor, mask: Tensor) -> Tensor:
        """把逐帧平移量在**每一段支撑相内保持恒定**，返回 ``[T,K,3]``。

        不能直接逐帧加：那会让支撑脚在踩地期间跟着根节点滑动。正确语义是
        "落脚点整体改位置，但落地后仍然不动"，故取每段接触起始帧的平移量。
        """
        T, K = mask.shape
        out = shift.unsqueeze(1).repeat(1, K, 1)
        for k in range(K):
            t = 0
            while t < T:
                if mask[t, k] > 0:
                    s = t
                    while t < T and mask[t, k] > 0:
                        t += 1
                    out[s:t, k] = shift[s]
                else:
                    t += 1
        return out

    # ------------------------------------------------------------------ #
    def _contact(self, pose: Tensor, world: Optional[Tensor] = None) -> Tensor:
        """支撑脚不滑动、不穿地。

        三个分量：
        1. **穿地**：任何一帧脚低于地面都惩罚（单边，抬脚不罚）；
        2. **防滑**：在参考动作判定为"支撑"的帧上，惩罚脚的水平位移速度；
        3. **锚定**：同样在支撑帧上，惩罚脚相对参考位置的漂移，防止整只脚
           以匀速平移的方式"合法地"滑过去（只罚速度时，恒定漂移量罚不到）。
        """
        joints = (self.skel.forward_kinematics(pose) if world is None else world)[..., :3, 3]
        pos = joints[:, self.foot_idx]                                  # [T,4,3]
        w = self.w
        penetration = torch.relu(self.ground_z - pos[..., _UP]).pow(2).mean()

        m = self.contact_mask
        if float(m.sum()) < 1.0:
            return w.contact_penetration * penetration

        slip = pose.new_zeros(())
        if self.num_frames > 1:
            vel = (pos[1:, :, :2] - pos[:-1, :, :2]).norm(dim=-1)       # [T-1,4]
            mv = m[1:] * m[:-1]                                          # 两帧都在支撑才算
            slip = (vel.pow(2) * mv).sum() / mv.sum().clamp_min(1.0)

        drift = ((pos - self.ref_foot).pow(2).sum(-1) * m).sum() / m.sum()
        return (w.contact_penetration * penetration
                + w.contact_slip * slip
                + w.contact_drift * drift)

    # ------------------------------------------------------------------ #
    def _task(self, cur: Dict[str, Tensor],
              tray_origin: Optional[Tensor]) -> tuple[Tensor, Dict[str, Tensor]]:
        t, w = self.target, self.w
        parts = {
            "knee": w.w_knee * (
                (cur["knee_flex_left"] - t.knee_flex_left).pow(2).mean()
                + (cur["knee_flex_right"] - t.knee_flex_right).pow(2).mean()),
            "trunk": w.w_trunk * (cur["trunk_lean"] - t.trunk_lean).pow(2).mean(),
            "elbow": w.w_elbow * (
                (cur["elbow_flex_left"] - t.elbow_flex_left).pow(2).mean()
                + (cur["elbow_flex_right"] - t.elbow_flex_right).pow(2).mean()),
            "root": w.w_root * (cur["root_world"] - t.root_world).pow(2).sum(-1).mean(),
        }
        # 托盘-胸骨距离：用**真实托盘原点**而非双手中点，参数语义才名实相符
        if t.tray_chest_distance is not None:
            ref = tray_origin if tray_origin is not None else \
                0.5 * (cur["hand_left"] + cur["hand_right"])
            d = (ref[:, :2] - cur["chest_world"][:, :2]).norm(dim=-1)
            parts["tray"] = w.w_root * (d - t.tray_chest_distance).pow(2).mean()
        return sum(parts.values()), parts  # type: ignore[return-value]

    def _prior(self, pose: Tensor) -> Tensor:
        """姿势先验：仅在可优化骨骼上度量与 A_ref 的四元数平方弦距离。"""
        g = quaternion_geodesic_sq(pose[:, self.mask, :4], self.ref_pose[:, self.mask, :4])
        return g.mean()

    # ------------------------------------------------------------------ #
    def _tray(self, pose: Tensor,
              world: Optional[Tensor] = None) -> tuple[Tensor, Tensor, Optional[Tensor]]:
        """抓握闭链与托盘水平两项，顺带返回托盘原点供任务项复用。

        闭链：两只手扣在同一个刚体上，各自推出的托盘位姿必须重合。
        之前没有这一项，实测右手会偏离其抓握点达 352 mm，双手间距从
        0.579 m 塌缩到 0.31 m，而托盘宽 0.493 m——手落到盘面正中央造成穿模。

        水平：腕关节在任务项里是个死自由度（实测仅动 0.82°），而手的世界朝向
        是整条链的乘积；``elbow_flex_offset`` 一动前臂就转，托盘实测倾倒 ~30°。
        加上本项后，求解器会自发用腕关节去补偿——正是真人端盘时做的事。
        """
        zero = pose.new_zeros(())
        if self.grip is None:
            return zero, zero, None

        w = self.w
        frames = tray_frames_from_hands(
            self.skel.forward_kinematics(pose) if world is None else world,
            self.skel, self.grip)                                       # [K,T,4,4]
        pos, rot = frames[:, :, :3, 3], frames[:, :, :3, :3]
        origin = pos.mean(0)

        # 位置分歧换算成度，旋转分歧由 Frobenius 距离近似成角度，两者同量纲
        d_pos = (pos - origin.unsqueeze(0)).norm(dim=-1).sum(0)
        d_rot = torch.rad2deg(
            (rot - rot.mean(0, keepdim=True)).flatten(2).norm(dim=-1).sum(0) / _SQRT2)
        grip = ((d_pos * w.grip_deg_per_m).pow(2) + d_rot.pow(2)).mean()

        normal = tray_normal_from_frames(frames, self.grip)
        up = torch.zeros_like(normal)
        up[:, _UP] = 1.0
        # 用 asin(|n × up|) 而非 acos(n·up)：后者在完全水平处梯度发散
        sin_t = torch.linalg.cross(normal, up, dim=-1).norm(dim=-1)
        tilt = torch.rad2deg(torch.asin(sin_t.clamp(max=1.0 - 1e-6)))
        allow = self.grip.tilt_max_deg + w.level_margin_deg
        level = torch.relu(tilt - allow).pow(2).mean()
        return grip, level, origin

    def _smooth(self, pose: Tensor) -> Tensor:
        """二阶差分：惩罚加速度，抑制高频抖动（一阶差分会把正常运动也压掉）。"""
        if self.num_frames < 3:
            return pose.new_zeros(())
        q = pose[:, self.mask, :4]
        acc_q = q[2:] - 2.0 * q[1:-1] + q[:-2]
        root = pose[:, self.skel.index("mixamorig:Hips"), 4:]
        acc_r = root[2:] - 2.0 * root[1:-1] + root[:-2]
        return acc_q.pow(2).mean() + acc_r.pow(2).mean()

    # ------------------------------------------------------------------ #
    def __call__(self, pose: Tensor, progress: float = 0.0) -> LossTerms:
        w = self.w.at(progress)
        world = self.skel.forward_kinematics(pose)
        cur = differentiable_curves(pose, self.skel, world)

        grip, level, tray_origin = self._tray(pose, world)
        task, parts = self._task(cur, tray_origin)
        prior = self._prior(pose)
        rom = self.rom(pose)
        smooth = self._smooth(pose)
        contact = self._contact(pose, world)

        total = (task + w.w_prior * prior + w.w_rom * rom
                 + w.w_smooth * smooth + w.w_contact * contact
                 + w.w_grip * grip + w.w_level * level)
        return LossTerms(total=total, task=task, prior=prior, rom=rom,
                         smooth=smooth, contact=contact, grip=grip, level=level,
                         task_parts=parts)

    # ------------------------------------------------------------------ #
    def calibrate(self, pose: Tensor, ratios: Optional[Dict[str, float]] = None) -> LossWeights:
        """用一次前向的真实数值自动定标三个正则项的基准权重（§4.5）。

        把权重设成"该项在初始点的贡献占任务项的指定比例"，避免手工试凑量纲。
        默认比例：prior 30%、ROM 10%、smooth 10%、contact 40%。

        ``contact`` 比例取得最高，因为脚底打滑是实测中最容易出现、且视觉上最刺眼的
        失败模式（见模块文档）。

        Notes
        -----
        必须在初始点（``pose = ref_pose``）附近调用才有意义。此时任务项数值反映的是
        "目标与参考的差距"，正是我们希望其余项去平衡的对象。
        """
        ratios = ratios or {"prior": 0.3, "rom": 0.1, "smooth": 0.1, "contact": 0.4}
        with torch.no_grad():
            t = self(pose, progress=0.0)
            base = float(t.task)
            for name, term in (("prior", t.prior), ("rom", t.rom),
                               ("smooth", t.smooth), ("contact", t.contact)):
                v = float(term)
                # ROM 在初始点通常恰为 0（参考动作合法），无法定标，保留默认值
                if v > 1e-9:
                    setattr(self.w, f"w_{name}", base * ratios[name] / v)
        return self.w
