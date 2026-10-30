"""身体级碰撞约束:胶囊体近似 + 参考动作自标定间隙。

为什么需要单独一层
------------------
姿态合法性有三层约束,**互不替代**:

1. **关节级**(:mod:`motionedit.rom`):单个关节能转多少。是**局部**性质。
2. **身体级**(本模块):身体各部分不能占据同一空间。是**全局**性质。
3. 统计级:姿态像不像人做得出来。目前只有退化版(``prior`` 拉向 A_ref)。

**没有任何局部约束能蕴含全局无碰撞**——每个关节都在合法范围内,膝盖仍可穿过
另一条大腿。exp07 round 4 实测:髋 ROM 限额 125°,实际合成 swing 不到 30°
(``rom=0.004``),而双膝间距从 A_ref 的 14.1 cm 塌到 1.1 cm。ROM 不是失效,
是从来没有这个能力——``max_swing`` 把屈伸与内收外展合并成一个标量,分不出方向。

为什么用胶囊而不是网格
----------------------
12711 顶点 × 上百帧 × 上千次迭代做网格相交,既太慢又不可微(三角形相交是离散的)。
胶囊-胶囊距离有闭式解(两线段最短距离减两半径),可微、可批量;约 20 个胶囊
不到 200 对,相比 65 根骨骼的 Python FK 循环可以忽略。

胶囊**从蒙皮权重自动拟合**,不是手调半径:取以该骨骼为主导影响的顶点,
轴向由骨架几何给出(指向子骨骼),半径取垂距分位数。

间隙为什么必须自标定
--------------------
相邻部位本来就该接触(大腿与小腿在膝盖处必然重叠)。若手写一张"允许碰撞表",
就回到了缝缝补补。改为从参考动作标定::

    margin_ij = min over A_ref frames of ( d_ij - r_i - r_j )
    penalty   = relu( margin_ij - slack - clearance_ij )

A_ref 上**恒为零,由构造保证**;只惩罚"比参考动作挤得更厉害"。这与 ``level``
(只罚比 A_ref 倾斜更多)、``grip``(以 1.84 mm 残差为基准)是同一个模式。

父子骨骼对被排除:它们的胶囊在共享关节处必然重叠,而弯曲该关节会合法地加深
重叠——罚它等于罚正常屈曲。这个排除来自运动学树(是数据),不是手写白名单。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import torch
from torch import Tensor

from .skeleton import CORE_BONES, Skeleton

__all__ = [
    "CapsuleSet",
    "CollisionPenalty",
    "fit_capsules",
    "segment_distance",
    "build_collision_penalty",
]

_EPS = 1e-8


# --------------------------------------------------------------------------- #
def segment_distance(p1: Tensor, q1: Tensor, p2: Tensor, q2: Tensor) -> Tensor:
    """两线段间最短距离,``[..., 3]`` 批量输入,返回 ``[...]``。

    标准的三次钳位解法(Ericson, *Real-Time Collision Detection* §5.1.9):
    先在无约束解上求参数,钳到 ``[0,1]``,再互相回代各钳一次。钳位使函数分段光滑,
    梯度几乎处处存在,对一阶/二阶优化器都够用。
    """
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a = (d1 * d1).sum(-1).clamp_min(_EPS)
    e = (d2 * d2).sum(-1).clamp_min(_EPS)
    b = (d1 * d2).sum(-1)
    c = (d1 * r).sum(-1)
    f = (d2 * r).sum(-1)

    denom = a * e - b * b
    s = torch.where(denom > _EPS,
                    ((b * f - c * e) / denom.clamp_min(_EPS)).clamp(0.0, 1.0),
                    torch.zeros_like(a))
    t = ((b * s + f) / e).clamp(0.0, 1.0)
    s = ((t * b - c) / a).clamp(0.0, 1.0)

    diff = (p1 + s.unsqueeze(-1) * d1) - (p2 + t.unsqueeze(-1) * d2)
    return diff.norm(dim=-1)


# --------------------------------------------------------------------------- #
@dataclass
class CapsuleSet:
    """绑定在骨骼上的胶囊体集合。端点存在**骨骼局部坐标系**下。"""

    names: List[str]
    bone_idx: Tensor             # [K] 所属骨骼下标
    local_a: Tensor              # [K, 3] 轴端点 A（局部）
    local_b: Tensor              # [K, 3] 轴端点 B（局部）
    radius: Tensor               # [K]

    @property
    def count(self) -> int:
        return len(self.names)

    def to(self, device: torch.device | str) -> "CapsuleSet":
        device = torch.device(device)
        if device == self.local_a.device:
            return self
        return CapsuleSet(
            names=list(self.names),
            bone_idx=self.bone_idx.to(device),
            local_a=self.local_a.to(device),
            local_b=self.local_b.to(device),
            radius=self.radius.to(device),
        )

    def world_endpoints(self, world: Tensor) -> Tuple[Tensor, Tensor]:
        """``world`` 为 ``[T,B,4,4]`` 的骨骼世界变换,返回两组 ``[T,K,3]`` 端点。"""
        m = world[:, self.bone_idx]                                   # [T,K,4,4]
        rot, trans = m[..., :3, :3], m[..., :3, 3]
        a = torch.einsum("tkij,kj->tki", rot, self.local_a) + trans
        b = torch.einsum("tkij,kj->tki", rot, self.local_b) + trans
        return a, b

    def summary(self) -> List[dict]:
        """逐胶囊的长度与半径(厘米),用于人工核对解剖学合理性。"""
        length = (self.local_b - self.local_a).norm(dim=-1)
        return [{"name": n,
                 "length_cm": round(float(length[k]) * 100, 1),
                 "radius_cm": round(float(self.radius[k]) * 100, 1)}
                for k, n in enumerate(self.names)]


# --------------------------------------------------------------------------- #
def _principal_axis(x: Tensor) -> Tensor:
    """点云最大方差方向,用于没有子骨骼(手、脚尖、头)时的轴向兜底。"""
    c = x - x.mean(0, keepdim=True)
    _, _, v = torch.linalg.svd(c.double(), full_matrices=False)
    return v[0].to(x.dtype)


def fit_capsules(
    skeleton_json: str | Path,
    skel: Skeleton,
    bones: Sequence[str] = CORE_BONES,
    dominant_weight: float = 0.5,
    min_vertices: int = 20,
    radius_quantile: float = 0.75,
) -> CapsuleSet:
    """从蒙皮权重自动拟合胶囊体。

    每个顶点归给**权重最大**的那根骨骼(且该权重需达到 ``dominant_weight``),
    于是同一块肌肉不会被两根骨骼重复计入。轴向优先取骨架几何给出的
    "指向子骨骼"方向(多个子骨骼时取平均),无子骨骼时退化为点云主轴。

    ``radius_quantile`` 取 0.75 而非最大值:少数外延顶点(如指尖、衣褶)会把半径
    撑得过大,使胶囊比身体胖一圈,反而降低灵敏度。
    """
    with open(skeleton_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    verts = torch.tensor(data["vertices"], dtype=skel.rest.dtype)     # [N,3]
    influences = data["bone_influences"]

    # 顶点 → 主导骨骼
    owner: dict[str, List[int]] = {}
    for vid, infl in influences.items():
        if not infl:
            continue
        best = max(infl, key=lambda d: d["weight"])
        if best["weight"] < dominant_weight:
            continue
        owner.setdefault(best["bone"], []).append(int(vid))

    name_set = set(skel.bone_names)
    names: List[str] = []
    idx_list: List[int] = []
    a_list: List[Tensor] = []
    b_list: List[Tensor] = []
    r_list: List[float] = []

    for bone in bones:
        if bone not in name_set:
            continue
        vids = owner.get(bone, [])
        if len(vids) < min_vertices:
            continue
        bi = skel.index(bone)

        # 世界静止坐标 → 骨骼局部坐标
        pts = verts[torch.tensor(vids, dtype=torch.long)]
        local = pts @ skel.rest_inv[bi][:3, :3].T + skel.rest_inv[bi][:3, 3]

        children = [c for c in range(skel.num_bones)
                    if int(skel.parent_idx[c]) == bi]
        if children:
            # 子骨骼原点在本骨骼局部系中的位置即长轴方向
            dirs = []
            for c in children:
                d = (skel.rest_inv[bi] @ skel.rest[c])[:3, 3]
                if float(d.norm()) > 1e-4:
                    dirs.append(d / d.norm())
            axis = (torch.stack(dirs).mean(0) if dirs else _principal_axis(local))
        else:
            axis = _principal_axis(local)
        n = float(axis.norm())
        axis = axis / n if n > _EPS else _principal_axis(local)

        center = local.mean(0)
        rel = local - center
        t = rel @ axis
        perp = (rel - t.unsqueeze(-1) * axis).norm(dim=-1)
        radius = float(torch.quantile(perp, radius_quantile))

        names.append(bone)
        idx_list.append(bi)
        a_list.append(center + float(t.min()) * axis)
        b_list.append(center + float(t.max()) * axis)
        r_list.append(max(radius, 1e-3))

    if not names:
        raise ValueError("未能拟合出任何胶囊体：检查蒙皮权重与骨骼名是否匹配")

    dev, dt = skel.rest.device, skel.rest.dtype
    return CapsuleSet(
        names=names,
        bone_idx=torch.tensor(idx_list, dtype=torch.long, device=dev),
        local_a=torch.stack(a_list).to(device=dev, dtype=dt),
        local_b=torch.stack(b_list).to(device=dev, dtype=dt),
        radius=torch.tensor(r_list, device=dev, dtype=dt),
    )


# --------------------------------------------------------------------------- #
def _tree_hops(skel: Skeleton, i: int, j: int) -> int:
    """两根骨骼在运动学树上的距离(父子为 1)。"""
    def chain(b: int) -> List[int]:
        out = []
        while b >= 0:
            out.append(b)
            b = int(skel.parent_idx[b])
        return out[::-1]

    ci, cj = chain(i), chain(j)
    k = 0
    while k < min(len(ci), len(cj)) and ci[k] == cj[k]:
        k += 1
    return (len(ci) - k) + (len(cj) - k)


@dataclass
class CollisionPenalty:
    """逐帧的肢间穿插惩罚,间隙基准由参考动作标定。"""

    caps: CapsuleSet
    pair_i: Tensor               # [P]
    pair_j: Tensor               # [P]
    margin: Tensor               # [P] 允许的最小间隙(米,可为负)
    deg_per_m: float = 200.0     # 米 → 度 的换算,与 grip 同口径(1° ≙ 5 mm)

    @property
    def num_pairs(self) -> int:
        return int(self.pair_i.shape[0])

    def to(self, device: torch.device | str) -> "CollisionPenalty":
        device = torch.device(device)
        if device == self.margin.device:
            return self
        return CollisionPenalty(
            caps=self.caps.to(device),
            pair_i=self.pair_i.to(device),
            pair_j=self.pair_j.to(device),
            margin=self.margin.to(device),
            deg_per_m=self.deg_per_m,
        )

    def clearance(self, world: Tensor) -> Tensor:
        """``[T,P]`` 逐帧逐对的间隙(米)。负值表示胶囊已重叠。"""
        a, b = self.caps.world_endpoints(world)
        d = segment_distance(a[:, self.pair_i], b[:, self.pair_i],
                             a[:, self.pair_j], b[:, self.pair_j])
        return d - self.caps.radius[self.pair_i] - self.caps.radius[self.pair_j]

    def __call__(self, world: Tensor) -> Tensor:
        # 对数求和、对帧求均值：对数取均值会把单对的严重穿插稀释掉上百倍
        viol = torch.relu(self.margin - self.clearance(world)) * self.deg_per_m
        return viol.pow(2).sum(-1).mean()

    def worst(self, world: Tensor) -> List[dict]:
        """违规最重的若干对,用于诊断。"""
        with torch.no_grad():
            deficit = (self.margin - self.clearance(world)).max(dim=0).values
        order = torch.argsort(deficit, descending=True)
        out = []
        for p in order[:8].tolist():
            if float(deficit[p]) <= 0:
                break
            out.append({
                "pair": f"{self.caps.names[int(self.pair_i[p])]} × "
                        f"{self.caps.names[int(self.pair_j[p])]}",
                "margin_cm": round(float(self.margin[p]) * 100, 1),
                "deficit_cm": round(float(deficit[p]) * 100, 1),
            })
        return out


def build_collision_penalty(
    caps: CapsuleSet,
    skel: Skeleton,
    ref_world: Tensor,
    exclude_hops: int = 1,
    slack_m: float = 0.02,
    deg_per_m: float = 200.0,
) -> CollisionPenalty:
    """用参考动作标定各对胶囊的允许间隙。

    ``exclude_hops=1`` 排除父子对:它们的胶囊在共享关节处必然重叠,而弯曲该关节
    会合法地加深重叠。

    ``slack_m`` 是“允许比参考动作多侵入多少”。取 2cm 而非更小值:
    步态中左右脚会**擦身而过**（A_ref 实测最小间隙 0.0cm、左右小腿 −0.2cm）,
    余量太紧会让这些对在任何微小扰动下立即激活,全程产生排斥力——
    实测 5mm 余量会把恒等参数下的逐骨骼偏离从 2.4° 推到 5.8°。
    本项要拦的是**总量穿插**（round 4 的胶囊重叠达 9.6cm）,不是去维持
    参考动作的毫米级间隙。平方 hinge 保证大违规仍然压倒性地占主。
    """
    k = caps.count
    pi, pj = [], []
    for i in range(k):
        for j in range(i + 1, k):
            if _tree_hops(skel, int(caps.bone_idx[i]), int(caps.bone_idx[j])) <= exclude_hops:
                continue
            pi.append(i)
            pj.append(j)

    dev = ref_world.device
    pen = CollisionPenalty(
        caps=caps.to(dev),
        pair_i=torch.tensor(pi, dtype=torch.long, device=dev),
        pair_j=torch.tensor(pj, dtype=torch.long, device=dev),
        margin=torch.zeros(len(pi), device=dev, dtype=ref_world.dtype),
        deg_per_m=deg_per_m,
    )
    with torch.no_grad():
        pen.margin = pen.clearance(ref_world).min(dim=0).values - slack_m
    return pen
